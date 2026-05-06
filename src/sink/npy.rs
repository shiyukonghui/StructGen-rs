//! NumPy .npy 格式输出适配器实现
//!
//! 将帧序列以 NumPy .npy 格式写入，方便 Python 端直接用 numpy.load() 加载。
//!
//! 文件格式 (NPY v1.0):
//! - Magic: \x93NUMPY (6 bytes)
//! - Version: 0x01, 0x00 (2 bytes)
//! - Header length: u32 LE (4 bytes)
//! - Padded header: Python dict 字面量，以 \n 结尾，总长度对齐到 64 字节
//! - Data: C-order (row-major) 原始二进制
//!
//! 数据类型映射:
//! - FrameState::Integer → <i8 (int64, little-endian)
//! - FrameState::Float   → <f8 (float64, little-endian)
//! - FrameState::Bool    → |b1 (byte)

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::core::*;

use super::adapter::{format_output_filename, OutputConfig, OutputStats, SinkAdapter};

/// NPY 文件魔数
const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";
/// NPY 版本 1.0
const NPY_VERSION: [u8; 2] = [0x01, 0x00];

/// NPY 数据类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NpyDtype {
    /// 小端序 int64
    Int64,
    /// 小端序 float64
    Float64,
    /// 字节布尔
    Bool,
}

impl NpyDtype {
    /// 返回 NPY dtype 描述字符串
    fn descr(&self) -> &'static str {
        match self {
            NpyDtype::Int64 => "<i8",
            NpyDtype::Float64 => "<f8",
            NpyDtype::Bool => "|b1",
        }
    }

    /// 每个元素的字节数
    fn element_size(&self) -> usize {
        match self {
            NpyDtype::Int64 => 8,
            NpyDtype::Float64 => 8,
            NpyDtype::Bool => 1,
        }
    }

    /// 从首个 FrameState 推断 dtype
    fn from_first_value(value: &FrameState) -> Self {
        match value {
            FrameState::Integer(_) => NpyDtype::Int64,
            FrameState::Float(_) => NpyDtype::Float64,
            FrameState::Bool(_) => NpyDtype::Bool,
        }
    }
}

/// NumPy .npy 输出适配器
///
/// 写入策略：先将帧数据写入暂存文件，close() 时拼接 NPY header + 数据到最终 .npy。
pub struct NpyAdapter {
    /// 最终输出文件路径
    output_path: Option<PathBuf>,
    /// 临时数据文件路径
    tmp_path: Option<PathBuf>,
    /// 暂存数据文件的写入器
    writer: Option<BufWriter<File>>,
    /// 已写入帧数
    frames_written: u64,
    /// 状态维度（首帧确定）
    state_dim: Option<usize>,
    /// 数据类型（首帧确定）
    dtype: Option<NpyDtype>,
    /// 网格形状（可选，配置传入）
    grid_shape: Option<Vec<usize>>,
    /// 已写入字节数（仅数据部分）
    bytes_written: u64,
}

impl NpyAdapter {
    /// 创建新的 NpyAdapter 实例
    pub fn new() -> Self {
        NpyAdapter {
            output_path: None,
            tmp_path: None,
            writer: None,
            frames_written: 0,
            state_dim: None,
            dtype: None,
            grid_shape: None,
            bytes_written: 0,
        }
    }

    /// 将帧数据写入暂存文件（仅原始二进制数据，无 header）
    fn write_frame_data(
        writer: &mut BufWriter<File>,
        frame: &SequenceFrame,
        dtype: NpyDtype,
    ) -> CoreResult<()> {
        for state in &frame.state.values {
            match dtype {
                NpyDtype::Int64 => {
                    let v = match state {
                        FrameState::Integer(v) => *v,
                        FrameState::Float(v) => *v as i64,
                        FrameState::Bool(v) => *v as i64,
                    };
                    writer.write_all(&v.to_le_bytes())?;
                }
                NpyDtype::Float64 => {
                    let v = match state {
                        FrameState::Integer(v) => *v as f64,
                        FrameState::Float(v) => *v,
                        FrameState::Bool(v) => *v as u8 as f64,
                    };
                    writer.write_all(&v.to_le_bytes())?;
                }
                NpyDtype::Bool => {
                    let v = match state {
                        FrameState::Integer(v) => (*v != 0) as u8,
                        FrameState::Float(v) => (*v != 0.0) as u8,
                        FrameState::Bool(v) => *v as u8,
                    };
                    writer.write_all(&[v])?;
                }
            }
        }
        Ok(())
    }

    /// 构建 NPY v1.0 header
    fn build_npy_header(dtype: NpyDtype, shape: &[usize]) -> Vec<u8> {
        let descr = dtype.descr();

        // 构建 shape 元组字符串
        let shape_str = if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            let parts: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
            format!("({})", parts.join(", "))
        };

        let header_dict = format!(
            "{{'descr': '{}', 'fortran_order': False, 'shape': {}}}",
            descr, shape_str
        );

        // NPY v1.0: prefix = magic(6) + version(2) + header_len(4) = 12
        // header_len 包含 padded header（以 \n 结尾）
        // 总长度 (12 + header_len) 必须是 64 的倍数
        let prefix_len = 6 + 2 + 4;

        // 加上 \n，计算需要多少 padding
        let header_with_newline = format!("{}\n", header_dict);
        let total_without_padding = prefix_len + header_with_newline.len();
        let remainder = total_without_padding % 64;
        let padding_needed = if remainder == 0 { 0 } else { 64 - remainder };

        // 在 \n 前插入空格 padding
        let padded_header = format!("{}{}\n", header_dict, " ".repeat(padding_needed));
        let header_len = padded_header.len() as u32;

        let mut result = Vec::with_capacity(prefix_len + padded_header.len());
        result.extend_from_slice(NPY_MAGIC);
        result.extend_from_slice(&NPY_VERSION);
        result.extend_from_slice(&header_len.to_le_bytes());
        result.extend_from_slice(padded_header.as_bytes());

        result
    }
}

impl Default for NpyAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl SinkAdapter for NpyAdapter {
    fn format(&self) -> OutputFormat {
        OutputFormat::Npy
    }

    fn open(
        &mut self,
        base_dir: &Path,
        task_name: &str,
        shard_id: usize,
        seed: u64,
        _config: &OutputConfig,
    ) -> CoreResult<()> {
        fs::create_dir_all(base_dir)?;

        let tmp_name = format_output_filename(task_name, shard_id, seed, "bin.tmp");
        let final_name = format_output_filename(task_name, shard_id, seed, "npy");
        let tmp_path = base_dir.join(&tmp_name);
        let output_path = base_dir.join(&final_name);

        if tmp_path.exists() {
            fs::remove_file(&tmp_path)?;
        }

        let file = File::create(&tmp_path)?;
        let writer = BufWriter::with_capacity(64 * 1024, file);

        self.output_path = Some(output_path);
        self.tmp_path = Some(tmp_path);
        self.writer = Some(writer);
        self.frames_written = 0;
        self.state_dim = None;
        self.dtype = None;
        self.grid_shape = None;
        self.bytes_written = 0;

        Ok(())
    }

    fn write_frame(&mut self, frame: &SequenceFrame) -> CoreResult<()> {
        let state_dim = frame.state.dim();

        // 首帧：确定 state_dim 和 dtype
        if self.state_dim.is_none() {
            if let Some(first_val) = frame.state.values.first() {
                self.dtype = Some(NpyDtype::from_first_value(first_val));
            } else {
                self.dtype = Some(NpyDtype::Int64); // 空帧默认
            }
            self.state_dim = Some(state_dim);
        }

        // 校验 state_dim 一致性
        let expected_dim = self.state_dim.unwrap();
        if state_dim != expected_dim {
            return Err(CoreError::SinkError(format!(
                "帧的 state_dim ({}) 与首帧 ({}) 不一致",
                state_dim, expected_dim
            )));
        }

        let dtype = self.dtype.unwrap();
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| CoreError::SinkError("适配器未打开".into()))?;

        Self::write_frame_data(writer, frame, dtype)?;

        self.bytes_written += (state_dim * dtype.element_size()) as u64;
        self.frames_written += 1;

        Ok(())
    }

    fn close(&mut self) -> CoreResult<OutputStats> {
        // 关闭暂存数据文件
        let writer = self
            .writer
            .take()
            .ok_or_else(|| CoreError::SinkError("适配器未打开或已关闭".into()))?;
        let mut data_file = writer
            .into_inner()
            .map_err(|e| CoreError::SinkError(format!("刷新 BufWriter 失败: {}", e)))?;
        data_file.flush()?;

        let tmp_path = self
            .tmp_path
            .as_ref()
            .ok_or_else(|| CoreError::SinkError("临时路径为空".into()))?
            .clone();
        let output_path = self
            .output_path
            .as_ref()
            .ok_or_else(|| CoreError::SinkError("输出路径为空".into()))?
            .clone();

        let dtype = self.dtype.unwrap_or(NpyDtype::Int64);
        let state_dim = self.state_dim.unwrap_or(0);

        // 计算 shape
        let shape: Vec<usize> = if let Some(ref grid_shape) = self.grid_shape {
            let mut s = vec![self.frames_written as usize];
            s.extend_from_slice(grid_shape);
            s
        } else {
            vec![self.frames_written as usize, state_dim]
        };

        // 构建 NPY header
        let header = Self::build_npy_header(dtype, &shape);

        // 创建最终 .npy 文件：header + 数据
        let mut npy_file = File::create(&output_path)?;
        npy_file.write_all(&header)?;

        // 从暂存文件拷贝数据
        if self.frames_written > 0 {
            drop(data_file); // 关闭暂存文件
            let mut src = BufReader::new(File::open(&tmp_path)?);
            std::io::copy(&mut src, &mut npy_file)?;
        }
        npy_file.flush()?;

        // 删除暂存文件
        if tmp_path.exists() {
            let _ = fs::remove_file(&tmp_path);
        }

        let file_bytes = header.len() as u64 + self.bytes_written;

        Ok(OutputStats {
            frames_written: self.frames_written,
            bytes_written: file_bytes,
            output_path: Some(output_path),
            file_hash: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{FrameData, FrameState, SequenceFrame};
    use tempfile::TempDir;

    fn make_test_frames(n: usize) -> Vec<SequenceFrame> {
        (0..n)
            .map(|i| {
                let values: Vec<FrameState> = vec![
                    FrameState::Integer(i as i64 * 100),
                    FrameState::Integer(i as i64 * 200),
                    FrameState::Integer(i as i64 * 300),
                ];
                let data: FrameData = values.into_iter().collect();
                SequenceFrame::new(i as u64, data)
            })
            .collect()
    }

    fn default_config() -> OutputConfig {
        OutputConfig::default()
    }

    #[test]
    fn test_npy_header_format() {
        let header = NpyAdapter::build_npy_header(NpyDtype::Int64, &[5, 3]);

        // 验证 magic
        assert_eq!(&header[0..6], NPY_MAGIC);
        // 验证 version
        assert_eq!(header[6], 0x01);
        assert_eq!(header[7], 0x00);
        // 验证 header_len
        let header_len = u32::from_le_bytes(header[8..12].try_into().unwrap());
        assert!(header_len > 0);
        // 总长度应是 64 的倍数
        assert_eq!((12 + header_len as usize) % 64, 0);
        // header 应包含 Python dict
        let header_str = std::str::from_utf8(&header[12..12 + header_len as usize]).unwrap();
        assert!(header_str.contains("'descr': '<i8'"));
        assert!(header_str.contains("'fortran_order': False"));
        assert!(header_str.contains("'shape': (5, 3)"));
    }

    #[test]
    fn test_npy_shape_correct() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(5);

        let mut adapter = NpyAdapter::new();
        adapter
            .open(tmp.path(), "shape", 1, 0x42, &default_config())
            .unwrap();
        for f in &frames {
            adapter.write_frame(f).unwrap();
        }
        let stats = adapter.close().unwrap();

        let raw = fs::read(&stats.output_path.unwrap()).unwrap();

        // 解析 header_len
        let header_len = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let header_str = std::str::from_utf8(&raw[12..12 + header_len]).unwrap();
        assert!(header_str.contains("'shape': (5, 3)"));
    }

    #[test]
    fn test_npy_int64_data() {
        let tmp = TempDir::new().unwrap();

        let values: Vec<FrameState> = vec![FrameState::Integer(42), FrameState::Integer(-7)];
        let data: FrameData = values.into_iter().collect();
        let frame = SequenceFrame::new(0, data);

        let mut adapter = NpyAdapter::new();
        adapter
            .open(tmp.path(), "int64", 0, 0, &default_config())
            .unwrap();
        adapter.write_frame(&frame).unwrap();
        let stats = adapter.close().unwrap();

        let raw = fs::read(&stats.output_path.unwrap()).unwrap();
        let header_len = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let data_start = 12 + header_len;

        // 第一个值: Integer(42) as i64 LE
        let val1 = i64::from_le_bytes(raw[data_start..data_start + 8].try_into().unwrap());
        assert_eq!(val1, 42);

        // 第二个值: Integer(-7) as i64 LE
        let val2 = i64::from_le_bytes(raw[data_start + 8..data_start + 16].try_into().unwrap());
        assert_eq!(val2, -7);
    }

    #[test]
    fn test_npy_float64_data() {
        let tmp = TempDir::new().unwrap();

        let values: Vec<FrameState> = vec![FrameState::Float(3.14), FrameState::Float(-2.5)];
        let data: FrameData = values.into_iter().collect();
        let frame = SequenceFrame::new(0, data);

        let mut adapter = NpyAdapter::new();
        adapter
            .open(tmp.path(), "f64", 0, 0, &default_config())
            .unwrap();
        adapter.write_frame(&frame).unwrap();
        let stats = adapter.close().unwrap();

        let raw = fs::read(&stats.output_path.unwrap()).unwrap();
        let header_len = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let data_start = 12 + header_len;

        let val1 = f64::from_le_bytes(raw[data_start..data_start + 8].try_into().unwrap());
        assert!((val1 - 3.14).abs() < 1e-10);
        let val2 = f64::from_le_bytes(raw[data_start + 8..data_start + 16].try_into().unwrap());
        assert!((val2 - (-2.5)).abs() < 1e-10);
    }

    #[test]
    fn test_npy_empty_no_frames() {
        let tmp = TempDir::new().unwrap();

        let mut adapter = NpyAdapter::new();
        adapter
            .open(tmp.path(), "empty", 0, 0, &default_config())
            .unwrap();
        let stats = adapter.close().unwrap();

        assert_eq!(stats.frames_written, 0);

        let raw = fs::read(&stats.output_path.unwrap()).unwrap();
        // 即使零帧也应写入合法的 NPY header
        assert_eq!(&raw[0..6], NPY_MAGIC);
    }

    #[test]
    fn test_npy_dimension_mismatch() {
        let tmp = TempDir::new().unwrap();

        let mut adapter = NpyAdapter::new();
        adapter
            .open(tmp.path(), "dim_err", 0, 0, &default_config())
            .unwrap();

        let values1: Vec<FrameState> = vec![FrameState::Integer(1), FrameState::Integer(2)];
        let data1: FrameData = values1.into_iter().collect();
        adapter.write_frame(&SequenceFrame::new(0, data1)).unwrap();

        let values2: Vec<FrameState> =
            vec![FrameState::Integer(1), FrameState::Integer(2), FrameState::Integer(3)];
        let data2: FrameData = values2.into_iter().collect();
        let result = adapter.write_frame(&SequenceFrame::new(1, data2));

        assert!(result.is_err());
    }

    #[test]
    fn test_npy_atomic_write() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(3);

        let mut adapter = NpyAdapter::new();
        adapter
            .open(tmp.path(), "atomic", 0, 0, &default_config())
            .unwrap();
        for f in &frames {
            adapter.write_frame(f).unwrap();
        }
        adapter.close().unwrap();

        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        for entry in &entries {
            let name = entry.file_name();
            let name_str = name.to_str().unwrap();
            assert!(
                !name_str.ends_with(".tmp"),
                "残留临时文件: {}",
                name_str
            );
        }
    }

    #[test]
    fn test_npy_header_alignment_64() {
        // 验证不同 shape 下的对齐
        let shapes: Vec<&[usize]> = vec![&[1, 1], &[100, 12, 12], &[1000, 144]];
        for shape in shapes {
            let header = NpyAdapter::build_npy_header(NpyDtype::Int64, shape);
            let header_len = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
            assert_eq!(
                (12 + header_len) % 64,
                0,
                "Total header not 64-byte aligned for shape {:?}",
                shape
            );
        }
    }
}
