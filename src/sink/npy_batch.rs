//! NumPy 批量数据输出适配器
//!
//! 专门用于输出批量训练数据的 NumPy 格式文件。
//! 支持输出 (B, T, H, W, C) 或 (B, T, state_dim) 形状的张量数据。
//!
//! 与 BatchCollector 配合使用，接收 label="batch_data" 标记的批量帧。

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::core::*;

use super::adapter::{format_output_filename, OutputConfig, OutputStats, SinkAdapter};

/// NPY 文件魔数
const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";
/// NPY 版本 2.0（支持 4 字节 header_len，可处理大 header）
const NPY_VERSION: [u8; 2] = [0x02, 0x00];

/// NpyBatchAdapter 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpyBatchConfig {
    /// 批量大小（样本数量）
    pub batch_size: usize,
    /// 每个样本的帧数
    pub num_frames: usize,
    /// 网格高度（可选，用于设置 shape）
    #[serde(default)]
    pub rows: Option<usize>,
    /// 网格宽度（可选，用于设置 shape）
    #[serde(default)]
    pub cols: Option<usize>,
    /// 通道数（可选，用于设置 shape）
    #[serde(default)]
    pub channels: Option<usize>,
}

/// NPY 数据类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NpyDtype {
    /// 小端序 int32（匹配 Python JAX array）
    Int32,
    /// 小端序 int64
    Int64,
    /// 小端序 float64
    Float64,
}

impl NpyDtype {
    /// 返回 NPY dtype 描述字符串
    fn descr(&self) -> &'static str {
        match self {
            NpyDtype::Int32 => "<i4",
            NpyDtype::Int64 => "<i8",
            NpyDtype::Float64 => "<f8",
        }
    }

    /// 每个元素的字节数
    fn element_size(&self) -> usize {
        match self {
            NpyDtype::Int32 => 4,
            NpyDtype::Int64 => 8,
            NpyDtype::Float64 => 8,
        }
    }

    /// 从首个 FrameState 推断 dtype
    fn from_first_value(value: &FrameState) -> Self {
        match value {
            FrameState::Integer(_) => NpyDtype::Int32, // 默认使用 int32 匹配 Python
            FrameState::Float(_) => NpyDtype::Float64,
            FrameState::Bool(_) => NpyDtype::Int32,
        }
    }
}

/// NumPy 批量数据输出适配器
///
/// 专门用于输出批量训练数据，支持 (B, T, H, W, C) 形状。
pub struct NpyBatchAdapter {
    /// 配置
    config: NpyBatchConfig,
    /// 最终输出文件路径
    output_path: Option<PathBuf>,
    /// 临时数据文件路径
    tmp_path: Option<PathBuf>,
    /// 暂存数据文件的写入器
    writer: Option<BufWriter<File>>,
    /// 已写入批次数
    batches_written: u64,
    /// 数据类型（首帧确定）
    dtype: Option<NpyDtype>,
    /// 已写入字节数（仅数据部分）
    bytes_written: u64,
}

impl NpyBatchAdapter {
    /// 创建新的 NpyBatchAdapter 实例
    pub fn new(config: NpyBatchConfig) -> Self {
        NpyBatchAdapter {
            config,
            output_path: None,
            tmp_path: None,
            writer: None,
            batches_written: 0,
            dtype: None,
            bytes_written: 0,
        }
    }

    /// 将帧数据写入暂存文件（仅原始二进制数据，无 header）
    fn write_batch_data(
        writer: &mut BufWriter<File>,
        frame: &SequenceFrame,
        dtype: NpyDtype,
    ) -> CoreResult<()> {
        for state in &frame.state.values {
            match dtype {
                NpyDtype::Int32 => {
                    let v = match state {
                        FrameState::Integer(v) => *v as i32,
                        FrameState::Float(v) => *v as i32,
                        FrameState::Bool(v) => *v as i32,
                    };
                    writer.write_all(&v.to_le_bytes())?;
                }
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
            }
        }
        Ok(())
    }

    /// 构建 NPY v2.0 header
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

        // NPY v2.0: prefix = magic(6) + version(2) + header_len(4) = 12
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
        let header_len = padded_header.len() as u32;  // NPY v2.0 使用 u32

        let mut result = Vec::with_capacity(prefix_len + padded_header.len());
        result.extend_from_slice(NPY_MAGIC);
        result.extend_from_slice(&NPY_VERSION);
        result.extend_from_slice(&header_len.to_le_bytes());  // 4 字节
        result.extend_from_slice(padded_header.as_bytes());

        result
    }

    /// 计算输出 shape
    fn compute_shape(&self, total_elements: usize) -> Vec<usize> {
        // 如果没有数据，返回默认 shape
        if total_elements == 0 || self.batches_written == 0 {
            if let (Some(rows), Some(cols), Some(channels)) =
                (self.config.rows, self.config.cols, self.config.channels)
            {
                return vec![0, self.config.num_frames, rows, cols, channels];
            } else {
                return vec![0, self.config.num_frames, 0];
            }
        }

        // 如果配置了网格形状，使用 (B, T, H, W, C)
        if let (Some(rows), Some(cols), Some(channels)) =
            (self.config.rows, self.config.cols, self.config.channels)
        {
            // 每个 frame 包含 num_frames * rows * cols * channels 个元素（一个样本）
            // batches_written 表示写入的样本数量
            // 从实际元素数计算样本数，确保 shape 与数据量匹配
            let elements_per_sample = self.config.num_frames * rows * cols * channels;
            let actual_samples = total_elements / elements_per_sample;
            vec![
                actual_samples,
                self.config.num_frames,
                rows,
                cols,
                channels,
            ]
        } else {
            // 否则使用 (B, T, state_dim) 形状
            // 从总元素数推断 shape
            let elements_per_sample = self.config.num_frames;
            let actual_samples = total_elements / elements_per_sample;
            let state_dim = if actual_samples > 0 {
                total_elements / (actual_samples * self.config.num_frames)
            } else {
                0
            };
            vec![
                actual_samples,
                self.config.num_frames,
                state_dim,
            ]
        }
    }
}

impl SinkAdapter for NpyBatchAdapter {
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
        self.batches_written = 0;
        self.dtype = None;
        self.bytes_written = 0;

        Ok(())
    }

    fn write_frame(&mut self, frame: &SequenceFrame) -> CoreResult<()> {
        // 检查是否是批量数据帧（label="batch_data"）
        if frame.label.as_deref() != Some("batch_data") {
            // 如果不是批量数据帧，直接写入（兼容普通帧）
            return self.write_single_frame(frame);
        }

        // 首帧：确定 dtype
        if self.dtype.is_none() {
            if let Some(first_val) = frame.state.values.first() {
                self.dtype = Some(NpyDtype::from_first_value(first_val));
            } else {
                self.dtype = Some(NpyDtype::Int32); // 空帧默认
            }
        }

        let dtype = self.dtype.unwrap();
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| CoreError::SinkError("适配器未打开".into()))?;

        Self::write_batch_data(writer, frame, dtype)?;

        self.bytes_written += (frame.state.values.len() * dtype.element_size()) as u64;
        self.batches_written += 1;

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

        let dtype = self.dtype.unwrap_or(NpyDtype::Int32);

        // 计算 shape
        let total_elements = self.bytes_written as usize / dtype.element_size();
        let shape = self.compute_shape(total_elements);

        // 构建 NPY header
        let header = Self::build_npy_header(dtype, &shape);

        // 创建最终 .npy 文件：header + 数据
        let mut npy_file = File::create(&output_path)?;
        npy_file.write_all(&header)?;

        // 从暂存文件拷贝数据
        if self.batches_written > 0 {
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
            frames_written: self.batches_written,
            bytes_written: file_bytes,
            output_path: Some(output_path),
            file_hash: None,
        })
    }
}

impl NpyBatchAdapter {
    /// 写入单个帧（非批量数据）
    fn write_single_frame(&mut self, frame: &SequenceFrame) -> CoreResult<()> {
        // 首帧：确定 dtype
        if self.dtype.is_none() {
            if let Some(first_val) = frame.state.values.first() {
                self.dtype = Some(NpyDtype::from_first_value(first_val));
            } else {
                self.dtype = Some(NpyDtype::Int32);
            }
        }

        let dtype = self.dtype.unwrap();
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| CoreError::SinkError("适配器未打开".into()))?;

        Self::write_batch_data(writer, frame, dtype)?;

        self.bytes_written += (frame.state.values.len() * dtype.element_size()) as u64;
        self.batches_written += 1;

        Ok(())
    }
}

/// 工厂函数：从 JSON 配置创建 NpyBatchAdapter
pub fn create_npy_batch_adapter(config: &serde_json::Value) -> CoreResult<Box<dyn SinkAdapter>> {
    let config: NpyBatchConfig = if config.is_null() {
        return Err(CoreError::InvalidParams(
            "npy_batch requires configuration (batch_size, num_frames)".into(),
        ));
    } else {
        serde_json::from_value(config.clone()).map_err(|e| {
            CoreError::SerializationError(format!("npy_batch 配置解析失败: {}", e))
        })?
    };

    if config.batch_size == 0 {
        return Err(CoreError::InvalidParams(
            "npy_batch: batch_size must be >= 1".into(),
        ));
    }
    if config.num_frames == 0 {
        return Err(CoreError::InvalidParams(
            "npy_batch: num_frames must be >= 1".into(),
        ));
    }

    Ok(Box::new(NpyBatchAdapter::new(config)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{FrameData, FrameState, SequenceFrame};
    use tempfile::TempDir;

    fn make_batch_frame(step: u64, values: Vec<i64>) -> SequenceFrame {
        let values: Vec<FrameState> = values.into_iter().map(FrameState::Integer).collect();
        SequenceFrame {
            step_index: step,
            state: FrameData { values },
            label: Some("batch_data".to_string()),
            sample_id: None,
        }
    }

    fn default_output_config() -> OutputConfig {
        OutputConfig::default()
    }

    #[test]
    fn test_npy_batch_basic() {
        let tmp = TempDir::new().unwrap();

        // batch_size=2, num_frames=3 → 每批 6 帧
        let config = NpyBatchConfig {
            batch_size: 2,
            num_frames: 3,
            rows: None,
            cols: None,
            channels: None,
        };

        let mut adapter = NpyBatchAdapter::new(config);
        adapter
            .open(tmp.path(), "batch", 0, 0, &default_output_config())
            .unwrap();

        // 写入一个批量（6 个值）
        let frame = make_batch_frame(0, vec![1, 2, 3, 4, 5, 6]);
        adapter.write_frame(&frame).unwrap();

        let stats = adapter.close().unwrap();
        assert_eq!(stats.frames_written, 1);

        // 验证文件存在
        assert!(stats.output_path.unwrap().exists());
    }

    #[test]
    fn test_npy_batch_shape_with_grid() {
        let tmp = TempDir::new().unwrap();

        // batch_size=2, num_frames=2, rows=3, cols=3, channels=1
        let config = NpyBatchConfig {
            batch_size: 2,
            num_frames: 2,
            rows: Some(3),
            cols: Some(3),
            channels: Some(1),
        };

        let mut adapter = NpyBatchAdapter::new(config);
        adapter
            .open(tmp.path(), "grid", 0, 0, &default_output_config())
            .unwrap();

        // 写入一个批量（batch_size * num_frames * rows * cols * channels = 2*2*3*3*1 = 36 个值）
        let values: Vec<i64> = (0..36).collect();
        let frame = make_batch_frame(0, values);
        adapter.write_frame(&frame).unwrap();

        let stats = adapter.close().unwrap();

        // 验证 NPY header 中的 shape
        let raw = fs::read(&stats.output_path.unwrap()).unwrap();
        // NPY v2.0: header_len 是 4 字节
        let header_len = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let header_str = std::str::from_utf8(&raw[12..12 + header_len]).unwrap();
        // shape 应为 (2, 2, 3, 3, 1) - 2 个样本，每个样本 2 帧，3x3 网格，1 通道
        assert!(header_str.contains("'shape': (2, 2, 3, 3, 1)"));
    }

    #[test]
    fn test_npy_batch_int32_dtype() {
        let tmp = TempDir::new().unwrap();

        let config = NpyBatchConfig {
            batch_size: 1,
            num_frames: 1,
            rows: None,
            cols: None,
            channels: None,
        };

        let mut adapter = NpyBatchAdapter::new(config);
        adapter
            .open(tmp.path(), "int32", 0, 0, &default_output_config())
            .unwrap();

        let frame = make_batch_frame(0, vec![42, -7]);
        adapter.write_frame(&frame).unwrap();

        let stats = adapter.close().unwrap();

        // 验证 dtype 为 int32
        let raw = fs::read(&stats.output_path.unwrap()).unwrap();
        // NPY v2.0: header_len 是 4 字节
        let header_len = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let header_str = std::str::from_utf8(&raw[12..12 + header_len]).unwrap();
        assert!(header_str.contains("'descr': '<i4'"));

        // 验证数据
        let data_start = 12 + header_len;
        let val1 = i32::from_le_bytes(raw[data_start..data_start + 4].try_into().unwrap());
        assert_eq!(val1, 42);
        let val2 = i32::from_le_bytes(raw[data_start + 4..data_start + 8].try_into().unwrap());
        assert_eq!(val2, -7);
    }

    #[test]
    fn test_npy_batch_empty() {
        let tmp = TempDir::new().unwrap();

        let config = NpyBatchConfig {
            batch_size: 2,
            num_frames: 3,
            rows: None,
            cols: None,
            channels: None,
        };

        let mut adapter = NpyBatchAdapter::new(config);
        adapter
            .open(tmp.path(), "empty", 0, 0, &default_output_config())
            .unwrap();

        let stats = adapter.close().unwrap();
        assert_eq!(stats.frames_written, 0);

        // 验证文件存在且有合法 header
        let raw = fs::read(&stats.output_path.unwrap()).unwrap();
        assert_eq!(&raw[0..6], NPY_MAGIC);
    }

    #[test]
    fn test_npy_batch_invalid_batch_size() {
        let config = NpyBatchConfig {
            batch_size: 0,
            num_frames: 3,
            rows: None,
            cols: None,
            channels: None,
        };
        let result = create_npy_batch_adapter(&serde_json::to_value(config).unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_npy_batch_invalid_num_frames() {
        let config = NpyBatchConfig {
            batch_size: 2,
            num_frames: 0,
            rows: None,
            cols: None,
            channels: None,
        };
        let result = create_npy_batch_adapter(&serde_json::to_value(config).unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_npy_batch_atomic_write() {
        let tmp = TempDir::new().unwrap();

        let config = NpyBatchConfig {
            batch_size: 2,
            num_frames: 3,
            rows: None,
            cols: None,
            channels: None,
        };

        let mut adapter = NpyBatchAdapter::new(config);
        adapter
            .open(tmp.path(), "atomic", 0, 0, &default_output_config())
            .unwrap();

        let frame = make_batch_frame(0, vec![1, 2, 3, 4, 5, 6]);
        adapter.write_frame(&frame).unwrap();
        adapter.close().unwrap();

        // 验证没有残留临时文件
        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        for entry in &entries {
            let name = entry.file_name();
            let name_str = name.to_str().unwrap();
            assert!(!name_str.ends_with(".tmp"), "残留临时文件: {}", name_str);
        }
    }
}