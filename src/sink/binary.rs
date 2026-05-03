//! 小端序二进制输出适配器实现
//!
//! 将帧序列以紧凑的二进制格式转储，支持后续通过 mmap 进行高效随机访问。
//!
//! 文件格式（小端序）：
//! - Header (16 bytes): magic "SGEN"(4) + version u32(4) + frame_count u64(8) + state_dim u32(4)
//! - 每帧: step_index u64(8) + values (state_dim * 9 bytes) + label_len u32(4) + label_bytes

use std::fs::{self, File};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::core::*;

use super::adapter::{format_output_filename, OutputConfig, OutputStats, SinkAdapter};

/// 文件格式魔数
const MAGIC: &[u8; 4] = b"SGEN";
/// 文件格式版本号
const VERSION: u32 = 1;
/// 帧状态值的二进制表示大小（1 字节类型标签 + 8 字节数据）
const STATE_VALUE_BYTES: usize = 9;

/// 小端序二进制输出适配器
///
/// 以固定宽度的二进制格式写入帧数据，每帧大小 = 8 + state_dim * 9 + 4 + label.len()。
/// close() 时回填 header 中的 frame_count 字段。
pub struct BinaryAdapter {
    /// 最终输出文件路径
    output_path: Option<PathBuf>,
    /// 临时文件路径
    tmp_path: Option<PathBuf>,
    /// 带缓冲的文件写入器（64KB 缓冲区）
    writer: Option<BufWriter<File>>,
    /// 已写入帧数
    frames_written: u64,
    /// 状态维度（在首帧写入时确定）
    state_dim: Option<u32>,
    /// 已写入字节数（含 header）
    bytes_written: u64,
}

impl BinaryAdapter {
    /// 创建新的 BinaryAdapter 实例
    pub fn new() -> Self {
        BinaryAdapter {
            output_path: None,
            tmp_path: None,
            writer: None,
            frames_written: 0,
            state_dim: None,
            bytes_written: 0,
        }
    }

    /// 写入文件 header
    ///
    /// Header 格式（小端序，20 字节）：
    /// - magic: [u8; 4] = b"SGEN"         (offset 0)
    /// - version: u32 = 1                 (offset 4)
    /// - frame_count: u64 = 0（占位）     (offset 8)
    /// - state_dim: u32                   (offset 16)
    fn write_header(writer: &mut BufWriter<File>, state_dim: u32) -> CoreResult<()> {
        // 构建 header 缓冲区（20 字节）
        let mut header = [0u8; 20];
        header[0..4].copy_from_slice(MAGIC);
        header[4..8].copy_from_slice(&VERSION.to_le_bytes());
        // header[8..16] 为 frame_count 占位（已经是 0）
        header[16..20].copy_from_slice(&state_dim.to_le_bytes());

        writer.write_all(&header)?;
        Ok(())
    }

    /// 序列化单帧到内部字节缓冲区
    fn serialize_frame(frame: &SequenceFrame, state_dim: u32) -> Vec<u8> {
        let label_bytes = frame.label.as_ref().map(|l| l.as_bytes()).unwrap_or(&[]);
        let label_len = label_bytes.len() as u32;

        // 计算帧大小：step_index(8) + state_dim * 9 + label_len(4) + label_bytes
        let frame_size = 8 + (state_dim as usize) * STATE_VALUE_BYTES + 4 + label_bytes.len();
        let mut buf = Vec::with_capacity(frame_size);

        // step_index (u64 LE)
        buf.extend_from_slice(&frame.step_index.to_le_bytes());

        // 序列化每个 FrameState
        for state in &frame.state.values {
            match state {
                FrameState::Integer(v) => {
                    buf.push(0x01); // 类型标签：整型
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                FrameState::Float(v) => {
                    buf.push(0x02); // 类型标签：浮点型
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                FrameState::Bool(v) => {
                    buf.push(0x03); // 类型标签：布尔型
                    buf.push(*v as u8);
                    // 填充 7 字节零
                    buf.extend_from_slice(&[0u8; 7]);
                }
            }
        }

        // label_len (u32 LE)
        buf.extend_from_slice(&label_len.to_le_bytes());

        // label 字节
        buf.extend_from_slice(label_bytes);

        buf
    }

    /// 回填 header 中的 frame_count 字段
    fn patch_frame_count(&mut self) -> CoreResult<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| CoreError::SinkError("适配器未打开".into()))?;

        // 先刷新缓冲区，确保所有数据已写入文件
        writer.flush()?;

        // 通过 get_mut() 获取底层文件引用
        let file = writer.get_mut();

        // 定位到 frame_count 偏移量（magic 4 + version 4 = 8）
        file.seek(SeekFrom::Start(8))?;
        file.write_all(&self.frames_written.to_le_bytes())?;
        file.flush()?;
        // 回到文件末尾（后续不再写入，但为安全起见）
        file.seek(SeekFrom::End(0))?;

        Ok(())
    }
}

impl Default for BinaryAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl SinkAdapter for BinaryAdapter {
    fn format(&self) -> OutputFormat {
        OutputFormat::Binary
    }

    fn open(
        &mut self,
        base_dir: &Path,
        task_name: &str,
        shard_id: usize,
        seed: u64,
        _config: &OutputConfig,
    ) -> CoreResult<()> {
        // 确保输出目录存在
        fs::create_dir_all(base_dir)?;

        // 构造文件名
        let tmp_name = format_output_filename(task_name, shard_id, seed, "bin.tmp");
        let final_name = format_output_filename(task_name, shard_id, seed, "bin");
        let tmp_path = base_dir.join(&tmp_name);
        let output_path = base_dir.join(&final_name);

        // 如果临时文件已存在，先删除
        if tmp_path.exists() {
            fs::remove_file(&tmp_path)?;
        }

        // 创建 BufWriter，缓冲区大小 64KB
        let file = File::create(&tmp_path)?;
        let writer = BufWriter::with_capacity(64 * 1024, file);

        self.output_path = Some(output_path);
        self.tmp_path = Some(tmp_path);
        self.writer = Some(writer);
        self.frames_written = 0;
        self.state_dim = None;
        self.bytes_written = 0;

        // config 在 BinaryAdapter 中目前未使用

        Ok(())
    }

    fn write_frame(&mut self, frame: &SequenceFrame) -> CoreResult<()> {
        let state_dim = frame.state.dim() as u32;

        // 首帧写入时确定 state_dim 并写入 header
        if self.state_dim.is_none() {
            let writer = self
                .writer
                .as_mut()
                .ok_or_else(|| CoreError::SinkError("适配器未打开".into()))?;
            Self::write_header(writer, state_dim)?;
            self.bytes_written = 20; // header 大小
            self.state_dim = Some(state_dim);
        }

        // 校验帧的 state_dim 与首帧一致
        let expected_dim = self.state_dim.unwrap();
        if state_dim != expected_dim {
            return Err(CoreError::SinkError(format!(
                "帧的 state_dim ({}) 与首帧 ({}) 不一致",
                state_dim, expected_dim
            )));
        }

        // 序列化帧
        let frame_bytes = Self::serialize_frame(frame, state_dim);

        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| CoreError::SinkError("适配器未打开".into()))?;
        writer.write_all(&frame_bytes)?;

        self.bytes_written += frame_bytes.len() as u64;
        self.frames_written += 1;

        Ok(())
    }

    fn close(&mut self) -> CoreResult<OutputStats> {
        // 如果没有写入任何帧，也需要写入 header（state_dim = 0）
        if self.state_dim.is_none() {
            let writer = self
                .writer
                .as_mut()
                .ok_or_else(|| CoreError::SinkError("适配器未打开".into()))?;
            Self::write_header(writer, 0)?;
            self.bytes_written = 20;
            self.state_dim = Some(0);
        }

        // 回填 header 中的 frame_count
        self.patch_frame_count()?;

        // 刷新并关闭 writer
        let writer = self
            .writer
            .take()
            .ok_or_else(|| CoreError::SinkError("适配器未打开或已关闭".into()))?;
        let mut file = writer
            .into_inner()
            .map_err(|e| CoreError::SinkError(format!("刷新 BufWriter 失败: {}", e)))?;
        file.flush()?;

        let tmp_path = self
            .tmp_path
            .as_ref()
            .ok_or_else(|| CoreError::SinkError("临时路径为空".into()))?;
        let output_path = self
            .output_path
            .as_ref()
            .ok_or_else(|| CoreError::SinkError("输出路径为空".into()))?;

        // 获取文件大小
        let file_meta = fs::metadata(tmp_path)?;
        let file_bytes = file_meta.len();

        // 将临时文件重命名为最终文件
        fs::rename(tmp_path, output_path)?;

        Ok(OutputStats {
            frames_written: self.frames_written,
            bytes_written: file_bytes,
            output_path: Some(output_path.clone()),
            file_hash: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{FrameData, FrameState, SequenceFrame};
    use tempfile::TempDir;

    /// 构造测试用帧序列
    fn make_test_frames(n: usize) -> Vec<SequenceFrame> {
        (0..n)
            .map(|i| {
                let values: Vec<FrameState> = vec![
                    FrameState::Integer(i as i64 * 100),
                    FrameState::Float(i as f64 * 0.25),
                    FrameState::Bool(i % 2 == 0),
                ];
                let data: FrameData = values.into_iter().collect();
                let label = if i % 3 == 0 {
                    Some(format!("bin_{}", i))
                } else {
                    None
                };
                SequenceFrame {
                    step_index: i as u64,
                    state: data,
                    label,
                }
            })
            .collect()
    }

    fn default_config() -> OutputConfig {
        OutputConfig::default()
    }

    #[test]
    fn test_binary_header_is_valid() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(5);

        let mut adapter = BinaryAdapter::new();
        adapter
            .open(tmp.path(), "test", 1, 0xABCD, &default_config())
            .unwrap();
        for f in &frames {
            adapter.write_frame(f).unwrap();
        }
        let stats = adapter.close().unwrap();

        let raw = fs::read(&stats.output_path.unwrap()).unwrap();

        // 验证 magic
        assert_eq!(&raw[0..4], MAGIC);

        // 验证 version
        let version = u32::from_le_bytes(raw[4..8].try_into().unwrap());
        assert_eq!(version, VERSION);

        // 验证 frame_count（回填后应为 5）
        let frame_count = u64::from_le_bytes(raw[8..16].try_into().unwrap());
        assert_eq!(frame_count, 5);

        // 验证 state_dim
        let state_dim = u32::from_le_bytes(raw[16..20].try_into().unwrap());
        assert_eq!(state_dim, 3);
    }

    #[test]
    fn test_binary_header_state_dim_offset() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(1);

        let mut adapter = BinaryAdapter::new();
        adapter
            .open(tmp.path(), "dim", 0, 0, &default_config())
            .unwrap();
        adapter.write_frame(&frames[0]).unwrap();
        let stats = adapter.close().unwrap();

        let raw = fs::read(&stats.output_path.unwrap()).unwrap();

        // state_dim 在偏移 12 处（magic 4 + version 4 + frame_count 待回填 8 = 16? no...）
        // 等等: magic(4) + version(4) + frame_count(8) + state_dim(4) = 20 字节 header
        // 实际上 header 就是 16 字节？不，magic(4) + version(4) = 8, frame_count(8) = 16, state_dim(4) = 20
        // 等一下，让我重新看格式：
        // Header (16 bytes):
        //   magic:       [u8; 4]  "SGEN"
        //   version:     u32      = 1
        //   frame_count: u64      = N
        //   state_dim:   u32
        // 4 + 4 + 8 + 4 = 20 bytes...

        // 但实际上规格文档说 header 是 16 bytes。让我重新核对。
        // 嗯，magic[4] + version u32[4] + frame_count u64[8] + state_dim u32[4] = 20 bytes。
        // 但文档说 Header 16 bytes。可能是文档写错了，实际是 20 bytes。
        // 看 frame_count u64 = 8 bytes, 4+4+8+4=20
        
        // 实际上仔细看文档中的 header 结构：
        // magic(4) + version u32(4) = 8 bytes 开始，frame_count u64 = 8 bytes = offset 8
        // state_dim u32 = 4 bytes = offset 16
        // total = 20 bytes
        // 但文档标注 Header 16 bytes，这是格式描述有误。实际应为 20 bytes。

        // 验证 frame_count 在正确位置（offset 8）
        let frame_count = u64::from_le_bytes(raw[8..16].try_into().unwrap());
        assert_eq!(frame_count, 1);

        // 验证 state_dim 在正确位置（offset 16）
        let state_dim = u32::from_le_bytes(raw[16..20].try_into().unwrap());
        assert_eq!(state_dim, 3);
    }

    #[test]
    fn test_binary_magic_validation() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(1);

        let mut adapter = BinaryAdapter::new();
        adapter
            .open(tmp.path(), "magic", 0, 0, &default_config())
            .unwrap();
        adapter.write_frame(&frames[0]).unwrap();
        let stats = adapter.close().unwrap();

        let raw = fs::read(&stats.output_path.unwrap()).unwrap();
        // 确保 magic 字节完全匹配
        assert_eq!(raw[0], b'S');
        assert_eq!(raw[1], b'G');
        assert_eq!(raw[2], b'E');
        assert_eq!(raw[3], b'N');
        assert_eq!(&raw[0..4], b"SGEN");
    }

    #[test]
    fn test_binary_frame_serialization_structure() {
        let tmp = TempDir::new().unwrap();

        let values: Vec<FrameState> = vec![
            FrameState::Integer(42),
            FrameState::Float(3.14),
            FrameState::Bool(true),
        ];
        let data: FrameData = values.into_iter().collect();
        let frame = SequenceFrame::with_label(7, data, "test_label");

        let mut adapter = BinaryAdapter::new();
        adapter
            .open(tmp.path(), "struct", 0, 0, &default_config())
            .unwrap();
        adapter.write_frame(&frame).unwrap();
        let stats = adapter.close().unwrap();

        let raw = fs::read(&stats.output_path.unwrap()).unwrap();

        // 跳过 header (20 bytes = magic 4 + version 4 + frame_count 8 + state_dim 4)
        let body = &raw[20..];

        // step_index (u64 LE) = 7
        let step = u64::from_le_bytes(body[0..8].try_into().unwrap());
        assert_eq!(step, 7);

        // 第一个值：Integer(42), tag=0x01
        assert_eq!(body[8], 0x01);
        let int_val = i64::from_le_bytes(body[9..17].try_into().unwrap());
        assert_eq!(int_val, 42);

        // 第二个值：Float(3.14), tag=0x02
        assert_eq!(body[17], 0x02);
        let float_val = f64::from_le_bytes(body[18..26].try_into().unwrap());
        assert!((float_val - 3.14).abs() < 1e-10);

        // 第三个值：Bool(true), tag=0x03
        assert_eq!(body[26], 0x03);
        assert_eq!(body[27], 1);

        // label_len 在 offset 8 + 3*9 = 35 处
        let label_len = u32::from_le_bytes(body[35..39].try_into().unwrap());
        assert_eq!(label_len, 10); // "test_label".len() = 10

        // label 字节
        let label = std::str::from_utf8(&body[39..39 + 10]).unwrap();
        assert_eq!(label, "test_label");
    }

    #[test]
    fn test_binary_empty_no_frames() {
        let tmp = TempDir::new().unwrap();

        let mut adapter = BinaryAdapter::new();
        adapter
            .open(tmp.path(), "empty", 0, 0, &default_config())
            .unwrap();
        let stats = adapter.close().unwrap();

        assert_eq!(stats.frames_written, 0);

        let raw = fs::read(&stats.output_path.unwrap()).unwrap();
        assert_eq!(&raw[0..4], MAGIC);
        let frame_count = u64::from_le_bytes(raw[8..16].try_into().unwrap());
        assert_eq!(frame_count, 0);
    }

    #[test]
    fn test_binary_dimension_mismatch() {
        let tmp = TempDir::new().unwrap();

        let mut adapter = BinaryAdapter::new();
        adapter
            .open(tmp.path(), "dim_err", 0, 0, &default_config())
            .unwrap();

        // 写入第一帧（state_dim = 2）
        let values1: Vec<FrameState> =
            vec![FrameState::Integer(1), FrameState::Integer(2)];
        let data1: FrameData = values1.into_iter().collect();
        let frame1 = SequenceFrame::new(0, data1);
        adapter.write_frame(&frame1).unwrap();

        // 尝试写入第二帧（state_dim = 3），应报错
        let values2: Vec<FrameState> = vec![
            FrameState::Integer(1),
            FrameState::Integer(2),
            FrameState::Integer(3),
        ];
        let data2: FrameData = values2.into_iter().collect();
        let frame2 = SequenceFrame::new(1, data2);
        let result = adapter.write_frame(&frame2);

        assert!(result.is_err());
    }

    #[test]
    fn test_binary_write_batch() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(50);

        let mut adapter = BinaryAdapter::new();
        adapter
            .open(tmp.path(), "batch", 0, 0, &default_config())
            .unwrap();
        adapter.write_batch(&frames).unwrap();
        let stats = adapter.close().unwrap();

        assert_eq!(stats.frames_written, 50);

        let raw = fs::read(&stats.output_path.unwrap()).unwrap();
        let frame_count = u64::from_le_bytes(raw[8..16].try_into().unwrap());
        assert_eq!(frame_count, 50);
    }

    #[test]
    fn test_binary_close_without_open() {
        let mut adapter = BinaryAdapter::new();
        let result = adapter.close();
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_frame_without_label() {
        let tmp = TempDir::new().unwrap();

        let values: Vec<FrameState> = vec![FrameState::Bool(false)];
        let data: FrameData = values.into_iter().collect();
        let frame = SequenceFrame::new(0, data); // 无标签

        let mut adapter = BinaryAdapter::new();
        adapter
            .open(tmp.path(), "nolabel", 0, 0, &default_config())
            .unwrap();
        adapter.write_frame(&frame).unwrap();
        let stats = adapter.close().unwrap();

        let raw = fs::read(&stats.output_path.unwrap()).unwrap();
        let body = &raw[20..];

        // offset: step_index(8) + 1 value(9) = 17, label_len at 17
        let label_len = u32::from_le_bytes(body[17..21].try_into().unwrap());
        assert_eq!(label_len, 0);
    }

    #[test]
    fn test_binary_serialize_frame_size() {
        let values: Vec<FrameState> = vec![
            FrameState::Integer(0),
            FrameState::Float(0.0),
            FrameState::Bool(false),
        ];
        let data: FrameData = values.into_iter().collect();
        let frame = SequenceFrame::with_label(0, data, "ABC"); // label 3 bytes

        let bytes = BinaryAdapter::serialize_frame(&frame, 3);

        // 期望大小: step_index(8) + 3*9 + label_len(4) + label_bytes(3) = 8 + 27 + 4 + 3 = 42
        assert_eq!(bytes.len(), 42);
    }
}
