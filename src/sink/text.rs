//! 纯文本输出适配器实现
//!
//! 将帧序列以 Unicode 文本格式写出，每行一帧，帧内所有状态值映射为字符后拼接。
//! 适合被语言模型的 DataLoader 直接加载。

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::core::*;

use super::adapter::{format_output_filename, OutputConfig, OutputStats, SinkAdapter};

/// 文本输出适配器
///
/// 每帧写入一行文本，格式为：
/// - Integer → `char::from_u32` 映射为 Unicode 字符（无效码点使用替换字符 U+FFFD）
/// - Float → 映射到 0-255 整数范围再映射为字符
/// - Bool → '0' 或 '1'
pub struct TextAdapter {
    /// 最终输出文件路径
    output_path: Option<PathBuf>,
    /// 临时文件路径
    tmp_path: Option<PathBuf>,
    /// 带缓冲的文件写入器（64KB 缓冲区）
    writer: Option<BufWriter<File>>,
    /// 已写入帧数
    frames_written: u64,
    /// 已写入字节数
    bytes_written: u64,
}

impl TextAdapter {
    /// 创建新的 TextAdapter 实例
    pub fn new() -> Self {
        TextAdapter {
            output_path: None,
            tmp_path: None,
            writer: None,
            frames_written: 0,
            bytes_written: 0,
        }
    }

    /// 将单个 FrameState 转换为 Unicode 字符
    fn frame_state_to_char(state: &FrameState) -> char {
        match state {
            FrameState::Integer(v) => {
                if *v < 0 {
                    // 负值使用替换字符
                    '\u{FFFD}'
                } else {
                    char::from_u32(*v as u32).unwrap_or('\u{FFFD}')
                }
            }
            FrameState::Float(v) => {
                // 将浮点值映射到 0-255 范围
                let clamped = v.clamp(0.0, 255.0);
                char::from(clamped as u8)
            }
            FrameState::Bool(v) => {
                if *v {
                    '1'
                } else {
                    '0'
                }
            }
        }
    }

    /// 将整帧转换为文本行（不含换行符）
    fn frame_to_line(frame: &SequenceFrame) -> String {
        let mut line = String::new();

        // 前缀 sample_id（如果存在）
        if let Some(sid) = frame.sample_id {
            line.push_str(&format!("{}:", sid));
        }

        // 状态值映射为字符
        let chars: String = frame
            .state
            .values
            .iter()
            .map(Self::frame_state_to_char)
            .collect();
        line.push_str(&chars);
        line
    }
}

impl Default for TextAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl SinkAdapter for TextAdapter {
    fn format(&self) -> OutputFormat {
        OutputFormat::Text
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
        let tmp_name = format_output_filename(task_name, shard_id, seed, "txt.tmp");
        let final_name = format_output_filename(task_name, shard_id, seed, "txt");
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
        self.bytes_written = 0;

        // config 在 TextAdapter 中目前未使用，保持接收以符合接口

        Ok(())
    }

    fn write_frame(&mut self, frame: &SequenceFrame) -> CoreResult<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| CoreError::SinkError("适配器未打开".into()))?;

        let mut line = Self::frame_to_line(frame);

        // 追加标签（如果有）
        if let Some(ref label) = frame.label {
            line.push(' ');
            line.push_str(label);
        }

        line.push('\n');

        let line_bytes = line.as_bytes();
        writer.write_all(line_bytes)?;
        self.bytes_written += line_bytes.len() as u64;
        self.frames_written += 1;

        Ok(())
    }

    fn close(&mut self) -> CoreResult<OutputStats> {
        let writer = self
            .writer
            .take()
            .ok_or_else(|| CoreError::SinkError("适配器未打开或已关闭".into()))?;

        // 刷新并关闭 BufWriter
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
                    FrameState::Integer(65 + i as i64), // 'A', 'B', 'C', ...
                    FrameState::Float(i as f64 * 10.0),
                    FrameState::Bool(i % 2 == 0),
                ];
                let data: FrameData = values.into_iter().collect();
                let label = if i % 5 == 0 {
                    Some(format!("label_{}", i))
                } else {
                    None
                };
                SequenceFrame {
                    step_index: i as u64,
                    state: data,
                    label,
                    sample_id: None,
                }
            })
            .collect()
    }

    fn default_config() -> OutputConfig {
        OutputConfig::default()
    }

    #[test]
    fn test_text_output_is_valid_utf8() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(10);

        let mut adapter = TextAdapter::new();
        adapter
            .open(tmp.path(), "test", 1, 0x42, &default_config())
            .unwrap();
        for f in &frames {
            adapter.write_frame(f).unwrap();
        }
        let stats = adapter.close().unwrap();

        let output_path = stats.output_path.unwrap();
        let content = fs::read_to_string(&output_path).unwrap();

        // 验证内容为有效 UTF-8，且行数 >= 10
        let lines: Vec<&str> = content.lines().collect();
        assert!(lines.len() >= 10);
        assert_eq!(stats.frames_written, 10);
        assert!(stats.bytes_written > 0);
    }

    #[test]
    fn test_text_line_format() {
        let tmp = TempDir::new().unwrap();

        let values: Vec<FrameState> = vec![
            FrameState::Integer(65), // 'A'
            FrameState::Bool(true),  // '1'
            FrameState::Float(128.0), // 映射为 char
        ];
        let data: FrameData = values.into_iter().collect();
        let frame = SequenceFrame::with_label(0, data, "hello");

        let mut adapter = TextAdapter::new();
        adapter
            .open(tmp.path(), "fmt", 0, 0, &default_config())
            .unwrap();
        adapter.write_frame(&frame).unwrap();
        let stats = adapter.close().unwrap();

        let content = fs::read_to_string(&stats.output_path.unwrap()).unwrap();
        // 'A' + '1' + char(128) + ' ' + "hello" + '\n'
        assert!(content.starts_with('A'));
        assert!(content.contains("hello"));
        assert!(content.ends_with('\n'));
    }

    #[test]
    fn test_text_empty_frame() {
        let tmp = TempDir::new().unwrap();
        let frame = SequenceFrame::new(0, FrameData::new());

        let mut adapter = TextAdapter::new();
        adapter
            .open(tmp.path(), "empty", 0, 0, &default_config())
            .unwrap();
        adapter.write_frame(&frame).unwrap();
        let stats = adapter.close().unwrap();

        let content = fs::read_to_string(&stats.output_path.unwrap()).unwrap();
        // 空帧只产生 '\n'
        assert_eq!(content.trim(), "");
        assert_eq!(stats.frames_written, 1);
    }

    #[test]
    fn test_text_unicode_integer_mapping() {
        // Integer 值映射到 Unicode 字符
        let tmp = TempDir::new().unwrap();

        // 中文字符 '中' 的码点为 U+4E2D = 20013
        let values: Vec<FrameState> = vec![FrameState::Integer(0x4E2D)];
        let data: FrameData = values.into_iter().collect();
        let frame = SequenceFrame::new(0, data);

        let mut adapter = TextAdapter::new();
        adapter
            .open(tmp.path(), "unicode", 0, 0, &default_config())
            .unwrap();
        adapter.write_frame(&frame).unwrap();
        let stats = adapter.close().unwrap();

        let content = fs::read_to_string(&stats.output_path.unwrap()).unwrap();
        assert!(content.contains('中'));
    }

    #[test]
    fn test_text_negative_integer_fallback() {
        // 负 Integer 值应回退到替换字符 U+FFFD
        let tmp = TempDir::new().unwrap();
        let values: Vec<FrameState> = vec![FrameState::Integer(-1)];
        let data: FrameData = values.into_iter().collect();
        let frame = SequenceFrame::new(0, data);

        let mut adapter = TextAdapter::new();
        adapter
            .open(tmp.path(), "neg", 0, 0, &default_config())
            .unwrap();
        adapter.write_frame(&frame).unwrap();
        let stats = adapter.close().unwrap();

        let content = fs::read_to_string(&stats.output_path.unwrap()).unwrap();
        assert!(content.contains('\u{FFFD}'));
    }

    #[test]
    fn test_text_invalid_codepoint_fallback() {
        // 代理对范围 0xD800-0xDFFF 无效，应回退
        let tmp = TempDir::new().unwrap();
        let values: Vec<FrameState> = vec![FrameState::Integer(0xD800)];
        let data: FrameData = values.into_iter().collect();
        let frame = SequenceFrame::new(0, data);

        let mut adapter = TextAdapter::new();
        adapter
            .open(tmp.path(), "surrogate", 0, 0, &default_config())
            .unwrap();
        adapter.write_frame(&frame).unwrap();
        let stats = adapter.close().unwrap();

        let content = fs::read_to_string(&stats.output_path.unwrap()).unwrap();
        assert!(content.contains('\u{FFFD}'));
    }

    #[test]
    fn test_text_float_mapping() {
        // 浮点数映射：clamp 到 0-255 后转为 char
        let tmp = TempDir::new().unwrap();
        let values: Vec<FrameState> = vec![FrameState::Float(65.0)]; // 'A'
        let data: FrameData = values.into_iter().collect();
        let frame = SequenceFrame::new(0, data);

        let mut adapter = TextAdapter::new();
        adapter
            .open(tmp.path(), "float_test", 0, 0, &default_config())
            .unwrap();
        adapter.write_frame(&frame).unwrap();
        let stats = adapter.close().unwrap();

        let content = fs::read_to_string(&stats.output_path.unwrap()).unwrap();
        assert!(content.contains('A'));
    }

    #[test]
    fn test_text_float_negative_clamped() {
        // 负数浮点值被 clamp 到 0（空字符 char(0)）
        let tmp = TempDir::new().unwrap();
        let values: Vec<FrameState> = vec![FrameState::Float(-100.0)];
        let data: FrameData = values.into_iter().collect();
        let frame = SequenceFrame::new(0, data);

        let mut adapter = TextAdapter::new();
        adapter
            .open(tmp.path(), "neg_float", 0, 0, &default_config())
            .unwrap();
        adapter.write_frame(&frame).unwrap();
        let stats = adapter.close().unwrap();

        let output_path = stats.output_path.unwrap();
        let bytes = fs::read(&output_path).unwrap();
        // clamp(-100, 0, 255) = 0 → '\0' 作为首字节，然后 '\n'
        assert_eq!(bytes[0], 0x00);
        assert_eq!(bytes[1], b'\n');
    }

    #[test]
    fn test_text_float_large_clamped() {
        // 大浮点值被 clamp 到 255 → char(255) = 'ÿ' (U+00FF)
        // UTF-8 编码为 0xC3 0xBF
        let tmp = TempDir::new().unwrap();
        let values: Vec<FrameState> = vec![FrameState::Float(1000.0)];
        let data: FrameData = values.into_iter().collect();
        let frame = SequenceFrame::new(0, data);

        let mut adapter = TextAdapter::new();
        adapter
            .open(tmp.path(), "large_float", 0, 0, &default_config())
            .unwrap();
        adapter.write_frame(&frame).unwrap();
        let stats = adapter.close().unwrap();

        let output_path = stats.output_path.unwrap();
        let bytes = fs::read(&output_path).unwrap();
        // 'ÿ' (U+00FF) 的 UTF-8 编码: 0xC3 0xBF
        assert_eq!(bytes[0], 0xC3);
        assert_eq!(bytes[1], 0xBF);
        assert_eq!(bytes[2], b'\n');
    }

    #[test]
    fn test_text_write_batch() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(30);

        let mut adapter = TextAdapter::new();
        adapter
            .open(tmp.path(), "batch", 0, 0, &default_config())
            .unwrap();
        adapter.write_batch(&frames).unwrap();
        let stats = adapter.close().unwrap();

        assert_eq!(stats.frames_written, 30);
    }

    #[test]
    fn test_text_close_without_open() {
        let mut adapter = TextAdapter::new();
        let result = adapter.close();
        assert!(result.is_err());
    }
}
