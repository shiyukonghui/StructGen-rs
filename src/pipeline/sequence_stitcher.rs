//! 序列串联处理器
//!
//! 将多个独立帧的 token 序列串联成训练格式。
//! 输入是已经过 PatchTokenizer 处理的帧，每帧格式为 [start][patches...][end]。
//! 输出将 frames_per_sequence 帧串联成一个序列。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::{CoreError, CoreResult, FrameData, FrameState, SequenceFrame};

use super::processor::Processor;

/// SequenceStitcher 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceStitcherConfig {
    /// 每个序列包含的帧数
    pub frames_per_sequence: usize,
    /// 是否在序列开头添加 start token
    #[serde(default)]
    pub add_sequence_start: bool,
    /// 是否在序列结尾添加 end token
    #[serde(default)]
    pub add_sequence_end: bool,
    /// 序列级 start token
    pub start_token: Option<i64>,
    /// 序列级 end token
    pub end_token: Option<i64>,
}

/// 序列串联处理器
///
/// 将多个独立帧的 token 序列串联成训练格式。
/// 例如，配置 frames_per_sequence=5 时：
/// - 输入：5 帧，每帧格式为 [frame_start][patches...][frame_end]
/// - 输出：1 帧，格式为 [seq_start][frame0_tokens][frame1_tokens]...[frame4_tokens][seq_end]
pub struct SequenceStitcher {
    config: SequenceStitcherConfig,
}

/// 序列串联迭代器适配器
struct SequenceStitcherIter<I: Iterator<Item = SequenceFrame>> {
    inner: I,
    config: SequenceStitcherConfig,
    /// 缓冲区，用于收集帧
    buffer: Vec<SequenceFrame>,
}

impl<I: Iterator<Item = SequenceFrame>> Iterator for SequenceStitcherIter<I> {
    type Item = SequenceFrame;

    fn next(&mut self) -> Option<Self::Item> {
        // 收集 frames_per_sequence 帧
        self.buffer.clear();

        for _ in 0..self.config.frames_per_sequence {
            match self.inner.next() {
                Some(frame) => self.buffer.push(frame),
                None => break,
            }
        }

        // 如果没有收集到任何帧，返回 None
        if self.buffer.is_empty() {
            return None;
        }

        // 计算总 token 数
        let mut total_tokens = 0;
        if self.config.add_sequence_start && self.config.start_token.is_some() {
            total_tokens += 1;
        }
        for frame in &self.buffer {
            total_tokens += frame.state.values.len();
        }
        if self.config.add_sequence_end && self.config.end_token.is_some() {
            total_tokens += 1;
        }

        // 构建串联后的 token 序列
        let mut tokens = Vec::with_capacity(total_tokens);

        // 添加序列级 start token
        if self.config.add_sequence_start {
            if let Some(start) = self.config.start_token {
                tokens.push(FrameState::Integer(start));
            }
        }

        // 串联所有帧的 tokens
        for frame in &self.buffer {
            tokens.extend(frame.state.values.iter().cloned());
        }

        // 添加序列级 end token
        if self.config.add_sequence_end {
            if let Some(end) = self.config.end_token {
                tokens.push(FrameState::Integer(end));
            }
        }

        // 使用第一个帧的 step_index 作为输出帧的 step_index
        let step_index = self.buffer.first().map(|f| f.step_index).unwrap_or(0);

        Some(SequenceFrame {
            step_index,
            state: FrameData { values: tokens },
            label: self.buffer.first().and_then(|f| f.label.clone()),
            sample_id: self.buffer.first().and_then(|f| f.sample_id.clone()),
        })
    }
}

impl SequenceStitcher {
    /// 根据配置创建 SequenceStitcher
    pub fn new(config: SequenceStitcherConfig) -> CoreResult<Self> {
        // 校验 frames_per_sequence >= 1
        if config.frames_per_sequence == 0 {
            return Err(CoreError::InvalidParams(
                "sequence_stitcher: frames_per_sequence must be >= 1".into(),
            ));
        }

        // 校验 add_sequence_start 和 start_token 的一致性
        if config.add_sequence_start && config.start_token.is_none() {
            return Err(CoreError::InvalidParams(
                "sequence_stitcher: start_token must be set when add_sequence_start is true".into(),
            ));
        }

        // 校验 add_sequence_end 和 end_token 的一致性
        if config.add_sequence_end && config.end_token.is_none() {
            return Err(CoreError::InvalidParams(
                "sequence_stitcher: end_token must be set when add_sequence_end is true".into(),
            ));
        }

        Ok(SequenceStitcher { config })
    }
}

impl Processor for SequenceStitcher {
    fn name(&self) -> &'static str {
        "sequence_stitcher"
    }

    fn process(
        &self,
        input: Box<dyn Iterator<Item = SequenceFrame> + Send>,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let iter = SequenceStitcherIter {
            inner: input,
            config: self.config.clone(),
            buffer: Vec::with_capacity(self.config.frames_per_sequence),
        };
        Ok(Box::new(iter))
    }
}

/// 工厂函数：从 JSON 配置创建 SequenceStitcher
pub fn create_sequence_stitcher(config: &Value) -> CoreResult<Box<dyn Processor>> {
    let config: SequenceStitcherConfig = if config.is_null() {
        return Err(CoreError::InvalidParams(
            "sequence_stitcher requires configuration (frames_per_sequence)".into(),
        ));
    } else {
        serde_json::from_value(config.clone()).map_err(|e| {
            CoreError::SerializationError(format!("sequence_stitcher 配置解析失败: {}", e))
        })?
    };
    let stitcher = SequenceStitcher::new(config)?;
    Ok(Box::new(stitcher))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::registry::ProcessorRegistry;
    use crate::pipeline::register_all;

    /// 创建测试用的帧
    fn make_frame(step: u64, tokens: Vec<i64>) -> SequenceFrame {
        SequenceFrame::new(step, FrameData {
            values: tokens.into_iter().map(FrameState::Integer).collect(),
        })
    }

    #[test]
    fn test_sequence_stitcher_basic() {
        // 每帧 3 个 token: [start, patch, end]
        let frame1 = make_frame(0, vec![100, 1, 101]);
        let frame2 = make_frame(1, vec![100, 2, 101]);
        let frame3 = make_frame(2, vec![100, 3, 101]);

        let config = SequenceStitcherConfig {
            frames_per_sequence: 3,
            add_sequence_start: false,
            add_sequence_end: false,
            start_token: None,
            end_token: None,
        };
        let stitcher = SequenceStitcher::new(config).unwrap();
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(vec![frame1, frame2, frame3].into_iter());
        let output: Vec<SequenceFrame> = stitcher.process(input).unwrap().collect();

        // 输出应为 1 帧，包含 9 个 token
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].state.values.len(), 9);

        // 验证 token 顺序
        let expected: Vec<i64> = vec![100, 1, 101, 100, 2, 101, 100, 3, 101];
        let actual: Vec<i64> = output[0]
            .state
            .values
            .iter()
            .map(|v| match v {
                FrameState::Integer(n) => *n,
                _ => panic!("Expected Integer"),
            })
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_sequence_stitcher_with_sequence_tokens() {
        // 测试添加序列级 start/end token
        let frame1 = make_frame(0, vec![100, 1, 101]);
        let frame2 = make_frame(1, vec![100, 2, 101]);

        let config = SequenceStitcherConfig {
            frames_per_sequence: 2,
            add_sequence_start: true,
            add_sequence_end: true,
            start_token: Some(1000),
            end_token: Some(1001),
        };
        let stitcher = SequenceStitcher::new(config).unwrap();
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(vec![frame1, frame2].into_iter());
        let output: Vec<SequenceFrame> = stitcher.process(input).unwrap().collect();

        assert_eq!(output.len(), 1);
        // 1 (seq_start) + 3 (frame1) + 3 (frame2) + 1 (seq_end) = 8
        assert_eq!(output[0].state.values.len(), 8);

        // 验证 token 顺序: [seq_start][frame1][frame2][seq_end]
        let expected: Vec<i64> = vec![1000, 100, 1, 101, 100, 2, 101, 1001];
        let actual: Vec<i64> = output[0]
            .state
            .values
            .iter()
            .map(|v| match v {
                FrameState::Integer(n) => *n,
                _ => panic!("Expected Integer"),
            })
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_sequence_stitcher_multiple_sequences() {
        // 测试生成多个序列
        let frames: Vec<SequenceFrame> = (0..6)
            .map(|i| make_frame(i, vec![100, i as i64, 101]))
            .collect();

        let config = SequenceStitcherConfig {
            frames_per_sequence: 2,
            add_sequence_start: true,
            add_sequence_end: true,
            start_token: Some(1000),
            end_token: Some(1001),
        };
        let stitcher = SequenceStitcher::new(config).unwrap();
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = stitcher.process(input).unwrap().collect();

        // 6 帧分为 3 个序列
        assert_eq!(output.len(), 3);

        // 每个序列: 1 + 3 + 3 + 1 = 8 tokens
        for seq in &output {
            assert_eq!(seq.state.values.len(), 8);
        }
    }

    #[test]
    fn test_sequence_stitcher_partial_sequence() {
        // 测试输入帧数不足的情况
        let frame1 = make_frame(0, vec![100, 1, 101]);

        let config = SequenceStitcherConfig {
            frames_per_sequence: 5, // 要求 5 帧，但只提供 1 帧
            add_sequence_start: true,
            add_sequence_end: true,
            start_token: Some(1000),
            end_token: Some(1001),
        };
        let stitcher = SequenceStitcher::new(config).unwrap();
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(std::iter::once(frame1));
        let output: Vec<SequenceFrame> = stitcher.process(input).unwrap().collect();

        // 应该输出 1 个部分序列
        assert_eq!(output.len(), 1);
        // 1 + 3 + 1 = 5 tokens
        assert_eq!(output[0].state.values.len(), 5);
    }

    #[test]
    fn test_sequence_stitcher_empty_input() {
        let config = SequenceStitcherConfig {
            frames_per_sequence: 3,
            add_sequence_start: false,
            add_sequence_end: false,
            start_token: None,
            end_token: None,
        };
        let stitcher = SequenceStitcher::new(config).unwrap();
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(std::iter::empty());
        let output: Vec<SequenceFrame> = stitcher.process(input).unwrap().collect();

        assert!(output.is_empty());
    }

    #[test]
    fn test_sequence_stitcher_invalid_frames_per_sequence() {
        let config = SequenceStitcherConfig {
            frames_per_sequence: 0,
            add_sequence_start: false,
            add_sequence_end: false,
            start_token: None,
            end_token: None,
        };
        let result = SequenceStitcher::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_sequence_stitcher_missing_start_token() {
        let config = SequenceStitcherConfig {
            frames_per_sequence: 2,
            add_sequence_start: true,
            add_sequence_end: false,
            start_token: None, // 缺少 start_token
            end_token: None,
        };
        let result = SequenceStitcher::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_sequence_stitcher_missing_end_token() {
        let config = SequenceStitcherConfig {
            frames_per_sequence: 2,
            add_sequence_start: false,
            add_sequence_end: true,
            start_token: None,
            end_token: None, // 缺少 end_token
        };
        let result = SequenceStitcher::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_sequence_stitcher_preserves_step_index() {
        let frame1 = make_frame(10, vec![100, 1, 101]);
        let frame2 = make_frame(20, vec![100, 2, 101]);

        let config = SequenceStitcherConfig {
            frames_per_sequence: 2,
            add_sequence_start: false,
            add_sequence_end: false,
            start_token: None,
            end_token: None,
        };
        let stitcher = SequenceStitcher::new(config).unwrap();
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(vec![frame1, frame2].into_iter());
        let output: Vec<SequenceFrame> = stitcher.process(input).unwrap().collect();

        // 使用第一帧的 step_index
        assert_eq!(output[0].step_index, 10);
    }

    #[test]
    fn test_sequence_stitcher_via_registry() {
        let mut registry = ProcessorRegistry::new();
        register_all(&mut registry).unwrap();

        let config = serde_json::json!({
            "frames_per_sequence": 5,
            "add_sequence_start": true,
            "add_sequence_end": true,
            "start_token": 10000,
            "end_token": 10001
        });

        let processor = registry.get("sequence_stitcher", &config);
        assert!(processor.is_ok(), "Should be able to instantiate sequence_stitcher via registry");
        assert_eq!(processor.unwrap().name(), "sequence_stitcher");
    }
}