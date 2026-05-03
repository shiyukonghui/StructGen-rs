use crate::core::{CoreResult, SequenceFrame};

use super::processor::Processor;

/// 空处理器：透传所有帧，不做任何变换
/// 主要用于测试和作为占位处理器使用
pub struct NullProcessor;

impl NullProcessor {
    /// 创建 NullProcessor 实例
    pub fn new() -> Self {
        NullProcessor
    }
}

impl Default for NullProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Processor for NullProcessor {
    fn name(&self) -> &'static str {
        "null"
    }

    fn process(
        &self,
        input: Box<dyn Iterator<Item = SequenceFrame> + Send>,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        Ok(input)
    }
}

/// 工厂函数：从 JSON 配置创建 NullProcessor
pub fn create_null_processor(_config: &serde_json::Value) -> CoreResult<Box<dyn Processor>> {
    Ok(Box::new(NullProcessor::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{FrameData, FrameState};

    /// 构造测试用的帧序列
    fn make_test_frames() -> Vec<SequenceFrame> {
        vec![
            SequenceFrame::new(
                0,
                FrameData {
                    values: vec![FrameState::Integer(1), FrameState::Float(2.0)],
                },
            ),
            SequenceFrame::new(
                1,
                FrameData {
                    values: vec![FrameState::Bool(true), FrameState::Integer(3)],
                },
            ),
            SequenceFrame::new(
                2,
                FrameData {
                    values: vec![FrameState::Float(4.0)],
                },
            ),
        ]
    }

    #[test]
    fn test_null_processor_passthrough() {
        let frames = make_test_frames();
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> = Box::new(frames.clone().into_iter());
        let processor = NullProcessor::new();
        let output: Vec<SequenceFrame> = processor.process(input).unwrap().collect();
        // 验证输入输出完全一致（逐帧比较）
        assert_eq!(output.len(), frames.len());
        for (i, (out_frame, in_frame)) in output.iter().zip(frames.iter()).enumerate() {
            assert_eq!(
                out_frame, in_frame,
                "帧 {} 不一致: 期望 {:?}, 实际 {:?}",
                i, in_frame, out_frame
            );
        }
    }

    #[test]
    fn test_null_processor_empty_input() {
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> = Box::new(std::iter::empty());
        let processor = NullProcessor::new();
        let output: Vec<SequenceFrame> = processor.process(input).unwrap().collect();
        assert!(output.is_empty());
    }

    #[test]
    fn test_null_processor_name() {
        let processor = NullProcessor::new();
        assert_eq!(processor.name(), "null");
    }
}
