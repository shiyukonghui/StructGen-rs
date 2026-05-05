use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::{CoreError, CoreResult, FrameData, FrameState, SequenceFrame};

use super::processor::Processor;

/// 差分编码器配置
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiffEncoderConfig {
    /// 是否在序列首帧前插入一个全零的参考帧（使首帧编码为首帧本身）
    #[serde(default)]
    pub prepend_zero_frame: bool,
}

/// 差分编码器：对相邻帧计算差分
///
/// 差分规则：
/// - Integer(a), Integer(b) → Integer(a - b)
/// - Float(a), Float(b) → Float(a - b)
/// - Bool(a), Bool(b) → Bool(a ^ b)  // XOR
/// - 类型不匹配 → 产出当前值
pub struct DiffEncoder {
    config: DiffEncoderConfig,
}

/// 差分迭代器适配器
struct DiffIter<I: Iterator<Item = SequenceFrame>> {
    inner: I,
    config: DiffEncoderConfig,
    prev_state: Option<FrameData>,
}

impl<I: Iterator<Item = SequenceFrame>> Iterator for DiffIter<I> {
    type Item = SequenceFrame;

    fn next(&mut self) -> Option<Self::Item> {
        let frame = self.inner.next()?;

        // 提前保存 frame.state 以便差分后设置 prev_state
        let next_prev = frame.state.clone();

        let diff_frame = match &self.prev_state {
            None => {
                if self.config.prepend_zero_frame {
                    // 与零帧差分 = 自身
                    let zero_values: Vec<FrameState> = frame
                        .state
                        .values
                        .iter()
                        .map(|s| match s {
                            FrameState::Integer(_) => FrameState::Integer(0),
                            FrameState::Float(_) => FrameState::Float(0.0),
                            FrameState::Bool(_) => FrameState::Bool(false),
                        })
                        .collect();
                    let diff_values: Vec<FrameState> = frame
                        .state
                        .values
                        .iter()
                        .zip(zero_values.iter())
                        .map(|(cur, prv)| compute_diff(cur, prv))
                        .collect();
                    SequenceFrame {
                        step_index: frame.step_index,
                        state: FrameData { values: diff_values },
                        label: frame.label,
                        sample_id: None,
                    }
                } else {
                    // 首帧直接产出
                    frame
                }
            }
            Some(ref previous) => {
                let diff_values: Vec<FrameState> = frame
                    .state
                    .values
                    .iter()
                    .zip(previous.values.iter())
                    .map(|(cur, prv)| compute_diff(cur, prv))
                    .collect();
                SequenceFrame {
                    step_index: frame.step_index,
                    state: FrameData { values: diff_values },
                    label: frame.label,
                    sample_id: None,
                }
            }
        };

        self.prev_state = Some(next_prev);
        Some(diff_frame)
    }
}

/// 计算两个 FrameState 之间的差分
fn compute_diff(cur: &FrameState, prev: &FrameState) -> FrameState {
    match (cur, prev) {
        (FrameState::Integer(a), FrameState::Integer(b)) => FrameState::Integer(a - b),
        (FrameState::Float(a), FrameState::Float(b)) => FrameState::Float(a - b),
        (FrameState::Bool(a), FrameState::Bool(b)) => FrameState::Bool(a ^ b),
        // 类型不匹配时产出当前值
        _ => *cur,
    }
}

impl DiffEncoder {
    /// 根据配置创建差分编码器
    pub fn new(config: DiffEncoderConfig) -> Self {
        DiffEncoder { config }
    }
}

impl Processor for DiffEncoder {
    fn name(&self) -> &'static str {
        "diff_encoder"
    }

    fn process(
        &self,
        input: Box<dyn Iterator<Item = SequenceFrame> + Send>,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let iter = DiffIter {
            inner: input,
            config: self.config.clone(),
            prev_state: None,
        };
        Ok(Box::new(iter))
    }
}

/// 工厂函数：从 JSON 配置创建差分编码器
pub fn create_diff_encoder(config: &Value) -> CoreResult<Box<dyn Processor>> {
    let config: DiffEncoderConfig = if config.is_null() {
        DiffEncoderConfig::default()
    } else {
        serde_json::from_value(config.clone()).map_err(|e| {
            CoreError::SerializationError(format!("差分编码器配置解析失败: {}", e))
        })?
    };
    Ok(Box::new(DiffEncoder::new(config)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(step: u64, values: Vec<FrameState>) -> SequenceFrame {
        SequenceFrame::new(step, FrameData { values })
    }

    #[test]
    fn test_diff_integer() {
        let frames = vec![
            make_frame(0, vec![FrameState::Integer(10)]),
            make_frame(1, vec![FrameState::Integer(7)]),
            make_frame(2, vec![FrameState::Integer(15)]),
        ];
        let config = DiffEncoderConfig::default();
        let encoder = DiffEncoder::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = encoder.process(input).unwrap().collect();

        assert_eq!(output.len(), 3);
        // 首帧直接产出
        assert_eq!(output[0].state.values[0], FrameState::Integer(10));
        // 第二帧: 7 - 10 = -3
        assert_eq!(output[1].state.values[0], FrameState::Integer(-3));
        // 第三帧: 15 - 7 = 8
        assert_eq!(output[2].state.values[0], FrameState::Integer(8));
    }

    #[test]
    fn test_diff_float() {
        let frames = vec![
            make_frame(0, vec![FrameState::Float(10.0)]),
            make_frame(1, vec![FrameState::Float(12.0)]),
        ];
        let config = DiffEncoderConfig::default();
        let encoder = DiffEncoder::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = encoder.process(input).unwrap().collect();

        assert_eq!(output[0].state.values[0], FrameState::Float(10.0));
        assert_eq!(output[1].state.values[0], FrameState::Float(2.0));
    }

    #[test]
    fn test_diff_bool_xor() {
        let frames = vec![
            make_frame(0, vec![FrameState::Bool(true)]),
            make_frame(1, vec![FrameState::Bool(true)]),   // true ^ true = false
            make_frame(2, vec![FrameState::Bool(false)]),  // false ^ true = true
            make_frame(3, vec![FrameState::Bool(false)]),  // false ^ false = false
        ];
        let config = DiffEncoderConfig::default();
        let encoder = DiffEncoder::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = encoder.process(input).unwrap().collect();

        assert_eq!(output[0].state.values[0], FrameState::Bool(true));   // 首帧
        assert_eq!(output[1].state.values[0], FrameState::Bool(false));  // true ^ true
        assert_eq!(output[2].state.values[0], FrameState::Bool(true));   // false ^ true
        assert_eq!(output[3].state.values[0], FrameState::Bool(false));  // false ^ false
    }

    #[test]
    fn test_diff_type_mismatch() {
        let frames = vec![
            make_frame(0, vec![FrameState::Integer(10)]),
            make_frame(1, vec![FrameState::Float(5.0)]), // 类型不匹配
        ];
        let config = DiffEncoderConfig::default();
        let encoder = DiffEncoder::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = encoder.process(input).unwrap().collect();

        // 类型不匹配时产出当前值
        assert_eq!(output[1].state.values[0], FrameState::Float(5.0));
    }

    #[test]
    fn test_diff_prepend_zero_frame() {
        let frames = vec![
            make_frame(0, vec![FrameState::Integer(5)]),
            make_frame(1, vec![FrameState::Integer(8)]),
        ];
        let config = DiffEncoderConfig {
            prepend_zero_frame: true,
        };
        let encoder = DiffEncoder::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = encoder.process(input).unwrap().collect();

        // 首帧与零帧差分 = 自身: 5 - 0 = 5
        assert_eq!(output[0].state.values[0], FrameState::Integer(5));
        // 第二帧: 8 - 5 = 3
        assert_eq!(output[1].state.values[0], FrameState::Integer(3));
    }

    #[test]
    fn test_diff_empty_input() {
        let config = DiffEncoderConfig::default();
        let encoder = DiffEncoder::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(std::iter::empty());
        let output: Vec<SequenceFrame> = encoder.process(input).unwrap().collect();
        assert!(output.is_empty());
    }

    #[test]
    fn test_diff_preserves_step_index() {
        let frames = vec![
            make_frame(0, vec![FrameState::Integer(100)]),
            make_frame(1, vec![FrameState::Integer(200)]),
        ];
        let config = DiffEncoderConfig::default();
        let encoder = DiffEncoder::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = encoder.process(input).unwrap().collect();

        assert_eq!(output[0].step_index, 0);
        assert_eq!(output[1].step_index, 1);
    }
}
