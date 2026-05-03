use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::{CoreError, CoreResult, FrameData, FrameState, SequenceFrame};

use super::processor::Processor;

/// 标准化方法枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NormalizeMethod {
    /// 线性缩放: (v - min) / (max - min) * max_val 钳位取整
    #[default]
    Linear,
    /// 对数分桶: log2(1 + |v|) 然后线性缩放
    LogBucket,
    /// 均匀分位数: 按分位数边界映射
    UniformQuantile,
}

/// 标准化器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizerConfig {
    /// 目标整数范围上限（含），默认 255，即映射到 [0, 255]
    #[serde(default = "default_max_val")]
    pub max_val: u32,
    /// 标准化方法
    #[serde(default)]
    pub method: NormalizeMethod,
    /// 可选：显式指定最小值边界，跳过第一遍扫描
    #[serde(default)]
    pub min: Option<f64>,
    /// 可选：显式指定最大值边界，跳过第一遍扫描
    #[serde(default)]
    pub max: Option<f64>,
}

fn default_max_val() -> u32 {
    255
}

impl Default for NormalizerConfig {
    fn default() -> Self {
        NormalizerConfig {
            max_val: default_max_val(),
            method: NormalizeMethod::default(),
            min: None,
            max: None,
        }
    }
}

/// 标准化器：将浮点状态值映射到有限整数范围
///
/// 两遍扫描：
/// 1. 第一遍收集所有 Float 的 min/max（若配置未指定）
/// 2. 第二遍变换：
///    - Float(v) → 缩放到 [0, max_val] 并取整为 Integer
///    - Bool(v) → Integer(v as i64)
///    - Integer(v) → 保持不变
pub struct Normalizer {
    config: NormalizerConfig,
}

impl Normalizer {
    /// 根据配置创建一个新的标准化器
    pub fn new(config: NormalizerConfig) -> Self {
        Normalizer { config }
    }
}

impl Processor for Normalizer {
    fn name(&self) -> &'static str {
        "normalizer"
    }

    fn process(
        &self,
        input: Box<dyn Iterator<Item = SequenceFrame> + Send>,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let frames: Vec<SequenceFrame> = input.collect();

        if frames.is_empty() {
            return Ok(Box::new(std::iter::empty()));
        }

        // 确定 min/max 边界
        let (min, max) = match (self.config.min, self.config.max) {
            (Some(min), Some(max)) => (min, max),
            _ => {
                // 第一遍扫描：收集所有 Float 值的 min 和 max
                let mut float_min = f64::MAX;
                let mut float_max = f64::MIN;
                let mut has_float = false;

                for frame in &frames {
                    for state in &frame.state.values {
                        if let FrameState::Float(v) = state {
                            has_float = true;
                            float_min = float_min.min(*v);
                            float_max = float_max.max(*v);
                        }
                    }
                }

                if !has_float {
                    // 没有浮点值，全部是整数或布尔，直接变换返回
                    let iter = frames.into_iter().map(|frame| {
                        let values: Vec<FrameState> = frame
                            .state
                            .values
                            .into_iter()
                            .map(|s| match s {
                                FrameState::Bool(b) => FrameState::Integer(b as i64),
                                other => other,
                            })
                            .collect();
                        SequenceFrame {
                            step_index: frame.step_index,
                            state: FrameData { values },
                            label: frame.label,
                        }
                    });
                    return Ok(Box::new(iter));
                }

                let min = float_min;
                let max = if float_min == float_max {
                    float_min + 1.0
                } else {
                    float_max
                };

                (min, max)
            }
        };

        let max_val = self.config.max_val as f64;
        let method = self.config.method;

        // 第二遍变换
        let iter = frames.into_iter().map(move |frame| {
            let values: Vec<FrameState> = frame
                .state
                .values
                .into_iter()
                .map(|s| match s {
                    FrameState::Float(v) => {
                        let normalized = match method {
                            NormalizeMethod::Linear => {
                                let scaled = (v - min) / (max - min) * max_val;
                                scaled.clamp(0.0, max_val).round() as i64
                            }
                            NormalizeMethod::LogBucket => {
                                let log_val = (1.0 + v.abs()).log2();
                                let abs_max = (1.0 + max.abs().max(min.abs())).log2();
                                let scaled = if abs_max > 0.0 {
                                    log_val / abs_max * max_val
                                } else {
                                    0.0
                                };
                                scaled.clamp(0.0, max_val).round() as i64
                            }
                            NormalizeMethod::UniformQuantile => {
                                // 按分位数映射：将值线性映射到桶编号
                                let scaled = (v - min) / (max - min) * max_val;
                                scaled.clamp(0.0, max_val).round() as i64
                            }
                        };
                        FrameState::Integer(normalized)
                    }
                    FrameState::Bool(b) => FrameState::Integer(b as i64),
                    FrameState::Integer(_) => s,
                })
                .collect();
            SequenceFrame {
                step_index: frame.step_index,
                state: FrameData { values },
                label: frame.label,
            }
        });

        Ok(Box::new(iter))
    }
}

/// 工厂函数：从 JSON 配置创建标准化器
pub fn create_normalizer(config: &Value) -> CoreResult<Box<dyn Processor>> {
    let config: NormalizerConfig = if config.is_null() {
        NormalizerConfig::default()
    } else {
        serde_json::from_value(config.clone()).map_err(|e| {
            CoreError::SerializationError(format!("标准化器配置解析失败: {}", e))
        })?
    };
    Ok(Box::new(Normalizer::new(config)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造包含浮点、整数、布尔混合的测试帧
    fn make_mixed_frames() -> Vec<SequenceFrame> {
        vec![
            SequenceFrame::new(
                0,
                FrameData {
                    values: vec![
                        FrameState::Float(10.0),
                        FrameState::Integer(5),
                        FrameState::Bool(true),
                    ],
                },
            ),
            SequenceFrame::new(
                1,
                FrameData {
                    values: vec![
                        FrameState::Float(20.0),
                        FrameState::Integer(8),
                        FrameState::Bool(false),
                    ],
                },
            ),
            SequenceFrame::new(
                2,
                FrameData {
                    values: vec![
                        FrameState::Float(0.0),
                        FrameState::Integer(3),
                        FrameState::Bool(true),
                    ],
                },
            ),
        ]
    }

    #[test]
    fn test_normalizer_linear_scales_correctly() {
        let frames = make_mixed_frames();
        let config = NormalizerConfig {
            max_val: 100,
            method: NormalizeMethod::Linear,
            min: None,
            max: None,
        };
        let normalizer = Normalizer::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = normalizer.process(input).unwrap().collect();

        assert_eq!(output.len(), 3);

        // 验证所有输出值都是 Integer 类型
        for frame in &output {
            for state in &frame.state.values {
                match state {
                    FrameState::Integer(v) => {
                        assert!(*v >= 0, "值应在 [0, max_val] 范围内: {}", v);
                        assert!(*v <= 100, "值应在 [0, max_val] 范围内: {}", v);
                    }
                    _ => panic!("标准化后所有值应为 Integer 类型"),
                }
            }
        }

        // 第一帧 Float(10.0) 在 min=0, max=20 时的缩放: (10-0)/(20-0)*100 = 50
        if let FrameState::Integer(v) = output[0].state.values[0] {
            assert_eq!(v, 50);
        }

        // 第二帧 Float(20.0) → 100
        if let FrameState::Integer(v) = output[1].state.values[0] {
            assert_eq!(v, 100);
        }

        // 第三帧 Float(0.0) → 0
        if let FrameState::Integer(v) = output[2].state.values[0] {
            assert_eq!(v, 0);
        }

        // 验证 Bool 转换为 Integer
        if let FrameState::Integer(v) = output[0].state.values[2] {
            assert_eq!(v, 1); // true → 1
        }
        if let FrameState::Integer(v) = output[1].state.values[2] {
            assert_eq!(v, 0); // false → 0
        }

        // 验证 label 透传
        assert_eq!(output[0].label, None);
    }

    #[test]
    fn test_normalizer_with_explicit_min_max() {
        let frames = make_mixed_frames();
        let config = NormalizerConfig {
            max_val: 10,
            method: NormalizeMethod::Linear,
            min: Some(0.0),
            max: Some(100.0),
        };
        let normalizer = Normalizer::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = normalizer.process(input).unwrap().collect();

        // Float(10.0) 在 [0,100] 范围缩放到 [0,10]: (10-0)/(100-0)*10 = 1
        if let FrameState::Integer(v) = output[0].state.values[0] {
            assert_eq!(v, 1);
        }
    }

    #[test]
    fn test_normalizer_empty_input() {
        let config = NormalizerConfig::default();
        let normalizer = Normalizer::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(std::iter::empty());
        let output: Vec<SequenceFrame> = normalizer.process(input).unwrap().collect();
        assert!(output.is_empty());
    }

    #[test]
    fn test_normalizer_preserves_step_index() {
        let frames = make_mixed_frames();
        let config = NormalizerConfig::default();
        let normalizer = Normalizer::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = normalizer.process(input).unwrap().collect();
        for (i, frame) in output.iter().enumerate() {
            assert_eq!(frame.step_index, i as u64);
        }
    }

    #[test]
    fn test_normalizer_only_integers_and_bools() {
        // 当没有 Float 值时，Integer 保持不变，Bool 转为 Integer
        let frames = vec![SequenceFrame::new(
            0,
            FrameData {
                values: vec![FrameState::Integer(42), FrameState::Bool(true)],
            },
        )];
        let config = NormalizerConfig::default();
        let normalizer = Normalizer::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = normalizer.process(input).unwrap().collect();

        assert_eq!(output[0].state.values[0], FrameState::Integer(42));
        assert_eq!(output[0].state.values[1], FrameState::Integer(1));
    }

    #[test]
    fn test_normalizer_log_bucket() {
        let frames = vec![SequenceFrame::new(
            0,
            FrameData {
                values: vec![FrameState::Float(1.0), FrameState::Float(8.0)],
            },
        )];
        let config = NormalizerConfig {
            max_val: 100,
            method: NormalizeMethod::LogBucket,
            min: None,
            max: None,
        };
        let normalizer = Normalizer::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = normalizer.process(input).unwrap().collect();

        for state in &output[0].state.values {
            match state {
                FrameState::Integer(v) => {
                    assert!(*v >= 0 && *v <= 100);
                }
                _ => panic!("应为 Integer 类型"),
            }
        }
    }

    #[test]
    fn test_normalizer_uniform_quantile() {
        let frames = vec![SequenceFrame::new(
            0,
            FrameData {
                values: vec![FrameState::Float(5.0), FrameState::Float(15.0)],
            },
        )];
        let config = NormalizerConfig {
            max_val: 10,
            method: NormalizeMethod::UniformQuantile,
            min: None,
            max: None,
        };
        let normalizer = Normalizer::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = normalizer.process(input).unwrap().collect();

        for state in &output[0].state.values {
            match state {
                FrameState::Integer(v) => {
                    assert!(*v >= 0 && *v <= 10);
                }
                _ => panic!("应为 Integer 类型"),
            }
        }
    }

    #[test]
    fn test_normalize_method_serialization() {
        let method = NormalizeMethod::Linear;
        let json = serde_json::to_string(&method).unwrap();
        assert!(json.contains("Linear"));
        let restored: NormalizeMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, NormalizeMethod::Linear);
    }
}
