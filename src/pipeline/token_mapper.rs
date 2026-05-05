use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::{CoreError, CoreResult, FrameData, FrameState, SequenceFrame};

use super::processor::Processor;

/// Unicode 最大码点值
const UNICODE_MAX: u32 = 0x10_FFFF;

/// 令牌映射器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenMapperConfig {
    /// 映射起始 Unicode 码点，如 0x4E00（CJK 统一汉字起始）
    #[serde(default = "default_start_codepoint")]
    pub start_codepoint: u32,
    /// 是否插入换行符作为帧分隔
    #[serde(default)]
    pub insert_newline: bool,
}

fn default_start_codepoint() -> u32 {
    0x4E00
}

impl Default for TokenMapperConfig {
    fn default() -> Self {
        TokenMapperConfig {
            start_codepoint: default_start_codepoint(),
            insert_newline: false,
        }
    }
}

/// 令牌映射器：将已离散化的整数值映射到 Unicode 字符码点
///
/// 映射逻辑：
/// - Integer(v) → start_codepoint + v 钳位到 [0, 0x10FFFF]
/// - Float(v) → start_codepoint + ((v * 256) as u32).clamp(0, 255)
/// - Bool(v) → start_codepoint + (v as u32)
///   如果 insert_newline，每帧值后添加换行码点（0x0A → Integer(10)）
pub struct TokenMapper {
    config: TokenMapperConfig,
}

/// 令牌映射迭代器适配器
struct TokenMapIter<I: Iterator<Item = SequenceFrame>> {
    inner: I,
    config: TokenMapperConfig,
}

impl<I: Iterator<Item = SequenceFrame>> Iterator for TokenMapIter<I> {
    type Item = SequenceFrame;

    fn next(&mut self) -> Option<Self::Item> {
        let frame = self.inner.next()?;

        let mut values: Vec<FrameState> = frame
            .state
            .values
            .iter()
            .map(|s| {
                let codepoint = match s {
                    FrameState::Integer(v) => {
                        let cp = (self.config.start_codepoint as i64).saturating_add(*v);
                        // 钳位到 [0, UNICODE_MAX]
                        if cp < 0 {
                            0
                        } else if cp > UNICODE_MAX as i64 {
                            UNICODE_MAX
                        } else {
                            cp as u32
                        }
                    }
                    FrameState::Float(v) => {
                        let offset = ((*v * 256.0) as i64).clamp(0, 255);
                        (self.config.start_codepoint as i64 + offset) as u32
                    }
                    FrameState::Bool(v) => {
                        self.config.start_codepoint + (*v as u32)
                    }
                };
                // 最终钳位确保不超过 Unicode 范围
                let safe_cp = codepoint.min(UNICODE_MAX);
                FrameState::Integer(safe_cp as i64)
            })
            .collect();

        // 如果设置了 insert_newline，在帧尾添加换行符码点
        if self.config.insert_newline {
            values.push(FrameState::Integer(0x0A));
        }

        Some(SequenceFrame {
            step_index: frame.step_index,
            state: FrameData { values },
            label: frame.label,
            sample_id: None,
        })
    }
}

impl TokenMapper {
    /// 根据配置创建令牌映射器
    pub fn new(config: TokenMapperConfig) -> Self {
        TokenMapper { config }
    }
}

impl Processor for TokenMapper {
    fn name(&self) -> &'static str {
        "token_mapper"
    }

    fn process(
        &self,
        input: Box<dyn Iterator<Item = SequenceFrame> + Send>,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let iter = TokenMapIter {
            inner: input,
            config: self.config.clone(),
        };
        Ok(Box::new(iter))
    }
}

/// 工厂函数：从 JSON 配置创建令牌映射器
pub fn create_token_mapper(config: &Value) -> CoreResult<Box<dyn Processor>> {
    let config: TokenMapperConfig = if config.is_null() {
        TokenMapperConfig::default()
    } else {
        serde_json::from_value(config.clone()).map_err(|e| {
            CoreError::SerializationError(format!("令牌映射器配置解析失败: {}", e))
        })?
    };
    Ok(Box::new(TokenMapper::new(config)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(step: u64, values: Vec<FrameState>) -> SequenceFrame {
        SequenceFrame::new(step, FrameData { values })
    }

    #[test]
    fn test_token_mapper_output_in_unicode_range() {
        let frames = vec![
            make_frame(0, vec![FrameState::Integer(0), FrameState::Integer(10)]),
            make_frame(1, vec![FrameState::Bool(true), FrameState::Bool(false)]),
            make_frame(2, vec![FrameState::Float(0.5), FrameState::Float(0.0)]),
        ];
        let config = TokenMapperConfig::default();
        let mapper = TokenMapper::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = mapper.process(input).unwrap().collect();

        assert_eq!(output.len(), 3);

        // 验证所有输出码点在 Unicode 安全范围内
        for frame in &output {
            for state in &frame.state.values {
                match state {
                    FrameState::Integer(v) => {
                        assert!(*v >= 0, "码点不能为负: {}", v);
                        assert!(
                            *v as u64 <= UNICODE_MAX as u64,
                            "码点超出 Unicode 范围: {} > {}",
                            v,
                            UNICODE_MAX
                        );
                    }
                    _ => panic!("令牌映射后所有值应为 Integer 类型"),
                }
            }
        }

        // Integer(0) 映射到 start_codepoint
        if let FrameState::Integer(v) = output[0].state.values[0] {
            assert_eq!(v, 0x4E00);
        }
        if let FrameState::Integer(v) = output[0].state.values[1] {
            assert_eq!(v, 0x4E00 + 10);
        }
    }

    #[test]
    fn test_token_mapper_bool_mapping() {
        let frames = vec![
            make_frame(0, vec![FrameState::Bool(true), FrameState::Bool(false)]),
        ];
        let config = TokenMapperConfig {
            start_codepoint: 0x4E00,
            insert_newline: false,
        };
        let mapper = TokenMapper::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = mapper.process(input).unwrap().collect();

        if let FrameState::Integer(v) = output[0].state.values[0] {
            assert_eq!(v, 0x4E01); // start + 1 (true)
        }
        if let FrameState::Integer(v) = output[0].state.values[1] {
            assert_eq!(v, 0x4E00); // start + 0 (false)
        }
    }

    #[test]
    fn test_token_mapper_float_mapping() {
        let frames = vec![
            make_frame(0, vec![FrameState::Float(0.0), FrameState::Float(0.5)]),
        ];
        let config = TokenMapperConfig {
            start_codepoint: 0x4E00,
            insert_newline: false,
        };
        let mapper = TokenMapper::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = mapper.process(input).unwrap().collect();

        // Float(0.0) → offset = (0.0 * 256) = 0
        if let FrameState::Integer(v) = output[0].state.values[0] {
            assert_eq!(v, 0x4E00);
        }
        // Float(0.5) → offset = (0.5 * 256) = 128
        if let FrameState::Integer(v) = output[0].state.values[1] {
            assert_eq!(v, 0x4E00 + 128);
        }
    }

    #[test]
    fn test_token_mapper_insert_newline() {
        let frames = vec![
            make_frame(0, vec![FrameState::Integer(1)]),
        ];
        let config = TokenMapperConfig {
            start_codepoint: 0x4E00,
            insert_newline: true,
        };
        let mapper = TokenMapper::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = mapper.process(input).unwrap().collect();

        // 应有 2 个值：映射值 + 换行符
        assert_eq!(output[0].state.values.len(), 2);
        assert_eq!(output[0].state.values[1], FrameState::Integer(0x0A));
    }

    #[test]
    fn test_token_mapper_clamps_to_unicode_max() {
        // 构造一个非常大的整数值，验证钳位
        let frames = vec![
            make_frame(0, vec![FrameState::Integer(i64::MAX)]),
        ];
        let config = TokenMapperConfig::default();
        let mapper = TokenMapper::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = mapper.process(input).unwrap().collect();

        if let FrameState::Integer(v) = output[0].state.values[0] {
            assert!(
                v <= UNICODE_MAX as i64,
                "钳位失败: {} > {}",
                v,
                UNICODE_MAX
            );
        }
    }

    #[test]
    fn test_token_mapper_negative_integer_clamped() {
        let frames = vec![
            make_frame(0, vec![FrameState::Integer(-100000)]),
        ];
        let config = TokenMapperConfig::default();
        let mapper = TokenMapper::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = mapper.process(input).unwrap().collect();

        // 负值应被钳位到 0
        if let FrameState::Integer(v) = output[0].state.values[0] {
            assert!(v >= 0, "负值应钳位到 0: {}", v);
        }
    }
}
