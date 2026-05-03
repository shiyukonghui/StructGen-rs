use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::{CoreError, CoreResult, FrameData, FrameState, SequenceFrame};

use super::processor::Processor;

/// 去重过滤器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupConfig {
    /// 是否移除连续重复帧（前后帧 FrameData 完全相同）
    #[serde(default = "default_true")]
    pub remove_consecutive_duplicates: bool,
    /// 是否移除全零帧（所有值为 Integer(0)/Bool(false)/Float(0.0)）
    #[serde(default = "default_true")]
    pub remove_all_zeros: bool,
    /// 最小熵阈值：移除熵值低于此阈值的帧（0.0 表示不过滤）
    #[serde(default)]
    pub min_entropy: f64,
}

fn default_true() -> bool {
    true
}

impl Default for DedupConfig {
    fn default() -> Self {
        DedupConfig {
            remove_consecutive_duplicates: true,
            remove_all_zeros: true,
            min_entropy: 0.0,
        }
    }
}

/// 去重过滤器：移除连续重复帧、全零帧和低熵帧
pub struct DedupFilter {
    config: DedupConfig,
}

/// 去重迭代器适配器：包装输入迭代器，惰性地跳过应被过滤的帧
struct DedupIter<I: Iterator<Item = SequenceFrame>> {
    inner: I,
    config: DedupConfig,
    prev_frame: Option<FrameData>,
}

impl<I: Iterator<Item = SequenceFrame>> Iterator for DedupIter<I> {
    type Item = SequenceFrame;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let frame = self.inner.next()?;

            // 检查全零帧
            if self.config.remove_all_zeros && is_all_zero(&frame.state) {
                continue;
            }

            // 检查连续重复帧
            if self.config.remove_consecutive_duplicates {
                if let Some(ref prev) = self.prev_frame {
                    if *prev == frame.state {
                        continue;
                    }
                }
            }

            // 检查低熵
            if self.config.min_entropy > 0.0 {
                let entropy = estimate_entropy(&frame.state);
                if entropy < self.config.min_entropy {
                    continue;
                }
            }

            self.prev_frame = Some(frame.state.clone());
            return Some(frame);
        }
    }
}

/// 判断帧数据是否为全零
fn is_all_zero(data: &FrameData) -> bool {
    data.values.iter().all(|s| match s {
        FrameState::Integer(v) => *v == 0,
        FrameState::Float(v) => *v == 0.0,
        FrameState::Bool(v) => !*v,
    })
}

/// 估算帧数据的香农熵（基于 256-bin 直方图）
/// 返回归一化熵值 H = -Σ p_i * log2(p_i)，归一化到 [0, 1]
fn estimate_entropy(data: &FrameData) -> f64 {
    let n = data.values.len();
    if n == 0 {
        return 0.0;
    }

    // 构建 256-bin 直方图
    let mut histogram = [0u64; 256];
    for state in &data.values {
        let bin = match state {
            FrameState::Integer(v) => {
                (v.unsigned_abs() % 256) as usize
            }
            FrameState::Float(v) => {
                let abs_v = v.abs();
                (abs_v as u64 % 256) as usize
            }
            FrameState::Bool(v) => {
                if *v { 1 } else { 0 }
            }
        };
        histogram[bin] += 1;
    }

    // 计算香农熵: H = -Σ p_i * log2(p_i)
    let n_f64 = n as f64;
    let entropy: f64 = histogram
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / n_f64;
            -p * p.log2()
        })
        .sum();

    // 归一化到 [0, 1]：最大可能熵为 log2(256) = 8
    let max_entropy = 8.0;
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        entropy
    }
}

impl DedupFilter {
    /// 根据配置创建去重过滤器
    pub fn new(config: DedupConfig) -> Self {
        DedupFilter { config }
    }
}

impl Processor for DedupFilter {
    fn name(&self) -> &'static str {
        "dedup"
    }

    fn process(
        &self,
        input: Box<dyn Iterator<Item = SequenceFrame> + Send>,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let iter = DedupIter {
            inner: input,
            config: self.config.clone(),
            prev_frame: None,
        };
        Ok(Box::new(iter))
    }
}

/// 工厂函数：从 JSON 配置创建去重过滤器
pub fn create_dedup(config: &Value) -> CoreResult<Box<dyn Processor>> {
    let config: DedupConfig = if config.is_null() {
        DedupConfig::default()
    } else {
        serde_json::from_value(config.clone()).map_err(|e| {
            CoreError::SerializationError(format!("去重过滤器配置解析失败: {}", e))
        })?
    };
    Ok(Box::new(DedupFilter::new(config)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(step: u64, values: Vec<FrameState>) -> SequenceFrame {
        SequenceFrame::new(step, FrameData { values })
    }

    #[test]
    fn test_removes_consecutive_duplicates() {
        let frames = vec![
            make_frame(0, vec![FrameState::Integer(1)]),
            make_frame(1, vec![FrameState::Integer(1)]), // 重复
            make_frame(2, vec![FrameState::Integer(1)]), // 重复
            make_frame(3, vec![FrameState::Integer(2)]),
            make_frame(4, vec![FrameState::Integer(1)]),
        ];
        let config = DedupConfig {
            remove_consecutive_duplicates: true,
            remove_all_zeros: false,
            min_entropy: 0.0,
        };
        let filter = DedupFilter::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = filter.process(input).unwrap().collect();

        // 应保留: step 0 (Integer(1)), step 3 (Integer(2)), step 4 (Integer(1))
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].step_index, 0);
        assert_eq!(output[1].step_index, 3);
        assert_eq!(output[2].step_index, 4);
    }

    #[test]
    fn test_removes_all_zeros() {
        let frames = vec![
            make_frame(0, vec![FrameState::Integer(0), FrameState::Bool(false)]),
            make_frame(1, vec![FrameState::Integer(1), FrameState::Bool(true)]),
            make_frame(2, vec![FrameState::Float(0.0), FrameState::Integer(0)]),
        ];
        let config = DedupConfig {
            remove_consecutive_duplicates: false,
            remove_all_zeros: true,
            min_entropy: 0.0,
        };
        let filter = DedupFilter::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = filter.process(input).unwrap().collect();

        assert_eq!(output.len(), 1);
        assert_eq!(output[0].step_index, 1);
    }

    #[test]
    fn test_removes_low_entropy() {
        // 全相同值的帧熵为 0
        let frames = vec![
            make_frame(0, vec![FrameState::Integer(5); 100]),
            make_frame(1, vec![FrameState::Integer(1), FrameState::Integer(2), FrameState::Integer(3)]),
        ];
        let config = DedupConfig {
            remove_consecutive_duplicates: false,
            remove_all_zeros: false,
            min_entropy: 0.1,
        };
        let filter = DedupFilter::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = filter.process(input).unwrap().collect();

        assert_eq!(output.len(), 1);
        assert_eq!(output[0].step_index, 1);
    }

    #[test]
    fn test_keep_all_when_disabled() {
        let frames = vec![
            make_frame(0, vec![FrameState::Integer(1)]),
            make_frame(1, vec![FrameState::Integer(1)]), // 重复
        ];
        let config = DedupConfig {
            remove_consecutive_duplicates: false,
            remove_all_zeros: false,
            min_entropy: 0.0,
        };
        let filter = DedupFilter::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = filter.process(input).unwrap().collect();

        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_empty_input() {
        let config = DedupConfig::default();
        let filter = DedupFilter::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(std::iter::empty());
        let output: Vec<SequenceFrame> = filter.process(input).unwrap().collect();
        assert!(output.is_empty());
    }

    #[test]
    fn test_entropy_of_empty_frame() {
        let data = FrameData { values: vec![] };
        let entropy = estimate_entropy(&data);
        assert!((entropy - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_entropy_of_uniform_frame() {
        // 所有值相同 → 熵为 0
        let data = FrameData {
            values: vec![FrameState::Integer(5); 50],
        };
        let entropy = estimate_entropy(&data);
        assert!((entropy - 0.0).abs() < f64::EPSILON);
    }
}
