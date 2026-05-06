//! Patch Tokenization 后处理器
//!
//! 等价实现 Python NCA_Tokenizer 的 patch tokenization 逻辑。
//! 将 2D 网格帧转换为 patch token 序列，每个 patch 通过 base-N 编码为单个 token。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::{CoreError, CoreResult, FrameData, FrameState, SequenceFrame};

use super::processor::Processor;

/// Patch Tokenizer 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchTokenizerConfig {
    /// 正方形 patch 边长
    pub patch: usize,
    /// 滑动步长，None 则等于 patch（无重叠）
    #[serde(default)]
    pub stride: Option<usize>,
    /// 颜色数 (= d_state)
    #[serde(default = "default_num_colors")]
    pub num_colors: usize,
    /// 网格行数
    pub rows: usize,
    /// 网格列数
    pub cols: usize,
    /// 独立规则组数，默认 1
    #[serde(default = "default_n_groups")]
    pub n_groups: usize,
}

fn default_num_colors() -> usize {
    10
}
fn default_n_groups() -> usize {
    1
}

/// Patch Tokenization 后处理器
///
/// 将 2D 网格帧转换为 patch token 序列。
/// 编码算法：
/// 1. 将 (rows, cols) 网格切分为 (N_H, N_W) 个 patch
/// 2. 每个 patch 做 base-N 编码：token = Σ val[i] * (num_colors ^ i)
/// 3. 添加 start_token 和 end_token
pub struct PatchTokenizer {
    config: PatchTokenizerConfig,
    /// 预计算的 powers 表: num_colors^0, num_colors^1, ..., num_colors^(patch*patch-1)
    powers: Vec<i64>,
    /// start token ID
    start_token: i64,
    /// end token ID
    end_token: i64,
}

/// Patch tokenization 迭代器适配器
struct PatchTokenizeIter<I: Iterator<Item = SequenceFrame>> {
    inner: I,
    patch: usize,
    stride: usize,
    num_colors: usize,
    rows: usize,
    cols: usize,
    n_groups: usize,
    powers: Vec<i64>,
    start_token: i64,
    end_token: i64,
}

impl<I: Iterator<Item = SequenceFrame>> Iterator for PatchTokenizeIter<I> {
    type Item = SequenceFrame;

    fn next(&mut self) -> Option<Self::Item> {
        let frame = self.inner.next()?;

        let n_h = self.rows / self.patch;
        let n_w = self.cols / self.patch;
        let patch_sq = self.patch * self.patch;

        // 每个 group 独立 tokenization
        let tokens_per_group = n_h * n_w;
        let total_tokens = 2 + self.n_groups * tokens_per_group; // start + n_groups * patches + end

        let mut tokens = Vec::with_capacity(total_tokens);
        tokens.push(self.start_token);

        for g in 0..self.n_groups {
            for ph in 0..n_h {
                for pw in 0..n_w {
                    let mut token: i64 = 0;
                    for di in 0..self.patch {
                        for dj in 0..self.patch {
                            let r = ph * self.stride + di;
                            let c = pw * self.stride + dj;
                            let idx = (r * self.cols + c) * self.n_groups + g;
                            let val = if idx < frame.state.values.len() {
                                match &frame.state.values[idx] {
                                    FrameState::Integer(v) => *v,
                                    FrameState::Float(v) => *v as i64,
                                    FrameState::Bool(v) => *v as i64,
                                }
                            } else {
                                0
                            };
                            let local_idx = di * self.patch + dj;
                            token += val * self.powers[local_idx % patch_sq];
                        }
                    }
                    tokens.push(token);
                }
            }
        }

        tokens.push(self.end_token);

        let values: Vec<FrameState> = tokens.into_iter().map(FrameState::Integer).collect();

        Some(SequenceFrame {
            step_index: frame.step_index,
            state: FrameData { values },
            label: frame.label,
            sample_id: None,
        })
    }
}

impl PatchTokenizer {
    /// 根据配置创建 PatchTokenizer，校验参数合法性
    pub fn new(config: PatchTokenizerConfig) -> CoreResult<Self> {
        // 校验 patch >= 1
        if config.patch == 0 {
            return Err(CoreError::InvalidParams(
                "patch_tokenizer: patch must be >= 1".into(),
            ));
        }
        // 校验 rows 和 cols 能被 patch 整除
        if config.rows % config.patch != 0 {
            return Err(CoreError::InvalidParams(format!(
                "patch_tokenizer: rows ({}) must be divisible by patch ({})",
                config.rows, config.patch
            )));
        }
        if config.cols % config.patch != 0 {
            return Err(CoreError::InvalidParams(format!(
                "patch_tokenizer: cols ({}) must be divisible by patch ({})",
                config.cols, config.patch
            )));
        }
        // 校验 num_colors >= 2
        if config.num_colors < 2 {
            return Err(CoreError::InvalidParams(
                "patch_tokenizer: num_colors must be >= 2".into(),
            ));
        }
        // 校验 num_colors^(patch*patch) 不超出 i64::MAX
        let patch_sq = config.patch * config.patch;
        let max_token = (config.num_colors as f64).powi(patch_sq as i32);
        if max_token > i64::MAX as f64 {
            return Err(CoreError::InvalidParams(format!(
                "patch_tokenizer: num_colors^({}) overflows i64 ({}^{} = {:.2e})",
                patch_sq, config.num_colors, patch_sq, max_token
            )));
        }

        // 预计算 powers 表
        let powers: Vec<i64> = (0..patch_sq)
            .map(|i| (config.num_colors as i64).pow(i as u32))
            .collect();

        let start_token = (config.num_colors as i64).pow(patch_sq as u32);
        let end_token = start_token + 1;

        Ok(PatchTokenizer {
            config,
            powers,
            start_token,
            end_token,
        })
    }
}

impl Processor for PatchTokenizer {
    fn name(&self) -> &'static str {
        "patch_tokenizer"
    }

    fn process(
        &self,
        input: Box<dyn Iterator<Item = SequenceFrame> + Send>,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let stride = self.config.stride.unwrap_or(self.config.patch);

        let iter = PatchTokenizeIter {
            inner: input,
            patch: self.config.patch,
            stride,
            num_colors: self.config.num_colors,
            rows: self.config.rows,
            cols: self.config.cols,
            n_groups: self.config.n_groups,
            powers: self.powers.clone(),
            start_token: self.start_token,
            end_token: self.end_token,
        };
        Ok(Box::new(iter))
    }
}

/// 工厂函数：从 JSON 配置创建 PatchTokenizer
pub fn create_patch_tokenizer(config: &Value) -> CoreResult<Box<dyn Processor>> {
    let config: PatchTokenizerConfig = if config.is_null() {
        return Err(CoreError::InvalidParams(
            "patch_tokenizer requires configuration (patch, rows, cols, num_colors)".into(),
        ));
    } else {
        serde_json::from_value(config.clone()).map_err(|e| {
            CoreError::SerializationError(format!("patch_tokenizer 配置解析失败: {}", e))
        })?
    };
    let tokenizer = PatchTokenizer::new(config)?;
    Ok(Box::new(tokenizer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::registry::ProcessorRegistry;
    use crate::pipeline::register_all;

    fn make_frame(step: u64, values: Vec<FrameState>) -> SequenceFrame {
        SequenceFrame::new(step, FrameData { values })
    }

    fn make_grid_frame(step: u64, rows: usize, cols: usize, n_groups: usize, data: &[u8]) -> SequenceFrame {
        let values: Vec<FrameState> = data.iter().map(|&v| FrameState::Integer(v as i64)).collect();
        assert_eq!(values.len(), rows * cols * n_groups);
        make_frame(step, values)
    }

    #[test]
    fn test_patch_tokenize_basic_4x4() {
        // 4×4 网格, patch=2, num_colors=10, n_groups=1
        let grid: Vec<u8> = vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
            0, 1, 2, 3,
            4, 5, 6, 7,
        ];
        let frame = make_grid_frame(0, 4, 4, 1, &grid);

        let config = PatchTokenizerConfig {
            patch: 2,
            stride: None,
            num_colors: 10,
            rows: 4,
            cols: 4,
            n_groups: 1,
        };
        let tokenizer = PatchTokenizer::new(config).unwrap();
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> = Box::new(std::iter::once(frame));
        let output: Vec<SequenceFrame> = tokenizer.process(input).unwrap().collect();

        assert_eq!(output.len(), 1);
        // 1 start + 4 patches + 1 end = 6 tokens
        assert_eq!(output[0].state.values.len(), 6);
    }

    #[test]
    fn test_patch_tokenize_base_n_encoding() {
        // 2×2 网格, patch=2, num_colors=10, n_groups=1
        // 整个网格就是一个 patch
        // grid = [[5, 3], [1, 7]]
        // powers = [1, 10, 100, 1000]
        // token = 5*1 + 3*10 + 1*100 + 7*1000 = 5 + 30 + 100 + 7000 = 7135
        let grid: Vec<u8> = vec![5, 3, 1, 7];
        let frame = make_grid_frame(0, 2, 2, 1, &grid);

        let config = PatchTokenizerConfig {
            patch: 2,
            stride: None,
            num_colors: 10,
            rows: 2,
            cols: 2,
            n_groups: 1,
        };
        let tokenizer = PatchTokenizer::new(config).unwrap();
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> = Box::new(std::iter::once(frame));
        let output: Vec<SequenceFrame> = tokenizer.process(input).unwrap().collect();

        assert_eq!(output.len(), 1);
        let values = &output[0].state.values;
        // start_token = 10^4 = 10000
        assert_eq!(values[0], FrameState::Integer(10000));
        // patch token = 7135
        assert_eq!(values[1], FrameState::Integer(7135));
        // end_token = 10001
        assert_eq!(values[2], FrameState::Integer(10001));
    }

    #[test]
    fn test_patch_tokenize_start_end_tokens() {
        let config = PatchTokenizerConfig {
            patch: 2,
            stride: None,
            num_colors: 10,
            rows: 2,
            cols: 2,
            n_groups: 1,
        };
        let tokenizer = PatchTokenizer::new(config).unwrap();

        // start_token = 10^(2*2) = 10000
        assert_eq!(tokenizer.start_token, 10000);
        assert_eq!(tokenizer.end_token, 10001);
    }

    #[test]
    fn test_patch_tokenize_preserves_step_index() {
        let grid1: Vec<u8> = vec![0; 4];
        let grid2: Vec<u8> = vec![1; 4];
        let frame1 = make_grid_frame(5, 2, 2, 1, &grid1);
        let frame2 = make_grid_frame(10, 2, 2, 1, &grid2);

        let config = PatchTokenizerConfig {
            patch: 2,
            stride: None,
            num_colors: 10,
            rows: 2,
            cols: 2,
            n_groups: 1,
        };
        let tokenizer = PatchTokenizer::new(config).unwrap();
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(vec![frame1, frame2].into_iter());
        let output: Vec<SequenceFrame> = tokenizer.process(input).unwrap().collect();

        assert_eq!(output[0].step_index, 5);
        assert_eq!(output[1].step_index, 10);
    }

    #[test]
    fn test_patch_tokenize_invalid_patch_size() {
        let config = PatchTokenizerConfig {
            patch: 3,
            stride: None,
            num_colors: 10,
            rows: 4,
            cols: 4,
            n_groups: 1,
        };
        let result = PatchTokenizer::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_patch_tokenize_zero_patch_rejected() {
        let config = PatchTokenizerConfig {
            patch: 0,
            stride: None,
            num_colors: 10,
            rows: 4,
            cols: 4,
            n_groups: 1,
        };
        let result = PatchTokenizer::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_patch_tokenize_overflow_protection() {
        // patch=4, num_colors=16 → 16^16 > i64::MAX
        let config = PatchTokenizerConfig {
            patch: 4,
            stride: None,
            num_colors: 16,
            rows: 4,
            cols: 4,
            n_groups: 1,
        };
        let result = PatchTokenizer::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_patch_tokenize_n_groups_2() {
        // 2×2 网格, patch=2, n_groups=2 → 每个 group 1 个 patch
        // 2 groups × 1 patch = 2 patch tokens + 2 special = 4
        let grid: Vec<u8> = vec![
            // pixel (0,0): group0=1, group1=2
            1, 2,
            // pixel (0,1): group0=3, group1=4
            3, 4,
            // pixel (1,0): group0=5, group1=6
            5, 6,
            // pixel (1,1): group0=7, group1=8
            7, 8,
        ];
        let frame = make_grid_frame(0, 2, 2, 2, &grid);

        let config = PatchTokenizerConfig {
            patch: 2,
            stride: None,
            num_colors: 10,
            rows: 2,
            cols: 2,
            n_groups: 2,
        };
        let tokenizer = PatchTokenizer::new(config).unwrap();
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> = Box::new(std::iter::once(frame));
        let output: Vec<SequenceFrame> = tokenizer.process(input).unwrap().collect();

        assert_eq!(output.len(), 1);
        // start + 2 groups * 1 patch + end = 4
        assert_eq!(output[0].state.values.len(), 4);

        // 验证 group0 的 patch token: grid values [1,3,5,7] → 1*1 + 3*10 + 5*100 + 7*1000 = 7531
        assert_eq!(output[0].state.values[1], FrameState::Integer(7531));
        // group1 的 patch token: grid values [2,4,6,8] → 2*1 + 4*10 + 6*100 + 8*1000 = 8642
        assert_eq!(output[0].state.values[2], FrameState::Integer(8642));
    }

    #[test]
    fn test_patch_tokenize_via_registry() {
        let mut registry = ProcessorRegistry::new();
        register_all(&mut registry).unwrap();

        let config = serde_json::json!({
            "patch": 2,
            "num_colors": 10,
            "rows": 2,
            "cols": 2,
            "n_groups": 1
        });

        let processor = registry.get("patch_tokenizer", &config);
        assert!(processor.is_ok(), "Should be able to instantiate patch_tokenizer via registry");
        assert_eq!(processor.unwrap().name(), "patch_tokenizer");
    }
}
