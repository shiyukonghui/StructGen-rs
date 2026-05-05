use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use super::rng::SeedRng;
use crate::core::*;

/// 2D 神经元胞自动机专用参数
#[derive(Debug, Clone, Deserialize)]
struct Nca2dParams {
    #[serde(default = "default_d_state_nca")]
    d_state: u8,
    #[serde(default = "default_n_groups")]
    n_groups: u8,
    #[serde(default = "default_rows_nca")]
    rows: usize,
    #[serde(default = "default_cols_nca")]
    cols: usize,
    #[serde(default)]
    identity_bias: f64,
    #[serde(default)]
    temperature: f64,
    #[serde(default = "default_hidden_dim")]
    hidden_dim: usize,
    #[serde(default = "default_conv_features")]
    conv_features: usize,
}

fn default_d_state_nca() -> u8 {
    10
}
fn default_n_groups() -> u8 {
    1
}
fn default_rows_nca() -> usize {
    12
}
fn default_cols_nca() -> usize {
    12
}
fn default_hidden_dim() -> usize {
    16
}
fn default_conv_features() -> usize {
    4
}

impl Default for Nca2dParams {
    fn default() -> Self {
        Nca2dParams {
            d_state: default_d_state_nca(),
            n_groups: default_n_groups(),
            rows: default_rows_nca(),
            cols: default_cols_nca(),
            identity_bias: 0.0,
            temperature: 0.0,
            hidden_dim: default_hidden_dim(),
            conv_features: default_conv_features(),
        }
    }
}

/// NCA 神经网络权重
///
/// 从 SeedRng 顺序采样初始化，每个种子 → 唯一权重 → 唯一 NCA 规则。
/// 初始化顺序：conv3x3_kernel → conv3x3_bias → fc1_weight → fc1_bias → fc2_weight → fc2_bias
struct NcaWeights {
    /// [conv_features × channels × 3 × 3]，channels = d_state * n_groups
    conv3x3_kernel: Vec<f64>,
    /// [conv_features]
    conv3x3_bias: Vec<f64>,
    /// [hidden_dim × conv_features]
    fc1_weight: Vec<f64>,
    /// [hidden_dim]
    fc1_bias: Vec<f64>,
    /// [channels × hidden_dim]，channels = d_state * n_groups
    fc2_weight: Vec<f64>,
    /// [channels]
    fc2_bias: Vec<f64>,
}

impl NcaWeights {
    /// 从 PRNG 初始化所有权重
    fn init_from_rng(rng: &mut SeedRng, conv_features: usize, hidden_dim: usize, channels: usize) -> Self {
        let init_weight = |rng: &mut SeedRng, len: usize| -> Vec<f64> {
            (0..len).map(|_| rng.next_f64() * 2.0 - 1.0).collect()
        };

        let conv3x3_kernel = init_weight(rng, conv_features * channels * 9);
        let conv3x3_bias = init_weight(rng, conv_features);
        let fc1_weight = init_weight(rng, hidden_dim * conv_features);
        let fc1_bias = init_weight(rng, hidden_dim);
        let fc2_weight = init_weight(rng, channels * hidden_dim);
        let fc2_bias = init_weight(rng, channels);

        NcaWeights {
            conv3x3_kernel,
            conv3x3_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        }
    }
}

/// 数值稳定的 softmax + CDF 采样
fn categorical_sample(logits: &[f64], rng: &mut SeedRng) -> usize {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let mut cdf_acc = 0.0f64;
    let u = rng.next_f64();
    for (i, &e) in exps.iter().enumerate() {
        cdf_acc += e / sum;
        if u < cdf_acc {
            return i;
        }
    }
    logits.len() - 1
}

/// 2D 神经元胞自动机生成器
///
/// 使用纯 Rust f64 实现小型神经网络作为 CA 转移函数。
/// 每个种子确定性产生唯一权重，网络极小（~584 参数），
/// 无需 ML 框架依赖。
pub struct NeuralCA2D {
    d_state: u8,
    n_groups: u8,
    rows: usize,
    cols: usize,
    identity_bias: f64,
    temperature: f64,
    hidden_dim: usize,
    conv_features: usize,
}

impl NeuralCA2D {
    /// One-hot 编码：grid[u8] → oh[f64]
    ///
    /// 输入 shape: [rows, cols, n_groups] (Vec<u8>)
    /// 输出 shape: [rows, cols, channels] (Vec<f64>)，channels = d_state * n_groups
    /// 索引: (r*cols + c) * channels + g*d_state + s
    fn one_hot_encode(
        grid: &[u8],
        rows: usize,
        cols: usize,
        n_groups: u8,
        d_state: u8,
        oh: &mut Vec<f64>,
    ) {
        let channels = (d_state as usize) * (n_groups as usize);
        let spatial = rows * cols;
        oh.clear();
        oh.resize(spatial * channels, 0.0);

        for r in 0..rows {
            for c in 0..cols {
                for g in 0..(n_groups as usize) {
                    let state = grid[(r * cols + c) * (n_groups as usize) + g] as usize;
                    let ch = g * (d_state as usize) + state;
                    oh[(r * cols + c) * channels + ch] = 1.0;
                }
            }
        }
    }

    /// 3×3 卷积（环面拓扑 wrap padding）
    fn conv3x3_forward(
        oh: &[f64],
        conv_out: &mut Vec<f64>,
        weights: &NcaWeights,
        rows: usize,
        cols: usize,
        channels: usize,
        conv_features: usize,
    ) {
        let spatial = rows * cols;
        conv_out.clear();
        conv_out.resize(spatial * conv_features, 0.0);

        for r in 0..rows {
            for c in 0..cols {
                for k in 0..conv_features {
                    let mut val = weights.conv3x3_bias[k];
                    for ch in 0..channels {
                        for di in 0..3usize {
                            for dj in 0..3usize {
                                // wrap padding
                                let sr = (r + di + rows - 1) % rows;
                                let sc = (c + dj + cols - 1) % cols;
                                let input_val = oh[(sr * cols + sc) * channels + ch];
                                let kernel_val =
                                    weights.conv3x3_kernel
                                        [k * channels * 9 + ch * 9 + di * 3 + dj];
                                val += kernel_val * input_val;
                            }
                        }
                    }
                    conv_out[(r * cols + c) * conv_features + k] = val;
                }
            }
        }
    }

    /// FC1 + ReLU：conv_features → hidden_dim
    fn fc1_forward(
        conv_out: &[f64],
        hidden: &mut Vec<f64>,
        weights: &NcaWeights,
        rows: usize,
        cols: usize,
        conv_features: usize,
        hidden_dim: usize,
    ) {
        let spatial = rows * cols;
        hidden.clear();
        hidden.resize(spatial * hidden_dim, 0.0);

        for r in 0..rows {
            for c in 0..cols {
                let spatial_idx = r * cols + c;
                for h in 0..hidden_dim {
                    let mut val = weights.fc1_bias[h];
                    for j in 0..conv_features {
                        val += weights.fc1_weight[h * conv_features + j]
                            * conv_out[spatial_idx * conv_features + j];
                    }
                    hidden[spatial_idx * hidden_dim + h] = val.max(0.0); // ReLU
                }
            }
        }
    }

    /// FC2：hidden_dim → channels (logits)
    fn fc2_forward(
        hidden: &[f64],
        logits: &mut Vec<f64>,
        weights: &NcaWeights,
        rows: usize,
        cols: usize,
        hidden_dim: usize,
        channels: usize,
    ) {
        let spatial = rows * cols;
        logits.clear();
        logits.resize(spatial * channels, 0.0);

        for r in 0..rows {
            for c in 0..cols {
                let spatial_idx = r * cols + c;
                for s in 0..channels {
                    let mut val = weights.fc2_bias[s];
                    for j in 0..hidden_dim {
                        val += weights.fc2_weight[s * hidden_dim + j]
                            * hidden[spatial_idx * hidden_dim + j];
                    }
                    logits[spatial_idx * channels + s] = val;
                }
            }
        }
    }

    /// 应用 identity bias：当前状态的 one-hot 通道加上 bias
    fn apply_identity_bias(
        logits: &mut Vec<f64>,
        grid: &[u8],
        identity_bias: f64,
        rows: usize,
        cols: usize,
        n_groups: u8,
        d_state: u8,
        channels: usize,
    ) {
        for r in 0..rows {
            for c in 0..cols {
                let spatial_idx = r * cols + c;
                for g in 0..(n_groups as usize) {
                    let current_state = grid[spatial_idx * (n_groups as usize) + g] as usize;
                    let ch = g * (d_state as usize) + current_state;
                    logits[spatial_idx * channels + ch] += identity_bias;
                }
            }
        }
    }

    /// 采样：根据 logits 决定下一状态
    fn sample_next_state(
        logits: &[f64],
        next_grid: &mut [u8],
        rng: &mut SeedRng,
        temperature: f64,
        rows: usize,
        cols: usize,
        n_groups: u8,
        d_state: u8,
        channels: usize,
    ) {
        let d_state_usize = d_state as usize;

        for r in 0..rows {
            for c in 0..cols {
                let spatial_idx = r * cols + c;
                for g in 0..(n_groups as usize) {
                    let group_start = g * d_state_usize;
                    let group_end = group_start + d_state_usize;
                    let group_logits = &logits[spatial_idx * channels + group_start
                        ..spatial_idx * channels + group_end];

                    let next_state = if temperature == 0.0 {
                        // argmax
                        let mut best_idx = 0;
                        let mut best_val = group_logits[0];
                        for (i, &val) in group_logits.iter().enumerate().skip(1) {
                            if val > best_val {
                                best_val = val;
                                best_idx = i;
                            }
                        }
                        best_idx as u8
                    } else {
                        // temperature > 0: 缩放后分类采样
                        let scaled: Vec<f64> =
                            group_logits.iter().map(|&l| l / temperature).collect();
                        categorical_sample(&scaled, rng) as u8
                    };

                    next_grid[spatial_idx * (n_groups as usize) + g] = next_state;
                }
            }
        }
    }
}

impl Generator for NeuralCA2D {
    fn name(&self) -> &'static str {
        "neural_cellular_automaton_2d"
    }

    fn generate_stream(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let d_state = self.d_state;
        let n_groups = self.n_groups;
        let rows = self.rows;
        let cols = self.cols;
        let identity_bias = self.identity_bias;
        let temperature = self.temperature;
        let hidden_dim = self.hidden_dim;
        let conv_features = self.conv_features;
        let seq_limit = params.seq_length;

        let channels = (d_state as usize) * (n_groups as usize);
        let spatial = rows * cols;

        // 初始化 PRNG 和权重
        let mut rng = SeedRng::new(seed);
        let weights = NcaWeights::init_from_rng(&mut rng, conv_features, hidden_dim, channels);

        // 随机初始化网格（随机 logits 分类采样）
        let grid_size = spatial * (n_groups as usize);
        let mut buf_a = vec![0u8; grid_size];
        for idx in 0..grid_size {
            let mut init_logits = vec![0.0f64; d_state as usize];
            for l in init_logits.iter_mut() {
                *l = rng.next_f64() * 2.0 - 1.0;
            }
            buf_a[idx] = categorical_sample(&init_logits, &mut rng) as u8;
        }

        let mut buf_b = vec![0u8; grid_size];
        let mut step_counter: u64 = 0;

        // 预分配中间缓冲区
        let mut oh_buf = Vec::with_capacity(spatial * channels);
        let mut conv_out_buf = Vec::with_capacity(spatial * conv_features);
        let mut hidden_buf = Vec::with_capacity(spatial * hidden_dim);
        let mut logits_buf = Vec::with_capacity(spatial * channels);

        let iter = std::iter::from_fn(move || {
            let step = step_counter;
            step_counter += 1;

            // 构建当前帧
            let values: Vec<FrameState> = buf_a
                .iter()
                .map(|&v| FrameState::Integer(v as i64))
                .collect();
            let frame = SequenceFrame::new(step, FrameData { values });

            // 前向传播
            NeuralCA2D::one_hot_encode(&buf_a, rows, cols, n_groups, d_state, &mut oh_buf);
            NeuralCA2D::conv3x3_forward(
                &oh_buf,
                &mut conv_out_buf,
                &weights,
                rows,
                cols,
                channels,
                conv_features,
            );
            NeuralCA2D::fc1_forward(
                &conv_out_buf,
                &mut hidden_buf,
                &weights,
                rows,
                cols,
                conv_features,
                hidden_dim,
            );
            NeuralCA2D::fc2_forward(
                &hidden_buf,
                &mut logits_buf,
                &weights,
                rows,
                cols,
                hidden_dim,
                channels,
            );
            NeuralCA2D::apply_identity_bias(
                &mut logits_buf,
                &buf_a,
                identity_bias,
                rows,
                cols,
                n_groups,
                d_state,
                channels,
            );
            NeuralCA2D::sample_next_state(
                &logits_buf,
                &mut buf_b,
                &mut rng,
                temperature,
                rows,
                cols,
                n_groups,
                d_state,
                channels,
            );

            std::mem::swap(&mut buf_a, &mut buf_b);

            Some(frame)
        });

        let iter: Box<dyn Iterator<Item = SequenceFrame> + Send> = if seq_limit == 0 {
            Box::new(iter)
        } else {
            Box::new(iter.take(seq_limit))
        };

        Ok(iter)
    }
}

/// 2D 神经元胞自动机工厂函数
pub fn nca2d_factory(extensions: &HashMap<String, Value>) -> CoreResult<Box<dyn Generator>> {
    let params: Nca2dParams = deserialize_extensions(extensions)?;

    if params.rows == 0 {
        return Err(CoreError::InvalidParams(
            "NCA2D rows must be greater than 0".into(),
        ));
    }
    if params.cols == 0 {
        return Err(CoreError::InvalidParams(
            "NCA2D cols must be greater than 0".into(),
        ));
    }
    if params.d_state < 2 {
        return Err(CoreError::InvalidParams(
            "NCA2D d_state must be at least 2".into(),
        ));
    }
    if params.n_groups == 0 {
        return Err(CoreError::InvalidParams(
            "NCA2D n_groups must be at least 1".into(),
        ));
    }
    if params.temperature < 0.0 {
        return Err(CoreError::InvalidParams(
            "NCA2D temperature must be >= 0".into(),
        ));
    }
    if params.hidden_dim == 0 {
        return Err(CoreError::InvalidParams(
            "NCA2D hidden_dim must be greater than 0".into(),
        ));
    }
    if params.conv_features == 0 {
        return Err(CoreError::InvalidParams(
            "NCA2D conv_features must be greater than 0".into(),
        ));
    }

    Ok(Box::new(NeuralCA2D {
        d_state: params.d_state,
        n_groups: params.n_groups,
        rows: params.rows,
        cols: params.cols,
        identity_bias: params.identity_bias,
        temperature: params.temperature,
        hidden_dim: params.hidden_dim,
        conv_features: params.conv_features,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn collect_frames(gen: &dyn Generator, seed: u64, num_frames: usize) -> Vec<SequenceFrame> {
        let params = GenParams::simple(num_frames);
        gen.generate_stream(seed, &params).unwrap().collect()
    }

    #[test]
    fn test_deterministic_same_seed() {
        let gen1 = nca2d_factory(&HashMap::new()).unwrap();
        let gen2 = nca2d_factory(&HashMap::new()).unwrap();

        let frames1 = collect_frames(gen1.as_ref(), 42, 5);
        let frames2 = collect_frames(gen2.as_ref(), 42, 5);

        assert_eq!(frames1.len(), 5);
        for i in 0..5 {
            assert_eq!(
                frames1[i].state.values, frames2[i].state.values,
                "Frame {} should be identical for same seed",
                i
            );
        }
    }

    #[test]
    fn test_different_seed_different_output() {
        let gen = nca2d_factory(&HashMap::new()).unwrap();

        let frames1 = collect_frames(gen.as_ref(), 100, 3);
        let frames2 = collect_frames(gen.as_ref(), 200, 3);

        let all_same = frames1
            .iter()
            .zip(frames2.iter())
            .all(|(a, b)| a.state.values == b.state.values);
        assert!(!all_same, "Different seeds should produce different output");
    }

    #[test]
    fn test_temperature_zero_deterministic() {
        // temperature=0 时，argmax 完全确定性（同种子）
        let gen1 = nca2d_factory(&HashMap::new()).unwrap();
        let gen2 = nca2d_factory(&HashMap::new()).unwrap();

        let frames1 = collect_frames(gen1.as_ref(), 42, 10);
        let frames2 = collect_frames(gen2.as_ref(), 42, 10);

        for i in 0..10 {
            assert_eq!(
                frames1[i].state.values, frames2[i].state.values,
                "Frame {} should be identical with temperature=0",
                i
            );
        }
    }

    #[test]
    fn test_identity_bias_persistence() {
        // 高 identity_bias 应使状态趋向不变
        let mut extensions = HashMap::new();
        extensions.insert("identity_bias".to_string(), json!(1000.0));
        extensions.insert("d_state".to_string(), json!(4));
        extensions.insert("rows".to_string(), json!(6));
        extensions.insert("cols".to_string(), json!(6));

        let gen = nca2d_factory(&extensions).unwrap();
        let params = GenParams::simple(3);
        let frames = gen.generate_stream(42, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 3);

        // 高 identity_bias 下，连续帧应非常相似
        let same_count = frames[0]
            .state
            .values
            .iter()
            .zip(frames[1].state.values.iter())
            .filter(|(a, b)| a == b)
            .count();
        let total = frames[0].state.values.len();
        // 至少 90% 的格子应保持不变
        assert!(
            same_count as f64 / total as f64 > 0.9,
            "With high identity_bias, at least 90% cells should stay the same, got {}/{}",
            same_count,
            total
        );
    }

    #[test]
    fn test_output_dimensions() {
        let mut extensions = HashMap::new();
        extensions.insert("rows".to_string(), json!(5));
        extensions.insert("cols".to_string(), json!(6));
        extensions.insert("n_groups".to_string(), json!(2));

        let gen = nca2d_factory(&extensions).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 2);
        for f in &frames {
            assert_eq!(f.state.values.len(), 5 * 6 * 2);
        }
    }

    #[test]
    fn test_output_values_in_range() {
        let mut extensions = HashMap::new();
        extensions.insert("d_state".to_string(), json!(5));
        extensions.insert("rows".to_string(), json!(4));
        extensions.insert("cols".to_string(), json!(4));

        let gen = nca2d_factory(&extensions).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 5);
        for f in &frames {
            for v in &f.state.values {
                let val = v.as_integer().unwrap();
                assert!(
                    val >= 0 && val < 5,
                    "Value should be in [0, d_state-1], got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_zero_rows_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("rows".to_string(), json!(0));
        let result = nca2d_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_temperature_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("temperature".to_string(), json!(-1.0));
        let result = nca2d_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_d_state_less_than_2_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("d_state".to_string(), json!(1));
        let result = nca2d_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_hidden_dim_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("hidden_dim".to_string(), json!(0));
        let result = nca2d_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_conv_features_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("conv_features".to_string(), json!(0));
        let result = nca2d_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_step_indices_sequential() {
        let gen = nca2d_factory(&HashMap::new()).unwrap();
        let params = GenParams::simple(10);
        let frames = gen.generate_stream(99, &params).unwrap().collect::<Vec<_>>();
        for (i, f) in frames.iter().enumerate() {
            assert_eq!(f.step_index, i as u64);
        }
    }

    #[test]
    fn test_unbounded_stream() {
        let gen = nca2d_factory(&HashMap::new()).unwrap();
        let params = GenParams::simple(0);
        let frames = gen
            .generate_stream(42, &params)
            .unwrap()
            .take(20)
            .collect::<Vec<_>>();
        assert_eq!(frames.len(), 20);
    }

    #[test]
    fn test_default_params() {
        let gen = nca2d_factory(&HashMap::new()).unwrap();
        assert_eq!(gen.name(), "neural_cellular_automaton_2d");
        let params = GenParams::simple(3);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 3);
        // 默认 12×12×1
        for f in &frames {
            assert_eq!(f.state.values.len(), 12 * 12);
        }
    }

    #[test]
    fn test_categorical_sample_with_equal_logits() {
        // 等概率 logits 应产生大致均匀的分布
        let mut rng = SeedRng::new(42);
        let logits = [0.0f64; 4];
        let mut counts = [0usize; 4];
        let n_samples = 1000;
        for _ in 0..n_samples {
            let idx = categorical_sample(&logits, &mut rng);
            counts[idx] += 1;
        }
        // 每个类别应约 25%，允许 ±10%
        for &count in &counts {
            let ratio = count as f64 / n_samples as f64;
            assert!(
                ratio > 0.15 && ratio < 0.35,
                "Expected roughly uniform distribution, got ratio {}",
                ratio
            );
        }
    }

    #[test]
    fn test_one_hot_encoding() {
        // 2×2 网格, d_state=3, n_groups=1
        let grid: Vec<u8> = vec![0, 1, 2, 0];
        let mut oh = Vec::new();
        NeuralCA2D::one_hot_encode(&grid, 2, 2, 1, 3, &mut oh);
        let channels = 3;
        // (0,0) state=0 → channel 0
        assert_eq!(oh[0 * channels + 0], 1.0);
        assert_eq!(oh[0 * channels + 1], 0.0);
        assert_eq!(oh[0 * channels + 2], 0.0);
        // (0,1) state=1 → channel 1
        assert_eq!(oh[1 * channels + 0], 0.0);
        assert_eq!(oh[1 * channels + 1], 1.0);
        assert_eq!(oh[1 * channels + 2], 0.0);
        // (1,0) state=2 → channel 2
        assert_eq!(oh[2 * channels + 0], 0.0);
        assert_eq!(oh[2 * channels + 1], 0.0);
        assert_eq!(oh[2 * channels + 2], 1.0);
        // (1,1) state=0 → channel 0
        assert_eq!(oh[3 * channels + 0], 1.0);
        assert_eq!(oh[3 * channels + 1], 0.0);
        assert_eq!(oh[3 * channels + 2], 0.0);
    }

    #[test]
    fn test_positive_temperature_stochastic() {
        // temperature>0 时，即使同种子同工厂，不同运行应有可能产生不同结果
        // （但由于完全确定性，同种子实际上会产生相同结果，
        //   这里验证的是 temperature>0 不会崩溃即可）
        let mut extensions = HashMap::new();
        extensions.insert("temperature".to_string(), json!(1.0));
        extensions.insert("d_state".to_string(), json!(4));
        extensions.insert("rows".to_string(), json!(4));
        extensions.insert("cols".to_string(), json!(4));

        let gen = nca2d_factory(&extensions).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 5);
        assert_eq!(frames.len(), 5);
        for f in &frames {
            for v in &f.state.values {
                let val = v.as_integer().unwrap();
                assert!(val >= 0 && val < 4);
            }
        }
    }
}
