use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use super::rng::SeedRng;
use crate::core::*;

/// 随机布尔网络专用参数
#[derive(Debug, Clone, Deserialize)]
struct BooleanNetworkParams {
    /// 网络节点数
    #[serde(default = "default_num_nodes")]
    num_nodes: usize,
}

fn default_num_nodes() -> usize {
    32
}

impl Default for BooleanNetworkParams {
    fn default() -> Self {
        BooleanNetworkParams {
            num_nodes: default_num_nodes(),
        }
    }
}

/// 随机布尔网络生成器
///
/// N 个节点的布尔网络，每个节点的更新函数为随机布尔函数。
/// 每步更新所有节点后输出全网络状态为一帧。
pub struct BooleanNetwork {
    num_nodes: usize,
}

/// 布尔函数：使用真值表表示，对于 3 输入产生 2^3=8 种输出
type BoolFunc = [bool; 8];

/// 生成一个随机的 3 输入布尔函数（真值表 8 项）
fn random_bool_func(rng: &mut SeedRng) -> BoolFunc {
    let mut table = [false; 8];
    for entry in table.iter_mut() {
        *entry = rng.next_bool();
    }
    table
}

impl Generator for BooleanNetwork {
    fn name(&self) -> &'static str {
        "boolean_network"
    }

    fn generate_stream(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let num_nodes = self.num_nodes;
        let seq_limit = params.seq_length;

        let mut rng = SeedRng::new(seed);

        // 为每个节点生成随机布尔函数（3 输入）
        let mut functions: Vec<BoolFunc> = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            functions.push(random_bool_func(&mut rng));
        }

        // 为每个节点选择 3 个输入来源
        let mut inputs: Vec<[usize; 3]> = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let a = rng.next_usize(num_nodes);
            let b = rng.next_usize(num_nodes);
            let c = rng.next_usize(num_nodes);
            inputs.push([a, b, c]);
        }

        // 生成初始状态
        let mut state = vec![false; num_nodes];
        for cell in state.iter_mut() {
            *cell = rng.next_bool();
        }
        let mut next_state = vec![false; num_nodes];
        let mut step_counter: u64 = 0;

        let iter = std::iter::from_fn(move || {
            let step = step_counter;
            step_counter += 1;

            // 输出当前状态
            let values: Vec<FrameState> =
                state.iter().map(|&b| FrameState::Bool(b)).collect();
            let frame = SequenceFrame::new(step, FrameData { values });

            // 同步更新所有节点
            for i in 0..num_nodes {
                let [a, b, c] = inputs[i];
                let idx = (state[a] as usize) * 4
                    + (state[b] as usize) * 2
                    + (state[c] as usize);
                next_state[i] = functions[i][idx];
            }
            std::mem::swap(&mut state, &mut next_state);

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

/// 布尔网络工厂函数
pub fn boolean_network_factory(
    extensions: &HashMap<String, Value>,
) -> CoreResult<Box<dyn Generator>> {
    let bn_params: BooleanNetworkParams = deserialize_extensions(extensions)?;

    if bn_params.num_nodes == 0 {
        return Err(CoreError::InvalidParams(
            "BooleanNetwork num_nodes must be greater than 0".into(),
        ));
    }

    Ok(Box::new(BooleanNetwork {
        num_nodes: bn_params.num_nodes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_frames(gen: &dyn Generator, seed: u64, num_frames: usize) -> Vec<SequenceFrame> {
        let params = GenParams::simple(num_frames);
        gen.generate_stream(seed, &params).unwrap().collect()
    }

    #[test]
    fn test_deterministic_same_seed_same_output() {
        let gen1 = boolean_network_factory(&HashMap::new()).unwrap();
        let gen2 = boolean_network_factory(&HashMap::new()).unwrap();

        let frames1 = collect_frames(gen1.as_ref(), 42, 10);
        let frames2 = collect_frames(gen2.as_ref(), 42, 10);

        assert_eq!(frames1.len(), 10);
        for i in 0..10 {
            assert_eq!(
                frames1[i].state.values, frames2[i].state.values,
                "Frame {} should be identical",
                i
            );
        }
    }

    #[test]
    fn test_different_seed_different_output() {
        let gen = boolean_network_factory(&HashMap::new()).unwrap();

        let frames1 = collect_frames(gen.as_ref(), 100, 5);
        let frames2 = collect_frames(gen.as_ref(), 200, 5);

        let all_same = frames1
            .iter()
            .zip(frames2.iter())
            .all(|(a, b)| a.state.values == b.state.values);
        assert!(!all_same);
    }

    #[test]
    fn test_output_dimensions() {
        let mut extensions = HashMap::new();
        extensions.insert("num_nodes".to_string(), serde_json::json!(10));

        let gen = boolean_network_factory(&extensions).unwrap();
        let params = GenParams::simple(5);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();

        for f in &frames {
            assert_eq!(f.state.values.len(), 10);
            // 验证所有值都是 Bool 类型
            for v in &f.state.values {
                assert!(matches!(v, FrameState::Bool(_)));
            }
        }
    }

    #[test]
    fn test_zero_nodes_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("num_nodes".to_string(), serde_json::json!(0));
        let result = boolean_network_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_default_params() {
        let gen = boolean_network_factory(&HashMap::new()).unwrap();
        assert_eq!(gen.name(), "boolean_network");

        let params = GenParams::simple(3);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 3);
        // 默认 32 个节点
        for f in &frames {
            assert_eq!(f.state.values.len(), 32);
        }
    }

    #[test]
    fn test_unbounded_stream() {
        let gen = boolean_network_factory(&HashMap::new()).unwrap();
        let params = GenParams::simple(0);
        let frames = gen
            .generate_stream(42, &params)
            .unwrap()
            .take(50)
            .collect::<Vec<_>>();
        assert_eq!(frames.len(), 50);
    }
}
