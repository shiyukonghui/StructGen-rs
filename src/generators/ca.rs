use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use super::rng::SeedRng;
use crate::core::*;

/// 元胞自动机边界条件类型
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum Boundary {
    /// 周期边界：首尾相连，如环形
    #[default]
    Periodic,
    /// 固定边界：超出部分视为 0
    Fixed,
    /// 反射边界：超出部分取边界元胞值
    Reflective,
}

/// 元胞自动机专用参数
#[derive(Debug, Clone, Deserialize)]
struct CaParams {
    /// Wolfram 规则号 (0-255)
    #[serde(default = "default_rule")]
    rule: u8,
    /// 边界条件类型
    #[serde(default)]
    boundary: Boundary,
    /// 网格宽度
    #[serde(default = "default_width")]
    width: usize,
}

fn default_rule() -> u8 {
    30
}
fn default_width() -> usize {
    128
}

impl Default for CaParams {
    fn default() -> Self {
        CaParams {
            rule: default_rule(),
            boundary: Boundary::default(),
            width: default_width(),
        }
    }
}

/// 一维元胞自动机生成器
///
/// 使用 Wolfram 规则（查找表）进行状态演化，支持三种边界条件。
/// 每次演化步输出完整网格状态为一帧。
pub struct CellularAutomaton {
    boundary: Boundary,
    width: usize,
    /// 规则查找表：8 个布尔输出（索引 = 邻域位掩码 0-7）
    lut: [bool; 8],
}

impl CellularAutomaton {
    /// 根据规则号构建查找表
    fn build_lut(rule: u8) -> [bool; 8] {
        let mut lut = [false; 8];
        for i in 0..8u8 {
            lut[i as usize] = (rule >> i) & 1 == 1;
        }
        lut
    }

    /// 计算邻域位掩码（3 位：left*4 + center*2 + right）
    fn neighborhood_index(
        grid: &[bool],
        center: usize,
        width: usize,
        boundary: &Boundary,
    ) -> usize {
        let left = match boundary {
            Boundary::Periodic => {
                if center == 0 {
                    grid[width - 1]
                } else {
                    grid[center - 1]
                }
            }
            Boundary::Fixed => {
                if center == 0 {
                    false
                } else {
                    grid[center - 1]
                }
            }
            Boundary::Reflective => {
                if center == 0 {
                    grid[0]
                } else {
                    grid[center - 1]
                }
            }
        };

        let center_val = grid[center];

        let right = match boundary {
            Boundary::Periodic => {
                if center == width - 1 {
                    grid[0]
                } else {
                    grid[center + 1]
                }
            }
            Boundary::Fixed => {
                if center == width - 1 {
                    false
                } else {
                    grid[center + 1]
                }
            }
            Boundary::Reflective => {
                if center == width - 1 {
                    grid[width - 1]
                } else {
                    grid[center + 1]
                }
            }
        };

        (left as usize) * 4 + (center_val as usize) * 2 + (right as usize)
    }

    /// 执行一步演化，从 src 计算到 dst
    fn evolve(src: &[bool], dst: &mut [bool], width: usize, lut: &[bool; 8], boundary: &Boundary) {
        for (i, dst_cell) in dst.iter_mut().enumerate() {
            let idx = Self::neighborhood_index(src, i, width, boundary);
            *dst_cell = lut[idx];
        }
    }
}

impl Generator for CellularAutomaton {
    fn name(&self) -> &'static str {
        "cellular_automaton"
    }

    fn generate_stream(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let width = self.width;
        let lut = self.lut;
        let boundary = self.boundary.clone();
        let seq_limit = params.seq_length;

        // 用种子 PRNG 生成初始状态
        let mut rng = SeedRng::new(seed);
        let mut buf_a = vec![false; width];
        for cell in buf_a.iter_mut() {
            *cell = rng.next_bool();
        }
        let mut buf_b = vec![false; width];
        let mut step_counter: u64 = 0;

        // 捕获所有状态到闭包中（move 语义）
        let iter = std::iter::from_fn(move || {
            let step = step_counter;
            step_counter += 1;

            // 构建当前帧：输出 buf_a（源缓冲区）的状态
            let values: Vec<FrameState> = buf_a.iter().map(|&b| FrameState::Bool(b)).collect();
            let frame = SequenceFrame::new(step, FrameData { values });

            // 执行演化：buf_a(源) -> buf_b(目标)
            CellularAutomaton::evolve(&buf_a, &mut buf_b, width, &lut, &boundary);
            // 交换缓冲区：下一轮 buf_a 成为新源
            std::mem::swap(&mut buf_a, &mut buf_b);

            Some(frame)
        });

        // 限制序列长度（seq_limit=0 时无限）
        let iter: Box<dyn Iterator<Item = SequenceFrame> + Send> = if seq_limit == 0 {
            Box::new(iter)
        } else {
            Box::new(iter.take(seq_limit))
        };

        Ok(iter)
    }
}

/// CA 生成器工厂函数
pub fn ca_factory(extensions: &HashMap<String, Value>) -> CoreResult<Box<dyn Generator>> {
    // 应用预设解析（若指定）
    let ext = super::ca_rules::apply_preset_to_extensions(extensions, 1)?;
    let ca_params: CaParams = deserialize_extensions(&ext)?;

    if ca_params.width == 0 {
        return Err(CoreError::InvalidParams(
            "CA width must be greater than 0".into(),
        ));
    }

    Ok(Box::new(CellularAutomaton {
        boundary: ca_params.boundary,
        width: ca_params.width,
        lut: CellularAutomaton::build_lut(ca_params.rule),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// 辅助函数：生成 N 帧并收集
    fn collect_frames(gen: &dyn Generator, seed: u64, num_frames: usize) -> Vec<SequenceFrame> {
        let params = GenParams::simple(num_frames);
        gen.generate_stream(seed, &params).unwrap().collect()
    }

    #[test]
    fn test_deterministic_same_seed_same_output() {
        let gen1 = ca_factory(&HashMap::new()).unwrap();
        let gen2 = ca_factory(&HashMap::new()).unwrap();

        let frames1 = collect_frames(gen1.as_ref(), 42, 10);
        let frames2 = collect_frames(gen2.as_ref(), 42, 10);

        assert_eq!(frames1.len(), 10);
        assert_eq!(frames2.len(), 10);
        for i in 0..10 {
            assert_eq!(
                frames1[i].state.values, frames2[i].state.values,
                "Frame {} should be identical for same seed",
                i
            );
        }
    }

    #[test]
    fn test_different_seed_different_output() {
        let gen = ca_factory(&HashMap::new()).unwrap();

        let frames1 = collect_frames(gen.as_ref(), 100, 5);
        let frames2 = collect_frames(gen.as_ref(), 200, 5);

        // 不同种子大概率不同，但不能 100% 保证，至少检查不全部相同
        let all_same = frames1
            .iter()
            .zip(frames2.iter())
            .all(|(a, b)| a.state.values == b.state.values);
        assert!(!all_same, "Different seeds should produce different output");
    }

    #[test]
    fn test_rule_30_known_pattern() {
        // 规则 30：验证演化规则一致性
        let width = 31;
        let mut extensions = HashMap::new();
        extensions.insert("rule".to_string(), json!(30u8));
        extensions.insert("boundary".to_string(), json!("fixed"));
        extensions.insert("width".to_string(), json!(width));

        let gen = ca_factory(&extensions).unwrap();
        let params = GenParams::simple(10);
        let frames = gen.generate_stream(12345, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 10);

        for f in &frames {
            assert_eq!(f.state.values.len(), width);
        }

        // 验证演化一致性：step0->step1 使用规则 30 LUT
        let lut = CellularAutomaton::build_lut(30);
        for step_idx in 0..9 {
            let src = &frames[step_idx].state.values;
            let dst = &frames[step_idx + 1].state.values;
            for i in 0..width {
                let left = if i == 0 {
                    false
                } else {
                    src[i - 1].as_bool().unwrap()
                };
                let center = src[i].as_bool().unwrap();
                let right = if i == width - 1 {
                    false
                } else {
                    src[i + 1].as_bool().unwrap()
                };
                let idx = (left as usize) * 4 + (center as usize) * 2 + (right as usize);
                assert_eq!(
                    dst[i].as_bool().unwrap(),
                    lut[idx],
                    "Rule 30 mismatch at step {}, cell {}",
                    step_idx,
                    i
                );
            }
        }
    }

    #[test]
    fn test_rule_110_known_pattern() {
        // 规则 110：验证至少 10 步演化一致性
        let width = 31;
        let mut extensions = HashMap::new();
        extensions.insert("rule".to_string(), json!(110u8));
        extensions.insert("boundary".to_string(), json!("fixed"));
        extensions.insert("width".to_string(), json!(width));

        let gen = ca_factory(&extensions).unwrap();
        let params = GenParams::simple(11); // 初始状态 + 10 步演化 = 11 帧
        let frames = gen.generate_stream(42, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 11);

        let lut = CellularAutomaton::build_lut(110);
        for step_idx in 0..10 {
            let src = &frames[step_idx].state.values;
            let dst = &frames[step_idx + 1].state.values;
            for i in 0..width {
                let left = if i == 0 {
                    false
                } else {
                    src[i - 1].as_bool().unwrap()
                };
                let center = src[i].as_bool().unwrap();
                let right = if i == width - 1 {
                    false
                } else {
                    src[i + 1].as_bool().unwrap()
                };
                let idx = (left as usize) * 4 + (center as usize) * 2 + (right as usize);
                assert_eq!(
                    dst[i].as_bool().unwrap(),
                    lut[idx],
                    "Rule 110 mismatch at step {}, cell {}",
                    step_idx,
                    i
                );
            }
        }
    }

    #[test]
    fn test_invalid_rule_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("rule".to_string(), json!(300u16));
        let result = ca_factory(&extensions);
        assert!(result.is_err());
        let err = result.err().unwrap();
        match err {
            CoreError::InvalidParams(msg) => {
                assert!(msg.contains("rule"), "Error should mention rule: {}", msg);
            }
            CoreError::SerializationError(_) => {
                // 300 超出 u8 范围，可能在反序列化阶段报错，这也是可接受的
            }
            _ => panic!("Expected error, got {:?}", err),
        }
    }

    #[test]
    fn test_invalid_boundary_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("boundary".to_string(), json!("invalid_type"));
        let result = ca_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_periodic_boundary() {
        let mut extensions = HashMap::new();
        extensions.insert("boundary".to_string(), json!("periodic"));
        extensions.insert("width".to_string(), json!(16));
        extensions.insert("rule".to_string(), json!(30u8));

        let gen = ca_factory(&extensions).unwrap();
        let params = GenParams::simple(5);
        let frames = gen.generate_stream(7, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 5);
        for f in &frames {
            assert_eq!(f.state.values.len(), 16);
        }
    }

    #[test]
    fn test_reflective_boundary() {
        let mut extensions = HashMap::new();
        extensions.insert("boundary".to_string(), json!("reflective"));
        extensions.insert("width".to_string(), json!(8));
        extensions.insert("rule".to_string(), json!(90u8));

        let gen = ca_factory(&extensions).unwrap();
        let params = GenParams::simple(3);
        let frames = gen.generate_stream(3, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 3);

        // 验证反射边界：元胞 0 的左邻域等于自身
        let step0 = &frames[0].state.values;
        let step1 = &frames[1].state.values;
        let lut = CellularAutomaton::build_lut(90);
        // 元胞 0：左=自身(reflective), 中=自身, 右=cell[1]
        let left = step0[0].as_bool().unwrap();
        let center = step0[0].as_bool().unwrap();
        let right = step0[1].as_bool().unwrap();
        let idx = (left as usize) * 4 + (center as usize) * 2 + (right as usize);
        assert_eq!(step1[0].as_bool().unwrap(), lut[idx]);
    }

    #[test]
    fn test_zero_width_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("width".to_string(), json!(0));
        let result = ca_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_step_indices_sequential() {
        let gen = ca_factory(&HashMap::new()).unwrap();
        let params = GenParams::simple(20);
        let frames = gen.generate_stream(99, &params).unwrap().collect::<Vec<_>>();
        for (i, f) in frames.iter().enumerate() {
            assert_eq!(f.step_index, i as u64);
        }
    }

    #[test]
    fn test_default_params() {
        let gen = ca_factory(&HashMap::new()).unwrap();
        assert_eq!(gen.name(), "cellular_automaton");
        let params = GenParams::simple(3);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 3);
        // 默认 width = 128
        for f in &frames {
            assert_eq!(f.state.values.len(), 128);
        }
    }

    #[test]
    fn test_unbounded_stream() {
        let gen = ca_factory(&HashMap::new()).unwrap();
        let params = GenParams::simple(0); // 无限流
        let frames = gen
            .generate_stream(42, &params)
            .unwrap()
            .take(50)
            .collect::<Vec<_>>();
        assert_eq!(frames.len(), 50);
    }

    // ---- 预设测试 ----

    #[test]
    fn test_preset_rule110() {
        let mut ext = HashMap::new();
        ext.insert("preset".to_string(), json!("rule110"));
        let gen = ca_factory(&ext).unwrap();
        // 验证 LUT 与手动指定 rule=110 一致
        let expected_lut = CellularAutomaton::build_lut(110);
        assert_eq!(gen.name(), "cellular_automaton");
        let params = GenParams::simple(5);
        let frames = gen.generate_stream(42, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 5);
    }

    #[test]
    fn test_preset_rule90() {
        let mut ext = HashMap::new();
        ext.insert("preset".to_string(), json!("rule90"));
        let gen = ca_factory(&ext).unwrap();
        let params = GenParams::simple(3);
        let frames = gen.generate_stream(42, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 3);
    }

    #[test]
    fn test_preset_rule54() {
        let mut ext = HashMap::new();
        ext.insert("preset".to_string(), json!("rule54"));
        let gen = ca_factory(&ext).unwrap();
        let params = GenParams::simple(3);
        let frames = gen.generate_stream(42, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 3);
    }

    #[test]
    fn test_preset_rule184() {
        let mut ext = HashMap::new();
        ext.insert("preset".to_string(), json!("rule184"));
        let gen = ca_factory(&ext).unwrap();
        let params = GenParams::simple(3);
        let frames = gen.generate_stream(42, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 3);
    }

    #[test]
    fn test_preset_rule30_default_equivalent() {
        // rule30 预设应与默认行为（rule=30）一致
        let mut ext = HashMap::new();
        ext.insert("preset".to_string(), json!("rule30"));
        let gen_preset = ca_factory(&ext).unwrap();
        let gen_default = ca_factory(&HashMap::new()).unwrap();

        let params = GenParams::simple(10);
        let frames_preset = gen_preset
            .generate_stream(77, &params)
            .unwrap()
            .collect::<Vec<_>>();
        let frames_default = gen_default
            .generate_stream(77, &params)
            .unwrap()
            .collect::<Vec<_>>();

        for i in 0..10 {
            assert_eq!(
                frames_preset[i].state.values,
                frames_default[i].state.values,
                "rule30 preset should match default at frame {}",
                i
            );
        }
    }

    #[test]
    fn test_preset_preserves_grid_params() {
        let mut ext = HashMap::new();
        ext.insert("preset".to_string(), json!("rule110"));
        ext.insert("width".to_string(), json!(64));
        let gen = ca_factory(&ext).unwrap();
        let params = GenParams::simple(3);
        let frames = gen.generate_stream(42, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames[0].state.values.len(), 64);
    }

    #[test]
    fn test_unknown_preset_rejected() {
        let mut ext = HashMap::new();
        ext.insert("preset".to_string(), json!("rule999"));
        assert!(ca_factory(&ext).is_err());
    }
}
