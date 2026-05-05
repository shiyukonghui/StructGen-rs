use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use super::ca_common::*;
use super::rng::SeedRng;
use crate::core::*;

/// 3D 元胞自动机专用参数
#[derive(Debug, Clone, Deserialize)]
struct Ca3dParams {
    #[serde(default = "default_rule_type")]
    rule_type: String,
    #[serde(default = "default_birth_3d")]
    birth: Vec<u8>,
    #[serde(default = "default_survival_3d")]
    survival: Vec<u8>,
    #[serde(default)]
    totalistic_table: Vec<u8>,
    #[serde(default = "default_d_state")]
    d_state: u8,
    #[serde(default = "default_depth")]
    depth: usize,
    #[serde(default = "default_rows_3d")]
    rows: usize,
    #[serde(default = "default_cols_3d")]
    cols: usize,
    #[serde(default)]
    boundary: Boundary,
    #[serde(default)]
    neighborhood: Neighborhood,
    #[serde(default)]
    init_mode: InitMode,
}

fn default_rule_type() -> String {
    "lifelike".to_string()
}
fn default_birth_3d() -> Vec<u8> {
    vec![5, 6, 7]
}
fn default_survival_3d() -> Vec<u8> {
    vec![5, 6]
}
fn default_d_state() -> u8 {
    2
}
fn default_depth() -> usize {
    16
}
fn default_rows_3d() -> usize {
    16
}
fn default_cols_3d() -> usize {
    16
}

impl Default for Ca3dParams {
    fn default() -> Self {
        Ca3dParams {
            rule_type: default_rule_type(),
            birth: default_birth_3d(),
            survival: default_survival_3d(),
            totalistic_table: Vec::new(),
            d_state: default_d_state(),
            depth: default_depth(),
            rows: default_rows_3d(),
            cols: default_cols_3d(),
            boundary: Boundary::default(),
            neighborhood: Neighborhood::default(),
            init_mode: InitMode::default(),
        }
    }
}

/// 3D 元胞自动机生成器
///
/// 支持 Life-like (B/S 记法) 和 Totalistic 两种规则系统，
/// 可配置多值离散状态、边界条件和邻域类型。
pub struct CellularAutomaton3D {
    rule: Rule3D,
    lut: Option<LifeLikeLUT>,
    d_state: u8,
    depth: usize,
    rows: usize,
    cols: usize,
    boundary: Boundary,
    neighborhood: Neighborhood,
    init_mode: InitMode,
}

impl CellularAutomaton3D {
    /// 执行一步 3D 演化
    fn evolve(
        src: &[u8],
        dst: &mut [u8],
        depth: usize,
        rows: usize,
        cols: usize,
        rule: &Rule3D,
        lut: &Option<LifeLikeLUT>,
        boundary: &Boundary,
        neighborhood: &Neighborhood,
        d_state: u8,
    ) {
        let offsets: &[(i32, i32, i32)] = match neighborhood {
            Neighborhood::Moore => &MOORE_3D_OFFSETS,
            Neighborhood::VonNeumann => &VONNEUMANN_3D_OFFSETS,
        };

        for d in 0..depth {
            for r in 0..rows {
                for c in 0..cols {
                    let idx = d * (rows * cols) + r * cols + c;
                    let current = src[idx];

                    match rule {
                        Rule3D::LifeLike3D { .. } => {
                            let lut = lut.as_ref().expect("LifeLike3D rule requires LUT");
                            let mut alive_count: usize = 0;
                            for &(dd, dr, dc) in offsets {
                                let n = get_neighbor_3d(
                                    src, d, r, c, depth, rows, cols, dd, dr, dc, boundary,
                                );
                                if n > 0 {
                                    alive_count += 1;
                                }
                            }
                            if current == 0 {
                                dst[idx] = if alive_count < lut.birth_lut.len()
                                    && lut.birth_lut[alive_count]
                                {
                                    1
                                } else {
                                    0
                                };
                            } else {
                                dst[idx] = if alive_count < lut.survival_lut.len()
                                    && lut.survival_lut[alive_count]
                                {
                                    1
                                } else {
                                    0
                                };
                            }
                        }
                        Rule3D::Totalistic3D { transition_table } => {
                            let mut state_sum: usize = 0;
                            for &(dd, dr, dc) in offsets {
                                let n = get_neighbor_3d(
                                    src, d, r, c, depth, rows, cols, dd, dr, dc, boundary,
                                );
                                state_sum += n as usize;
                            }
                            dst[idx] = if state_sum < transition_table.len() {
                                transition_table[state_sum]
                            } else {
                                0
                            };
                            if dst[idx] >= d_state {
                                dst[idx] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}

impl Generator for CellularAutomaton3D {
    fn name(&self) -> &'static str {
        "cellular_automaton_3d"
    }

    fn generate_stream(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let depth = self.depth;
        let rows = self.rows;
        let cols = self.cols;
        let d_state = self.d_state;
        let boundary = self.boundary.clone();
        let neighborhood = self.neighborhood.clone();
        let init_mode = self.init_mode.clone();
        let rule = self.rule.clone();
        let lut = self.lut.clone();
        let seq_limit = params.seq_length;

        let mut rng = SeedRng::new(seed);
        let grid_size = depth * rows * cols;
        let mut buf_a = vec![0u8; grid_size];

        match init_mode {
            InitMode::Random => {
                for cell in buf_a.iter_mut() {
                    *cell = rng.next_usize(d_state as usize) as u8;
                }
            }
            InitMode::SingleCenter => {
                buf_a[(depth / 2) * (rows * cols) + (rows / 2) * cols + (cols / 2)] = 1;
            }
        }

        let mut buf_b = vec![0u8; grid_size];
        let mut step_counter: u64 = 0;

        let iter = std::iter::from_fn(move || {
            let step = step_counter;
            step_counter += 1;

            let values: Vec<FrameState> = buf_a
                .iter()
                .map(|&v| FrameState::Integer(v as i64))
                .collect();
            let frame = SequenceFrame::new(step, FrameData { values });

            CellularAutomaton3D::evolve(
                &buf_a,
                &mut buf_b,
                depth,
                rows,
                cols,
                &rule,
                &lut,
                &boundary,
                &neighborhood,
                d_state,
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

/// 3D 元胞自动机工厂函数
pub fn ca3d_factory(extensions: &HashMap<String, Value>) -> CoreResult<Box<dyn Generator>> {
    let params: Ca3dParams = deserialize_extensions(extensions)?;

    if params.depth == 0 {
        return Err(CoreError::InvalidParams(
            "CA3D depth must be greater than 0".into(),
        ));
    }
    if params.rows == 0 {
        return Err(CoreError::InvalidParams(
            "CA3D rows must be greater than 0".into(),
        ));
    }
    if params.cols == 0 {
        return Err(CoreError::InvalidParams(
            "CA3D cols must be greater than 0".into(),
        ));
    }
    if params.d_state < 2 {
        return Err(CoreError::InvalidParams(
            "CA3D d_state must be at least 2".into(),
        ));
    }

    let max_neighbors = match params.neighborhood {
        Neighborhood::Moore => 26,
        Neighborhood::VonNeumann => 6,
    };

    let (rule, lut) = match params.rule_type.as_str() {
        "lifelike" => {
            if params.d_state != 2 {
                return Err(CoreError::InvalidParams(
                    "LifeLike3D rules require d_state == 2".into(),
                ));
            }
            for &b in &params.birth {
                if b as usize > max_neighbors {
                    return Err(CoreError::InvalidParams(format!(
                        "birth value {} exceeds max neighbors {}",
                        b, max_neighbors
                    )));
                }
            }
            for &s in &params.survival {
                if s as usize > max_neighbors {
                    return Err(CoreError::InvalidParams(format!(
                        "survival value {} exceeds max neighbors {}",
                        s, max_neighbors
                    )));
                }
            }
            let lut = LifeLikeLUT::from_birth_survival(&params.birth, &params.survival, max_neighbors);
            let rule = Rule3D::LifeLike3D {
                birth: params.birth,
                survival: params.survival,
            };
            (rule, Some(lut))
        }
        "totalistic" => {
            if params.totalistic_table.is_empty() {
                return Err(CoreError::InvalidParams(
                    "Totalistic3D rule requires a non-empty totalistic_table".into(),
                ));
            }
            let max_sum = max_neighbors * (params.d_state as usize - 1);
            if params.totalistic_table.len() <= max_sum {
                return Err(CoreError::InvalidParams(format!(
                    "totalistic_table length {} must be > max neighbor sum {}",
                    params.totalistic_table.len(),
                    max_sum
                )));
            }
            for (i, &val) in params.totalistic_table.iter().enumerate() {
                if val >= params.d_state {
                    return Err(CoreError::InvalidParams(format!(
                        "totalistic_table[{}] = {} exceeds d_state {}",
                        i, val, params.d_state
                    )));
                }
            }
            let rule = Rule3D::Totalistic3D {
                transition_table: params.totalistic_table,
            };
            (rule, None)
        }
        other => {
            return Err(CoreError::InvalidParams(format!(
                "unknown rule_type '{}', expected 'lifelike' or 'totalistic'",
                other
            )));
        }
    };

    Ok(Box::new(CellularAutomaton3D {
        rule,
        lut,
        d_state: params.d_state,
        depth: params.depth,
        rows: params.rows,
        cols: params.cols,
        boundary: params.boundary,
        neighborhood: params.neighborhood,
        init_mode: params.init_mode,
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
        let gen1 = ca3d_factory(&HashMap::new()).unwrap();
        let gen2 = ca3d_factory(&HashMap::new()).unwrap();

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
        let gen = ca3d_factory(&HashMap::new()).unwrap();

        let frames1 = collect_frames(gen.as_ref(), 100, 3);
        let frames2 = collect_frames(gen.as_ref(), 200, 3);

        let all_same = frames1
            .iter()
            .zip(frames2.iter())
            .all(|(a, b)| a.state.values == b.state.values);
        assert!(!all_same, "Different seeds should produce different output");
    }

    #[test]
    fn test_output_dimensions() {
        let mut extensions = HashMap::new();
        extensions.insert("depth".to_string(), json!(4));
        extensions.insert("rows".to_string(), json!(5));
        extensions.insert("cols".to_string(), json!(6));

        let gen = ca3d_factory(&extensions).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 3);
        for f in &frames {
            assert_eq!(f.state.values.len(), 4 * 5 * 6);
        }
    }

    #[test]
    fn test_3d_single_center_neighbor_count() {
        // 3×3×3 网格，中心 (1,1,1) = 1，其余为 0
        // 使用 Moore 邻域，中心格有 0 个活邻居（它自己是活的但不计入邻居）
        // 它的 26 个邻居应该都是 0
        let mut extensions = HashMap::new();
        extensions.insert("rule_type".to_string(), json!("lifelike"));
        extensions.insert("birth".to_string(), json!([3]));
        extensions.insert("survival".to_string(), json!([2, 3]));
        extensions.insert("depth".to_string(), json!(3));
        extensions.insert("rows".to_string(), json!(3));
        extensions.insert("cols".to_string(), json!(3));
        extensions.insert("boundary".to_string(), json!("fixed"));
        extensions.insert("init_mode".to_string(), json!("singlecenter"));

        let gen = ca3d_factory(&extensions).unwrap();
        let params = GenParams::simple(2);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 2);

        // 帧 0: 仅中心 (1,1,1) 为 1
        let f0 = &frames[0];
        let center_idx = 1 * (3 * 3) + 1 * 3 + 1;
        assert_eq!(f0.state.values[center_idx].as_integer().unwrap(), 1);

        // 帧 1: 中心格的 26 邻居中都是 0（fixed 边界），
        // 所以中心格有 0 个活邻居，B3/S23 下不存活
        let f1 = &frames[1];
        assert_eq!(f1.state.values[center_idx].as_integer().unwrap(), 0);

        // 与中心格相邻的 6 个轴邻居各有 1 个活邻居（中心格），
        // 但 B3 要求 3 个活邻居，所以它们也不出生
    }

    #[test]
    fn test_periodic_3d_boundary() {
        let mut extensions = HashMap::new();
        extensions.insert("boundary".to_string(), json!("periodic"));
        extensions.insert("depth".to_string(), json!(4));
        extensions.insert("rows".to_string(), json!(4));
        extensions.insert("cols".to_string(), json!(4));

        let gen = ca3d_factory(&extensions).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 3);
        assert_eq!(frames.len(), 3);
        for f in &frames {
            assert_eq!(f.state.values.len(), 4 * 4 * 4);
        }
    }

    #[test]
    fn test_moore_26_neighborhood() {
        let mut extensions = HashMap::new();
        extensions.insert("neighborhood".to_string(), json!("moore"));
        extensions.insert("depth".to_string(), json!(4));
        extensions.insert("rows".to_string(), json!(4));
        extensions.insert("cols".to_string(), json!(4));

        let gen = ca3d_factory(&extensions).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 3);
        assert_eq!(frames.len(), 3);
    }

    #[test]
    fn test_vonneumann_6_neighborhood() {
        let mut extensions = HashMap::new();
        extensions.insert("neighborhood".to_string(), json!("vonneumann"));
        extensions.insert("birth".to_string(), json!([3, 4]));
        extensions.insert("survival".to_string(), json!([2, 3]));
        extensions.insert("depth".to_string(), json!(4));
        extensions.insert("rows".to_string(), json!(4));
        extensions.insert("cols".to_string(), json!(4));

        let gen = ca3d_factory(&extensions).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 3);
        assert_eq!(frames.len(), 3);
    }

    #[test]
    fn test_zero_depth_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("depth".to_string(), json!(0));
        let result = ca3d_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_rows_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("rows".to_string(), json!(0));
        let result = ca3d_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_lifelike_requires_dstate2() {
        let mut extensions = HashMap::new();
        extensions.insert("rule_type".to_string(), json!("lifelike"));
        extensions.insert("d_state".to_string(), json!(3));
        let result = ca3d_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_step_indices_sequential() {
        let gen = ca3d_factory(&HashMap::new()).unwrap();
        let params = GenParams::simple(10);
        let frames = gen.generate_stream(99, &params).unwrap().collect::<Vec<_>>();
        for (i, f) in frames.iter().enumerate() {
            assert_eq!(f.step_index, i as u64);
        }
    }

    #[test]
    fn test_unbounded_stream() {
        let gen = ca3d_factory(&HashMap::new()).unwrap();
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
        let gen = ca3d_factory(&HashMap::new()).unwrap();
        assert_eq!(gen.name(), "cellular_automaton_3d");
        let params = GenParams::simple(3);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 3);
        // 默认 16³
        for f in &frames {
            assert_eq!(f.state.values.len(), 16 * 16 * 16);
        }
    }

    #[test]
    fn test_totalistic_3d() {
        let mut extensions = HashMap::new();
        extensions.insert("rule_type".to_string(), json!("totalistic"));
        extensions.insert("d_state".to_string(), json!(2));
        // max_sum = 26 * 1 = 26, 需要 len > 26
        let table: Vec<u8> = (0..27).map(|i| if i > 0 && i <= 4 { 1 } else { 0 }).collect();
        extensions.insert("totalistic_table".to_string(), json!(table));
        extensions.insert("depth".to_string(), json!(3));
        extensions.insert("rows".to_string(), json!(3));
        extensions.insert("cols".to_string(), json!(3));

        let gen = ca3d_factory(&extensions).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 3);
        assert_eq!(frames.len(), 3);
        for f in &frames {
            assert_eq!(f.state.values.len(), 27);
            for v in &f.state.values {
                let val = v.as_integer().unwrap();
                assert!(val == 0 || val == 1, "Value should be 0 or 1, got {}", val);
            }
        }
    }
}
