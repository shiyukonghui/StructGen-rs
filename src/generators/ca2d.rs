use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use super::ca_common::*;
use super::ca_rules;
use super::rng::SeedRng;
use crate::core::*;

/// 2D 元胞自动机专用参数
#[derive(Debug, Clone, Deserialize)]
struct Ca2dParams {
    #[serde(default = "default_rule_type")]
    rule_type: String,
    #[serde(default = "default_birth")]
    birth: Vec<u8>,
    #[serde(default = "default_survival")]
    survival: Vec<u8>,
    #[serde(default)]
    totalistic_table: Vec<u8>,
    #[serde(default = "default_d_state")]
    d_state: u8,
    #[serde(default = "default_rows")]
    rows: usize,
    #[serde(default = "default_cols")]
    cols: usize,
    #[serde(default)]
    boundary: Boundary,
    #[serde(default)]
    neighborhood: Neighborhood,
    #[serde(default)]
    init_mode: InitMode,
    /// Hensel 各向同性非总量规则记法（仅 rule_type="hensel" 时使用）
    #[serde(default)]
    hensel_notation: Option<String>,
    /// 128 字符十六进制查找表（仅 rule_type="lookuptable" 时使用）
    #[serde(default)]
    lookup_table_hex: Option<String>,
    /// 512 项 0/1 查找表数组（仅 rule_type="lookuptable" 时使用）
    #[serde(default)]
    lookup_table_array: Option<Vec<u8>>,
    /// 循环 CA 状态数（仅 rule_type="cyclic" 时使用）
    #[serde(default = "default_n_states")]
    n_states: u8,
    /// 循环 CA 阈值（仅 rule_type="cyclic" 时使用）
    #[serde(default = "default_threshold")]
    threshold: u8,
}

fn default_rule_type() -> String {
    "lifelike".to_string()
}
fn default_birth() -> Vec<u8> {
    vec![3]
}
fn default_survival() -> Vec<u8> {
    vec![2, 3]
}
fn default_d_state() -> u8 {
    2
}
fn default_rows() -> usize {
    64
}
fn default_cols() -> usize {
    64
}
fn default_n_states() -> u8 {
    14
}
fn default_threshold() -> u8 {
    1
}

impl Default for Ca2dParams {
    fn default() -> Self {
        Ca2dParams {
            rule_type: default_rule_type(),
            birth: default_birth(),
            survival: default_survival(),
            totalistic_table: Vec::new(),
            d_state: default_d_state(),
            rows: default_rows(),
            cols: default_cols(),
            boundary: Boundary::default(),
            neighborhood: Neighborhood::default(),
            init_mode: InitMode::default(),
            hensel_notation: None,
            lookup_table_hex: None,
            lookup_table_array: None,
            n_states: default_n_states(),
            threshold: default_threshold(),
        }
    }
}

/// 2D 元胞自动机生成器
///
/// 支持 Life-like (B/S 记法)、Totalistic、WireWorld、循环 CA、
/// Hensel 各向同性非总量规则和完整查找表规则，
/// 可配置多值离散状态、边界条件和邻域类型。
pub struct CellularAutomaton2D {
    rule: Rule2D,
    lut: Option<LifeLikeLUT>,
    d_state: u8,
    rows: usize,
    cols: usize,
    boundary: Boundary,
    neighborhood: Neighborhood,
    init_mode: InitMode,
}

impl CellularAutomaton2D {
    /// 执行一步 2D 演化
    fn evolve(
        src: &[u8],
        dst: &mut [u8],
        rows: usize,
        cols: usize,
        rule: &Rule2D,
        lut: &Option<LifeLikeLUT>,
        boundary: &Boundary,
        neighborhood: &Neighborhood,
        d_state: u8,
    ) {
        let offsets: &[(i32, i32)] = match neighborhood {
            Neighborhood::Moore => &MOORE_2D_OFFSETS,
            Neighborhood::VonNeumann => &VONNEUMANN_2D_OFFSETS,
        };

        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let current = src[idx];

                match rule {
                    Rule2D::LifeLike { .. } => {
                        let lut = lut.as_ref().expect("LifeLike rule requires LUT");
                        let mut alive_count: usize = 0;
                        for &(dr, dc) in offsets {
                            let n = get_neighbor_2d(src, r, c, rows, cols, dr, dc, boundary);
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
                    Rule2D::Totalistic { transition_table } => {
                        let mut state_sum: usize = 0;
                        for &(dr, dc) in offsets {
                            let n = get_neighbor_2d(src, r, c, rows, cols, dr, dc, boundary);
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
                    Rule2D::WireWorld => {
                        // WireWorld 转移规则:
                        // 0(空)→0, 1(头)→2(尾), 2(尾)→3(铜), 3(铜)→1(头) 若恰好1或2个头邻居
                        dst[idx] = match current {
                            0 => 0,
                            1 => 2,
                            2 => 3,
                            3 => {
                                let mut head_count: usize = 0;
                                for &(dr, dc) in offsets {
                                    let n =
                                        get_neighbor_2d(src, r, c, rows, cols, dr, dc, boundary);
                                    if n == 1 {
                                        head_count += 1;
                                    }
                                }
                                if head_count == 1 || head_count == 2 {
                                    1
                                } else {
                                    3
                                }
                            }
                            _ => 0,
                        };
                    }
                    Rule2D::Cyclic {
                        n_states,
                        threshold,
                    } => {
                        // 循环 CA: 若邻居中有 ≥ threshold 个处于下一状态，则前进
                        let next_state = ((current as u16 + 1) % *n_states as u16) as u8;
                        let mut next_count: usize = 0;
                        for &(dr, dc) in offsets {
                            let n = get_neighbor_2d(src, r, c, rows, cols, dr, dc, boundary);
                            if n == next_state {
                                next_count += 1;
                            }
                        }
                        dst[idx] = if next_count >= *threshold as usize {
                            next_state
                        } else {
                            current
                        };
                    }
                    Rule2D::Hensel { table } | Rule2D::LookupTable { table } => {
                        // 9 位邻域索引: bit0..7=8 邻居(按 MOORE_2D_OFFSETS 顺序), bit8=中心格
                        // 索引 = (center_state) * 256 + neighbor_8bit
                        let mut neighbor_bits: usize = 0;
                        for (i, &(dr, dc)) in offsets.iter().enumerate() {
                            let n = get_neighbor_2d(src, r, c, rows, cols, dr, dc, boundary);
                            if n > 0 {
                                neighbor_bits |= 1 << i;
                            }
                        }
                        let index = (current as usize) * 256 + neighbor_bits;
                        dst[idx] = if index < table.len() && table[index] {
                            1
                        } else {
                            0
                        };
                    }
                }
            }
        }
    }
}

impl Generator for CellularAutomaton2D {
    fn name(&self) -> &'static str {
        "cellular_automaton_2d"
    }

    fn generate_stream(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let rows = self.rows;
        let cols = self.cols;
        let d_state = self.d_state;
        let boundary = self.boundary.clone();
        let neighborhood = self.neighborhood.clone();
        let init_mode = self.init_mode.clone();
        let rule = self.rule.clone();
        let lut = self.lut.clone();
        let seq_limit = params.seq_length;

        // 用种子 PRNG 生成初始状态
        let mut rng = SeedRng::new(seed);
        let grid_size = rows * cols;
        let mut buf_a = vec![0u8; grid_size];

        match init_mode {
            InitMode::Random => {
                for cell in buf_a.iter_mut() {
                    *cell = rng.next_usize(d_state as usize) as u8;
                }
            }
            InitMode::SingleCenter => {
                buf_a[(rows / 2) * cols + (cols / 2)] = 1;
            }
        }

        let mut buf_b = vec![0u8; grid_size];
        let mut step_counter: u64 = 0;

        let iter = std::iter::from_fn(move || {
            let step = step_counter;
            step_counter += 1;

            // 构建当前帧
            let values: Vec<FrameState> = buf_a
                .iter()
                .map(|&v| FrameState::Integer(v as i64))
                .collect();
            let frame = SequenceFrame::new(step, FrameData { values });

            // 执行演化
            CellularAutomaton2D::evolve(
                &buf_a,
                &mut buf_b,
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

/// 2D 元胞自动机工厂函数
pub fn ca2d_factory(extensions: &HashMap<String, Value>) -> CoreResult<Box<dyn Generator>> {
    // 应用预设解析（若指定）
    let ext = ca_rules::apply_preset_to_extensions(extensions, 2)?;
    let params: Ca2dParams = deserialize_extensions(&ext)?;

    if params.rows == 0 {
        return Err(CoreError::InvalidParams(
            "CA2D rows must be greater than 0".into(),
        ));
    }
    if params.cols == 0 {
        return Err(CoreError::InvalidParams(
            "CA2D cols must be greater than 0".into(),
        ));
    }
    if params.d_state < 2 {
        return Err(CoreError::InvalidParams(
            "CA2D d_state must be at least 2".into(),
        ));
    }

    let max_neighbors = match params.neighborhood {
        Neighborhood::Moore => 8,
        Neighborhood::VonNeumann => 4,
    };

    let (rule, lut) = match params.rule_type.as_str() {
        "lifelike" => {
            if params.d_state != 2 {
                return Err(CoreError::InvalidParams(
                    "LifeLike rules require d_state == 2".into(),
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
            let rule = Rule2D::LifeLike {
                birth: params.birth,
                survival: params.survival,
            };
            (rule, Some(lut))
        }
        "totalistic" => {
            if params.totalistic_table.is_empty() {
                return Err(CoreError::InvalidParams(
                    "Totalistic rule requires a non-empty totalistic_table".into(),
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
            let rule = Rule2D::Totalistic {
                transition_table: params.totalistic_table,
            };
            (rule, None)
        }
        "wireworld" => {
            if params.d_state != 4 {
                return Err(CoreError::InvalidParams(
                    "WireWorld requires d_state == 4".into(),
                ));
            }
            (Rule2D::WireWorld, None)
        }
        "cyclic" => {
            if params.n_states < 2 {
                return Err(CoreError::InvalidParams(
                    "Cyclic CA requires n_states >= 2".into(),
                ));
            }
            if params.threshold < 1 {
                return Err(CoreError::InvalidParams(
                    "Cyclic CA requires threshold >= 1".into(),
                ));
            }
            // 循环 CA 的 d_state 由 n_states 决定
            let rule = Rule2D::Cyclic {
                n_states: params.n_states,
                threshold: params.threshold,
            };
            (rule, None)
        }
        "hensel" => {
            if params.d_state != 2 {
                return Err(CoreError::InvalidParams(
                    "Hensel rules require d_state == 2".into(),
                ));
            }
            let notation = params.hensel_notation.as_deref().ok_or_else(|| {
                CoreError::InvalidParams(
                    "Hensel rule_type requires hensel_notation parameter".into(),
                )
            })?;
            let table = ca_rules::parse_hensel_notation(notation)?;
            (Rule2D::Hensel { table }, None)
        }
        "lookuptable" => {
            if params.d_state != 2 {
                return Err(CoreError::InvalidParams(
                    "LookupTable rules require d_state == 2".into(),
                ));
            }
            let table = if let Some(ref hex) = params.lookup_table_hex {
                ca_rules::parse_lookup_hex(hex)?
            } else if let Some(ref arr) = params.lookup_table_array {
                ca_rules::parse_lookup_array(arr)?
            } else {
                return Err(CoreError::InvalidParams(
                    "LookupTable rule_type requires lookup_table_hex or lookup_table_array parameter".into(),
                ));
            };
            (Rule2D::LookupTable { table }, None)
        }
        other => {
            return Err(CoreError::InvalidParams(format!(
                "unknown rule_type '{}', expected 'lifelike', 'totalistic', 'wireworld', 'cyclic', 'hensel', or 'lookuptable'",
                other
            )));
        }
    };

    // 循环 CA 需要覆盖 d_state
    let effective_d_state = match &rule {
        Rule2D::Cyclic { n_states, .. } => *n_states,
        _ => params.d_state,
    };

    Ok(Box::new(CellularAutomaton2D {
        rule,
        lut,
        d_state: effective_d_state,
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
        let gen1 = ca2d_factory(&HashMap::new()).unwrap();
        let gen2 = ca2d_factory(&HashMap::new()).unwrap();

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
        let gen = ca2d_factory(&HashMap::new()).unwrap();

        let frames1 = collect_frames(gen.as_ref(), 100, 5);
        let frames2 = collect_frames(gen.as_ref(), 200, 5);

        let all_same = frames1
            .iter()
            .zip(frames2.iter())
            .all(|(a, b)| a.state.values == b.state.values);
        assert!(!all_same, "Different seeds should produce different output");
    }

    #[test]
    fn test_conway_life_blinker() {
        let mut extensions = HashMap::new();
        extensions.insert("rule_type".to_string(), json!("lifelike"));
        extensions.insert("birth".to_string(), json!([3]));
        extensions.insert("survival".to_string(), json!([2, 3]));
        extensions.insert("d_state".to_string(), json!(2));
        extensions.insert("rows".to_string(), json!(5));
        extensions.insert("cols".to_string(), json!(5));
        extensions.insert("boundary".to_string(), json!("fixed"));
        extensions.insert("neighborhood".to_string(), json!("moore"));
        extensions.insert("init_mode".to_string(), json!("singlecenter"));

        let gen = ca2d_factory(&extensions).unwrap();
        let params = GenParams::simple(3);
        let frames = gen.generate_stream(42, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 3);

        for f in &frames {
            assert_eq!(f.state.values.len(), 25);
        }

        for f in &frames {
            for v in &f.state.values {
                let val = v.as_integer().unwrap();
                assert!(val == 0 || val == 1, "Value should be 0 or 1, got {}", val);
            }
        }
    }

    #[test]
    fn test_periodic_boundary() {
        let mut extensions = HashMap::new();
        extensions.insert("boundary".to_string(), json!("periodic"));
        extensions.insert("rows".to_string(), json!(16));
        extensions.insert("cols".to_string(), json!(16));

        let gen = ca2d_factory(&extensions).unwrap();
        let frames = collect_frames(gen.as_ref(), 7, 5);
        assert_eq!(frames.len(), 5);
        for f in &frames {
            assert_eq!(f.state.values.len(), 16 * 16);
        }
    }

    #[test]
    fn test_fixed_boundary() {
        let mut extensions = HashMap::new();
        extensions.insert("boundary".to_string(), json!("fixed"));
        extensions.insert("rows".to_string(), json!(8));
        extensions.insert("cols".to_string(), json!(8));

        let gen = ca2d_factory(&extensions).unwrap();
        let frames = collect_frames(gen.as_ref(), 3, 3);
        assert_eq!(frames.len(), 3);
        for f in &frames {
            assert_eq!(f.state.values.len(), 64);
        }
    }

    #[test]
    fn test_reflective_boundary() {
        let mut extensions = HashMap::new();
        extensions.insert("boundary".to_string(), json!("reflective"));
        extensions.insert("rows".to_string(), json!(8));
        extensions.insert("cols".to_string(), json!(8));

        let gen = ca2d_factory(&extensions).unwrap();
        let frames = collect_frames(gen.as_ref(), 3, 3);
        assert_eq!(frames.len(), 3);
    }

    #[test]
    fn test_vonneumann_neighborhood() {
        let mut extensions = HashMap::new();
        extensions.insert("neighborhood".to_string(), json!("vonneumann"));
        extensions.insert("rows".to_string(), json!(8));
        extensions.insert("cols".to_string(), json!(8));

        let gen = ca2d_factory(&extensions).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 5);
        assert_eq!(frames.len(), 5);
    }

    #[test]
    fn test_single_center_init() {
        let mut extensions = HashMap::new();
        extensions.insert("init_mode".to_string(), json!("singlecenter"));
        extensions.insert("rows".to_string(), json!(5));
        extensions.insert("cols".to_string(), json!(5));

        let gen = ca2d_factory(&extensions).unwrap();
        let params = GenParams::simple(1);
        let frames = gen.generate_stream(12345, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 1);

        let alive_count = frames[0]
            .state
            .values
            .iter()
            .filter(|v| v.as_integer().unwrap() > 0)
            .count();
        assert_eq!(alive_count, 1, "SingleCenter init should have exactly 1 alive cell");

        let center_idx = 2 * 5 + 2;
        assert_eq!(frames[0].state.values[center_idx].as_integer().unwrap(), 1);
    }

    #[test]
    fn test_output_dimensions() {
        let mut extensions = HashMap::new();
        extensions.insert("rows".to_string(), json!(10));
        extensions.insert("cols".to_string(), json!(20));

        let gen = ca2d_factory(&extensions).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 3);
        for f in &frames {
            assert_eq!(f.state.values.len(), 10 * 20);
        }
    }

    #[test]
    fn test_output_values_in_range() {
        let mut extensions = HashMap::new();
        extensions.insert("d_state".to_string(), json!(2));
        extensions.insert("rows".to_string(), json!(8));
        extensions.insert("cols".to_string(), json!(8));

        let gen = ca2d_factory(&extensions).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 5);
        for f in &frames {
            for v in &f.state.values {
                let val = v.as_integer().unwrap();
                assert!(
                    val >= 0 && val < 2,
                    "Value should be in [0, d_state-1], got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_totalistic_rule() {
        let mut extensions = HashMap::new();
        extensions.insert("rule_type".to_string(), json!("totalistic"));
        extensions.insert("d_state".to_string(), json!(4));
        let table: Vec<u8> = (0..25).map(|i| (i % 4) as u8).collect();
        extensions.insert("totalistic_table".to_string(), json!(table));
        extensions.insert("rows".to_string(), json!(4));
        extensions.insert("cols".to_string(), json!(4));

        let gen = ca2d_factory(&extensions).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 3);
        assert_eq!(frames.len(), 3);
        for f in &frames {
            assert_eq!(f.state.values.len(), 16);
            for v in &f.state.values {
                let val = v.as_integer().unwrap();
                assert!(val >= 0 && val < 4, "Value should be in [0,3], got {}", val);
            }
        }
    }

    #[test]
    fn test_lifelike_requires_dstate2() {
        let mut extensions = HashMap::new();
        extensions.insert("rule_type".to_string(), json!("lifelike"));
        extensions.insert("d_state".to_string(), json!(3));

        let result = ca2d_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_rows_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("rows".to_string(), json!(0));
        let result = ca2d_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_cols_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("cols".to_string(), json!(0));
        let result = ca2d_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_rule_type_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("rule_type".to_string(), json!("unknown"));
        let result = ca2d_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_totalistic_empty_table_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("rule_type".to_string(), json!("totalistic"));
        extensions.insert("totalistic_table".to_string(), json!([]));
        let result = ca2d_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_totalistic_short_table_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("rule_type".to_string(), json!("totalistic"));
        extensions.insert("d_state".to_string(), json!(4));
        extensions.insert("totalistic_table".to_string(), json!([0, 1]));
        let result = ca2d_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_step_indices_sequential() {
        let gen = ca2d_factory(&HashMap::new()).unwrap();
        let params = GenParams::simple(20);
        let frames = gen.generate_stream(99, &params).unwrap().collect::<Vec<_>>();
        for (i, f) in frames.iter().enumerate() {
            assert_eq!(f.step_index, i as u64);
        }
    }

    #[test]
    fn test_unbounded_stream() {
        let gen = ca2d_factory(&HashMap::new()).unwrap();
        let params = GenParams::simple(0);
        let frames = gen
            .generate_stream(42, &params)
            .unwrap()
            .take(50)
            .collect::<Vec<_>>();
        assert_eq!(frames.len(), 50);
    }

    #[test]
    fn test_default_params() {
        let gen = ca2d_factory(&HashMap::new()).unwrap();
        assert_eq!(gen.name(), "cellular_automaton_2d");
        let params = GenParams::simple(3);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 3);
        for f in &frames {
            assert_eq!(f.state.values.len(), 64 * 64);
        }
    }

    // ---- WireWorld 测试 ----

    #[test]
    fn test_wireworld_signal_propagation() {
        let mut ext = HashMap::new();
        ext.insert("rule_type".to_string(), json!("wireworld"));
        ext.insert("d_state".to_string(), json!(4));
        ext.insert("rows".to_string(), json!(1));
        ext.insert("cols".to_string(), json!(5));
        ext.insert("boundary".to_string(), json!("fixed"));
        ext.insert("init_mode".to_string(), json!("singlecenter"));
        let gen = ca2d_factory(&ext).unwrap();

        // 手动构建初始状态：头-尾-铜-铜-铜
        let params = GenParams::simple(5);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 5);
        // 所有帧应有 5 个单元格，值在 0..4 范围内
        for f in &frames {
            assert_eq!(f.state.values.len(), 5);
            for v in &f.state.values {
                let val = v.as_integer().unwrap();
                assert!(val >= 0 && val < 4, "WireWorld state should be 0-3, got {}", val);
            }
        }
    }

    #[test]
    fn test_wireworld_requires_dstate4() {
        let mut ext = HashMap::new();
        ext.insert("rule_type".to_string(), json!("wireworld"));
        ext.insert("d_state".to_string(), json!(2));
        assert!(ca2d_factory(&ext).is_err());
    }

    // ---- 循环 CA 测试 ----

    #[test]
    fn test_cyclic_evolution() {
        let mut ext = HashMap::new();
        ext.insert("rule_type".to_string(), json!("cyclic"));
        ext.insert("n_states".to_string(), json!(4));
        ext.insert("threshold".to_string(), json!(1));
        ext.insert("rows".to_string(), json!(8));
        ext.insert("cols".to_string(), json!(8));
        let gen = ca2d_factory(&ext).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 10);
        assert_eq!(frames.len(), 10);
        // 所有值应在 0..4 范围内
        for f in &frames {
            for v in &f.state.values {
                let val = v.as_integer().unwrap();
                assert!(val >= 0 && val < 4, "Cyclic CA state should be 0-3, got {}", val);
            }
        }
    }

    #[test]
    fn test_cyclic_requires_nstates2() {
        let mut ext = HashMap::new();
        ext.insert("rule_type".to_string(), json!("cyclic"));
        ext.insert("n_states".to_string(), json!(1));
        assert!(ca2d_factory(&ext).is_err());
    }

    // ---- Hensel 测试 ----

    #[test]
    fn test_hensel_conway_equals_lifelike() {
        // Hensel B3/S23 应与 LifeLike B3/S23 产生相同的演化
        let mut hensel_ext = HashMap::new();
        hensel_ext.insert("rule_type".to_string(), json!("hensel"));
        hensel_ext.insert("hensel_notation".to_string(), json!("B3/S23"));
        hensel_ext.insert("rows".to_string(), json!(16));
        hensel_ext.insert("cols".to_string(), json!(16));

        let mut ll_ext = HashMap::new();
        ll_ext.insert("rule_type".to_string(), json!("lifelike"));
        ll_ext.insert("birth".to_string(), json!([3]));
        ll_ext.insert("survival".to_string(), json!([2, 3]));
        ll_ext.insert("rows".to_string(), json!(16));
        ll_ext.insert("cols".to_string(), json!(16));

        let gen_hensel = ca2d_factory(&hensel_ext).unwrap();
        let gen_ll = ca2d_factory(&ll_ext).unwrap();

        let params = GenParams::simple(10);
        let frames_hensel = gen_hensel.generate_stream(42, &params).unwrap().collect::<Vec<_>>();
        let frames_ll = gen_ll.generate_stream(42, &params).unwrap().collect::<Vec<_>>();

        for i in 0..10 {
            assert_eq!(
                frames_hensel[i].state.values,
                frames_ll[i].state.values,
                "Hensel B3/S23 should match LifeLike B3/S23 at frame {}",
                i
            );
        }
    }

    #[test]
    fn test_hensel_ameyalli() {
        let mut ext = HashMap::new();
        ext.insert("rule_type".to_string(), json!("hensel"));
        ext.insert("hensel_notation".to_string(), json!("B2ci3ar4krtz5cq6c7ce/S01e2ek3qj4kt5ceayq6cki7c8"));
        ext.insert("rows".to_string(), json!(8));
        ext.insert("cols".to_string(), json!(8));

        let gen = ca2d_factory(&ext).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 5);
        assert_eq!(frames.len(), 5);
    }

    #[test]
    fn test_hensel_requires_notation() {
        let mut ext = HashMap::new();
        ext.insert("rule_type".to_string(), json!("hensel"));
        assert!(ca2d_factory(&ext).is_err());
    }

    // ---- LookupTable 测试 ----

    #[test]
    fn test_lookuptable_from_array() {
        // 构建一个简单的查找表：Conway's Life B3/S23
        let mut table = vec![0u8; 512];
        for pattern in 0u8..=255 {
            let count = pattern.count_ones() as usize;
            // B 部：center=0，index=pattern
            if count == 3 {
                table[pattern as usize] = 1;
            }
            // S 部：center=1，index=256+pattern
            if count == 2 || count == 3 {
                table[256 + pattern as usize] = 1;
            }
        }

        let mut ext = HashMap::new();
        ext.insert("rule_type".to_string(), json!("lookuptable"));
        ext.insert("lookup_table_array".to_string(), json!(table));
        ext.insert("rows".to_string(), json!(8));
        ext.insert("cols".to_string(), json!(8));

        let gen = ca2d_factory(&ext).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 5);
        assert_eq!(frames.len(), 5);
    }

    #[test]
    fn test_lookuptable_requires_data() {
        let mut ext = HashMap::new();
        ext.insert("rule_type".to_string(), json!("lookuptable"));
        assert!(ca2d_factory(&ext).is_err());
    }

    // ---- 预设测试 ----

    #[test]
    fn test_preset_game_of_life() {
        let mut ext = HashMap::new();
        ext.insert("preset".to_string(), json!("game_of_life"));
        ext.insert("rows".to_string(), json!(8));
        ext.insert("cols".to_string(), json!(8));
        let gen = ca2d_factory(&ext).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 5);
        assert_eq!(frames.len(), 5);
    }

    #[test]
    fn test_preset_highlife() {
        let mut ext = HashMap::new();
        ext.insert("preset".to_string(), json!("highlife"));
        ext.insert("rows".to_string(), json!(8));
        ext.insert("cols".to_string(), json!(8));
        let gen = ca2d_factory(&ext).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 5);
        assert_eq!(frames.len(), 5);
    }

    #[test]
    fn test_preset_wireworld() {
        let mut ext = HashMap::new();
        ext.insert("preset".to_string(), json!("wireworld"));
        ext.insert("rows".to_string(), json!(8));
        ext.insert("cols".to_string(), json!(16));
        let gen = ca2d_factory(&ext).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 10);
        assert_eq!(frames.len(), 10);
        // WireWorld 应有 4 种状态
        for f in &frames {
            for v in &f.state.values {
                let val = v.as_integer().unwrap();
                assert!(val >= 0 && val < 4);
            }
        }
    }

    #[test]
    fn test_preset_cyclic() {
        let mut ext = HashMap::new();
        ext.insert("preset".to_string(), json!("cyclic"));
        ext.insert("rows".to_string(), json!(8));
        ext.insert("cols".to_string(), json!(8));
        let gen = ca2d_factory(&ext).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 10);
        assert_eq!(frames.len(), 10);
    }

    #[test]
    fn test_preset_ameyalli() {
        let mut ext = HashMap::new();
        ext.insert("preset".to_string(), json!("ameyalli"));
        ext.insert("rows".to_string(), json!(8));
        ext.insert("cols".to_string(), json!(8));
        let gen = ca2d_factory(&ext).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 5);
        assert_eq!(frames.len(), 5);
    }

    #[test]
    fn test_preset_day_night() {
        let mut ext = HashMap::new();
        ext.insert("preset".to_string(), json!("day_night"));
        ext.insert("rows".to_string(), json!(8));
        ext.insert("cols".to_string(), json!(8));
        let gen = ca2d_factory(&ext).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 5);
        assert_eq!(frames.len(), 5);
    }

    #[test]
    fn test_unknown_preset_rejected() {
        let mut ext = HashMap::new();
        ext.insert("preset".to_string(), json!("nonexistent_rule"));
        assert!(ca2d_factory(&ext).is_err());
    }

    #[test]
    fn test_preset_preserves_grid_params() {
        let mut ext = HashMap::new();
        ext.insert("preset".to_string(), json!("highlife"));
        ext.insert("rows".to_string(), json!(10));
        ext.insert("cols".to_string(), json!(20));
        let gen = ca2d_factory(&ext).unwrap();
        let frames = collect_frames(gen.as_ref(), 42, 2);
        assert_eq!(frames[0].state.values.len(), 10 * 20);
    }
}
