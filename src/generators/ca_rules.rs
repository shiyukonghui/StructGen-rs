//! 元胞自动机著名规则预设与解析
//!
//! 提供 1D/2D/3D 规则预设解析、Hensel 各向同性非总量规则解析、
//! 完整查找表解析，以及预设合并逻辑。
//!
//! # 预设命名
//!
//! - 1D: `rule30`, `rule54`, `rule90`, `rule110`, `rule184`
//! - 2D: `game_of_life`/`conway`, `highlife`, `day_night`, `seeds`,
//!        `diamoeba`, `replicator_2d`, `maze`, `wireworld`, `cyclic`,
//!        `ameyalli`, `xrule`
//! - 3D: `life_3d`, `cyclic_3d`, `fredkin_3d`

use crate::core::*;
use serde_json::{json, Value};
use std::collections::HashMap;

// ============================================================
// D4 对称群操作（用于 Hensel 各向同性规则解析）
// ============================================================

/// 8 位邻域位布局（与 MOORE_2D_OFFSETS 顺序一致）:
///
/// ```text
/// bit0=NW(-1,-1)  bit1=N(-1,0)   bit2=NE(-1,1)
/// bit3=W(0,-1)                    bit4=E(0,1)
/// bit5=SW(1,-1)   bit6=S(1,0)    bit7=SE(1,1)
/// ```
///
/// D4 对称变换表：`source[i]` 表示变换后新位置 i 的值来自旧位置 source[i]。
const D4_PERMS: [[usize; 8]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7], // 恒等
    [5, 3, 0, 6, 1, 7, 4, 2], // 顺时针旋转 90°
    [7, 6, 5, 4, 3, 2, 1, 0], // 旋转 180°
    [2, 4, 7, 1, 6, 0, 3, 5], // 顺时针旋转 270°
    [2, 1, 0, 4, 3, 7, 6, 5], // 水平反射 (左右)
    [5, 6, 7, 3, 4, 0, 1, 2], // 垂直反射 (上下)
    [0, 3, 5, 1, 6, 2, 4, 7], // 主对角线反射 (NW-SE)
    [7, 4, 2, 6, 1, 5, 3, 0], // 反对角线反射 (NE-SW)
];

/// 对 8 位邻域模式应用对称变换
fn apply_perm(pattern: u8, perm: &[usize; 8]) -> u8 {
    let mut result = 0u8;
    for i in 0..8 {
        if pattern & (1 << perm[i]) != 0 {
            result |= 1 << i;
        }
    }
    result
}

/// 计算 8 位邻域模式的规范形式（D4 对称群下的最小值）
fn canonicalize(pattern: u8) -> u8 {
    let mut min_val = pattern;
    for perm in &D4_PERMS[1..] {
        let val = apply_perm(pattern, perm);
        if val < min_val {
            min_val = val;
        }
    }
    min_val
}

/// Hensel 代码字母分配序列（按规范形式从小到大排序）
const HENSEL_LETTERS: &[u8] = b"cekainyqjrtwz";

/// 计算 Hensel 各向同性轨道映射
///
/// 返回 `orbit_table[count][orbit_index] = (letter, Vec<u8>)`，
/// 其中 Vec<u8> 是该轨道包含的所有 8 位邻域模式。
fn compute_hensel_orbits() -> Vec<Vec<(char, Vec<u8>)>> {
    let mut result = vec![vec![]; 9];
    let mut assigned = [false; 256];

    for pattern in 0u8..=255 {
        if assigned[pattern as usize] {
            continue;
        }
        let count = pattern.count_ones() as usize;
        let _canon = canonicalize(pattern);

        // 收集该轨道的所有模式
        let mut orbit_patterns: Vec<u8> = Vec::new();
        for perm in &D4_PERMS {
            let transformed = apply_perm(pattern, perm);
            if !assigned[transformed as usize] {
                assigned[transformed as usize] = true;
                orbit_patterns.push(transformed);
            }
        }
        orbit_patterns.sort();
        orbit_patterns.dedup();

        // 规范代表（最小值）
        let canon_rep = *orbit_patterns.iter().min().unwrap_or(&pattern);

        result[count].push((canon_rep, orbit_patterns));
    }

    // 对每个计数的轨道按规范代表排序，并分配 Hensel 代码字母
    for count in 0..=8 {
        result[count].sort_by_key(|(canon_rep, _)| *canon_rep);
        for (_i, (canon_rep, _patterns)) in result[count].iter_mut().enumerate() {
            let _letter = HENSEL_LETTERS[_i] as char;
            // canon_rep 已用于排序，清零
            *canon_rep = 0;
        }
    }

    // 重构返回值：只保留 (letter, patterns)
    result
        .into_iter()
        .map(|orbits| {
            orbits
                .into_iter()
                .enumerate()
                .map(|(i, (_, patterns))| (HENSEL_LETTERS[i] as char, patterns))
                .collect()
        })
        .collect()
}

/// 全局缓存的 Hensel 轨道映射
static HENSEL_ORBITS: std::sync::OnceLock<Vec<Vec<(char, Vec<u8>)>>> =
    std::sync::OnceLock::new();

/// 获取缓存的 Hensel 轨道映射
fn get_hensel_orbits() -> &'static Vec<Vec<(char, Vec<u8>)>> {
    HENSEL_ORBITS.get_or_init(compute_hensel_orbits)
}

// ============================================================
// Hensel 记法解析
// ============================================================

/// 解析 Hensel 记法字符串为 512 项查找表
///
/// 格式: `B<codes>/S<codes>`
///
/// codes 为 `数字+字母组` 序列，如 `"2ci3ar"` 表示 count=2 的 c,i 轨道和
/// count=3 的 a,r 轨道。只有数字没有字母表示该计数的所有轨道都激活。
///
/// 索引约定:
/// - `index = (center as usize) * 256 + neighbor_8bit`
/// - B 部设置 table[0..255]（center=0，出生条件）
/// - S 部设置 table[256..511]（center=1，存活条件）
pub fn parse_hensel_notation(notation: &str) -> CoreResult<[bool; 512]> {
    let mut table = [false; 512];
    let orbits = get_hensel_orbits();

    // 构建 (count, letter) → patterns 查找
    let mut code_map: HashMap<(usize, char), &Vec<u8>> = HashMap::new();
    for count in 0..=8usize {
        for (letter, patterns) in &orbits[count] {
            code_map.insert((count, *letter), patterns);
        }
    }

    // 按 / 分割
    let parts: Vec<&str> = notation.split('/').collect();
    if parts.len() != 2 {
        return Err(CoreError::InvalidParams(format!(
            "Hensel notation must contain exactly one '/', got: {}",
            notation
        )));
    }

    let b_part = parts[0].strip_prefix('B').ok_or_else(|| {
        CoreError::InvalidParams(format!(
            "Hensel B-part must start with 'B', got: {}",
            parts[0]
        ))
    })?;
    let s_part = parts[1].strip_prefix('S').ok_or_else(|| {
        CoreError::InvalidParams(format!(
            "Hensel S-part must start with 'S', got: {}",
            parts[1]
        ))
    })?;

    parse_hensel_part(b_part, &code_map, &mut table[..256])?;
    parse_hensel_part(s_part, &code_map, &mut table[256..])?;

    Ok(table)
}

/// 解析 Hensel 的 B 或 S 部分
fn parse_hensel_part(
    part: &str,
    code_map: &HashMap<(usize, char), &Vec<u8>>,
    table: &mut [bool],
) -> CoreResult<()> {
    let chars: Vec<char> = part.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // 读取数字（邻域计数）— Hensel 记法中每个 count 都是单个数字 (0-8)
        if !chars[i].is_ascii_digit() {
            return Err(CoreError::InvalidParams(format!(
                "Expected digit in Hensel notation at position {}, got '{}'",
                i, chars[i]
            )));
        }

        let count: usize = chars[i].to_digit(10).unwrap() as usize;
        i += 1;

        if count > 8 {
            return Err(CoreError::InvalidParams(format!(
                "Count {} exceeds max neighbors 8 in Hensel notation",
                count
            )));
        }

        // 读取字母代码（直到下一个数字或字符串结束）
        let mut letters = String::new();
        while i < chars.len() && chars[i].is_ascii_lowercase() {
            letters.push(chars[i]);
            i += 1;
        }

        if letters.is_empty() {
            // 没有字母 → 该计数的所有轨道都激活
            for (_, patterns) in code_map.iter().filter(|((c, _), _)| *c == count) {
                for &pattern in *patterns {
                    table[pattern as usize] = true;
                }
            }
        } else {
            for letter in letters.chars() {
                let patterns = code_map.get(&(count, letter)).ok_or_else(|| {
                    CoreError::InvalidParams(format!(
                        "Unknown Hensel code '{}' at count {}",
                        letter, count
                    ))
                })?;
                for &pattern in *patterns {
                    table[pattern as usize] = true;
                }
            }
        }
    }

    Ok(())
}

// ============================================================
// 查找表解析
// ============================================================

/// 解析十六进制查找表字符串为 512 项布尔数组
///
/// 输入为 128 个十六进制字符，每字符 4 位，共 512 位。
/// 最高有效位在前（大端序）。
pub fn parse_lookup_hex(hex: &str) -> CoreResult<[bool; 512]> {
    if hex.len() != 128 {
        return Err(CoreError::InvalidParams(format!(
            "Lookup table hex string must be 128 chars (512 bits), got {} chars",
            hex.len()
        )));
    }

    let mut table = [false; 512];
    for (i, ch) in hex.chars().enumerate() {
        let digit = ch.to_digit(16).ok_or_else(|| {
            CoreError::InvalidParams(format!(
                "Invalid hex character '{}' at position {}",
                ch, i
            ))
        })? as u8;

        for bit in 0..4u8 {
            let table_idx = i * 4 + (3 - bit as usize);
            if table_idx < 512 {
                table[table_idx] = (digit >> bit) & 1 == 1;
            }
        }
    }

    Ok(table)
}

/// 解析字节数组查找表为 512 项布尔数组
///
/// 输入为 512 个整数值（0 或 1）。
pub fn parse_lookup_array(arr: &[u8]) -> CoreResult<[bool; 512]> {
    if arr.len() != 512 {
        return Err(CoreError::InvalidParams(format!(
            "Lookup table array must have 512 entries, got {}",
            arr.len()
        )));
    }

    let mut table = [false; 512];
    for (i, &val) in arr.iter().enumerate() {
        match val {
            0 => table[i] = false,
            1 => table[i] = true,
            _ => {
                return Err(CoreError::InvalidParams(format!(
                    "Lookup table array entry {} must be 0 or 1, got {}",
                    i, val
                )))
            }
        }
    }

    Ok(table)
}

// ============================================================
// 预设解析
// ============================================================

/// 1D CA 规则预设参数键
const RULE_PARAM_KEYS_1D: &[&str] = &["rule"];

/// 2D CA 规则预设参数键
const RULE_PARAM_KEYS_2D: &[&str] = &[
    "rule_type",
    "birth",
    "survival",
    "totalistic_table",
    "d_state",
    "hensel_notation",
    "lookup_table_hex",
    "lookup_table_array",
    "n_states",
    "threshold",
];

/// 3D CA 规则预设参数键
const RULE_PARAM_KEYS_3D: &[&str] = &[
    "rule_type",
    "birth",
    "survival",
    "totalistic_table",
    "d_state",
    "n_states",
    "threshold",
];

/// 解析 1D CA 预设
fn resolve_preset_1d(name: &str) -> CoreResult<HashMap<String, Value>> {
    match name {
        "rule30" | "wolfram30" => Ok(HashMap::from([("rule".into(), json!(30))])),
        "rule54" | "wolfram54" => Ok(HashMap::from([("rule".into(), json!(54))])),
        "rule90" | "wolfram90" => Ok(HashMap::from([("rule".into(), json!(90))])),
        "rule110" | "wolfram110" => Ok(HashMap::from([("rule".into(), json!(110))])),
        "rule184" | "wolfram184" => Ok(HashMap::from([("rule".into(), json!(184))])),
        _ => Err(CoreError::InvalidParams(format!(
            "Unknown 1D CA preset '{}'",
            name
        ))),
    }
}

/// 解析 2D CA 预设
fn resolve_preset_2d(name: &str) -> CoreResult<HashMap<String, Value>> {
    match name {
        // Life-like 规则
        "game_of_life" | "conway" => Ok(HashMap::from([
            ("rule_type".into(), json!("lifelike")),
            ("birth".into(), json!([3])),
            ("survival".into(), json!([2, 3])),
            ("d_state".into(), json!(2)),
        ])),
        "highlife" => Ok(HashMap::from([
            ("rule_type".into(), json!("lifelike")),
            ("birth".into(), json!([3, 6])),
            ("survival".into(), json!([2, 3])),
            ("d_state".into(), json!(2)),
        ])),
        "day_night" => Ok(HashMap::from([
            ("rule_type".into(), json!("lifelike")),
            ("birth".into(), json!([3, 6, 7, 8])),
            ("survival".into(), json!([3, 4, 6, 7, 8])),
            ("d_state".into(), json!(2)),
        ])),
        "seeds" => Ok(HashMap::from([
            ("rule_type".into(), json!("lifelike")),
            ("birth".into(), json!([2])),
            ("survival".into(), json!([])),
            ("d_state".into(), json!(2)),
        ])),
        "diamoeba" => Ok(HashMap::from([
            ("rule_type".into(), json!("lifelike")),
            ("birth".into(), json!([3, 5, 6, 7, 8])),
            ("survival".into(), json!([5, 6, 7, 8])),
            ("d_state".into(), json!(2)),
        ])),
        "replicator_2d" => Ok(HashMap::from([
            ("rule_type".into(), json!("lifelike")),
            ("birth".into(), json!([1, 3, 5, 7])),
            ("survival".into(), json!([1, 3, 5, 7])),
            ("d_state".into(), json!(2)),
        ])),
        "maze" => Ok(HashMap::from([
            ("rule_type".into(), json!("lifelike")),
            ("birth".into(), json!([3])),
            ("survival".into(), json!([1, 2, 3, 4, 5])),
            ("d_state".into(), json!(2)),
        ])),
        // WireWorld
        "wireworld" => Ok(HashMap::from([
            ("rule_type".into(), json!("wireworld")),
            ("d_state".into(), json!(4)),
        ])),
        // 循环元胞自动机
        "cyclic" => Ok(HashMap::from([
            ("rule_type".into(), json!("cyclic")),
            ("n_states".into(), json!(14)),
            ("threshold".into(), json!(1)),
            ("d_state".into(), json!(14)),
        ])),
        // Ameyalli 规则 (各向同性非总量)
        "ameyalli" => Ok(HashMap::from([
            ("rule_type".into(), json!("hensel")),
            ("hensel_notation".into(), json!("B2ci3ar4krtz5cq6c7ce/S01e2ek3qj4kt5ceayq6cki7c8")),
            ("d_state".into(), json!(2)),
        ])),
        // X-Rule (非各向同性，完整查找表)
        // 注意：X-Rule 的精确 512 位查找表需从 Golly/DDLab 规则库获取。
        // 当前使用空占位符，用户需通过 lookup_table_hex 参数自行提供数据。
        "xrule" => Err(CoreError::InvalidParams(
            "X-Rule preset requires the exact 512-bit lookup table from Golly/DDLab. \
             Use rule_type: \"lookuptable\" with lookup_table_hex parameter instead."
                .into(),
        )),
        _ => Err(CoreError::InvalidParams(format!(
            "Unknown 2D CA preset '{}'",
            name
        ))),
    }
}

/// 解析 3D CA 预设
fn resolve_preset_3d(name: &str) -> CoreResult<HashMap<String, Value>> {
    match name {
        "life_3d" => Ok(HashMap::from([
            ("rule_type".into(), json!("lifelike")),
            ("birth".into(), json!([5, 6, 7])),
            ("survival".into(), json!([5, 6])),
            ("d_state".into(), json!(2)),
        ])),
        "cyclic_3d" => Ok(HashMap::from([
            ("rule_type".into(), json!("cyclic")),
            ("n_states".into(), json!(14)),
            ("threshold".into(), json!(1)),
            ("d_state".into(), json!(14)),
        ])),
        "fredkin_3d" => Ok(HashMap::from([
            ("rule_type".into(), json!("fredkin")),
            ("d_state".into(), json!(2)),
        ])),
        _ => Err(CoreError::InvalidParams(format!(
            "Unknown 3D CA preset '{}'",
            name
        ))),
    }
}

/// 将预设解析结果合并到扩展参数中
///
/// 预设仅覆盖规则相关参数，不覆盖网格参数（rows, cols, depth, boundary 等）。
/// 若用户同时提供了规则参数，预设值优先。
pub fn apply_preset_to_extensions(
    extensions: &HashMap<String, Value>,
    dimension: u8,
) -> CoreResult<HashMap<String, Value>> {
    let preset_val = match extensions.get("preset") {
        Some(v) => v,
        None => return Ok(extensions.clone()),
    };

    let preset_name = preset_val.as_str().ok_or_else(|| {
        CoreError::InvalidParams("preset must be a string".into())
    })?;

    let overrides = match dimension {
        1 => resolve_preset_1d(preset_name)?,
        2 => resolve_preset_2d(preset_name)?,
        3 => resolve_preset_3d(preset_name)?,
        _ => {
            return Err(CoreError::InvalidParams(format!(
                "Invalid dimension {} for preset resolution",
                dimension
            )))
        }
    };

    // 克隆扩展参数，移除规则相关键和 preset 键
    let mut result = extensions.clone();
    let rule_keys = match dimension {
        1 => RULE_PARAM_KEYS_1D,
        2 => RULE_PARAM_KEYS_2D,
        3 => RULE_PARAM_KEYS_3D,
        _ => unreachable!(),
    };
    for key in rule_keys {
        result.remove(*key);
    }
    result.remove("preset");

    // 合并预设覆盖值
    for (k, v) in overrides {
        result.insert(k, v);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- D4 对称群测试 ----

    #[test]
    fn test_d4_identity() {
        let p = 0b10100110; // 随机模式
        assert_eq!(apply_perm(p, &D4_PERMS[0]), p);
    }

    #[test]
    fn test_d4_rot180_is_reverse() {
        let p = 0b10100110;
        let rot180 = apply_perm(p, &D4_PERMS[2]);
        assert_eq!(rot180, p.reverse_bits());
    }

    #[test]
    fn test_d4_rot90_4x_identity() {
        let p = 0b11001010;
        let mut val = p;
        for _ in 0..4 {
            val = apply_perm(val, &D4_PERMS[1]); // Rot90CW
        }
        assert_eq!(val, p);
    }

    #[test]
    fn test_d4_reflect_double_identity() {
        let p = 0b10110100;
        let mut val = apply_perm(p, &D4_PERMS[4]); // ReflectH
        val = apply_perm(val, &D4_PERMS[4]);
        assert_eq!(val, p);
    }

    #[test]
    fn test_canonicalize_corners() {
        // 四个角位 (0, 2, 5, 7) 的规范形式应相同
        assert_eq!(canonicalize(0b00000001), canonicalize(0b00000100));
        assert_eq!(canonicalize(0b00000001), canonicalize(0b00100000));
        assert_eq!(canonicalize(0b00000001), canonicalize(0b10000000));
        // 规范形式应是最小值
        assert_eq!(canonicalize(0b00000001), 1);
    }

    #[test]
    fn test_canonicalize_edges() {
        // 四个边位 (1, 3, 4, 6) 的规范形式应相同
        assert_eq!(canonicalize(0b00000010), canonicalize(0b00001000));
        assert_eq!(canonicalize(0b00000010), canonicalize(0b00010000));
        assert_eq!(canonicalize(0b00000010), canonicalize(0b01000000));
        // 规范形式应是最小值
        assert_eq!(canonicalize(0b00000010), 2);
    }

    // ---- Hensel 轨道测试 ----

    #[test]
    fn test_hensel_orbit_counts() {
        let orbits = get_hensel_orbits();
        // 各计数的轨道数应与标准一致
        assert_eq!(orbits[0].len(), 1); // 0 邻居：1 个轨道
        assert_eq!(orbits[1].len(), 2); // 1 邻居：2 个轨道 (c, e)
        assert_eq!(orbits[2].len(), 6); // 2 邻居：6 个轨道
        assert_eq!(orbits[3].len(), 10); // 3 邻居：10 个轨道
        assert_eq!(orbits[4].len(), 13); // 4 邻居：13 个轨道
        assert_eq!(orbits[5].len(), 10); // 5 邻居：10 个轨道
        assert_eq!(orbits[6].len(), 6); // 6 邻居：6 个轨道
        assert_eq!(orbits[7].len(), 2); // 7 邻居：2 个轨道
        assert_eq!(orbits[8].len(), 1); // 8 邻居：1 个轨道
    }

    #[test]
    fn test_hensel_orbit_total_patterns() {
        let orbits = get_hensel_orbits();
        let total: usize = orbits
            .iter()
            .flat_map(|v| v.iter())
            .map(|(_, patterns)| patterns.len())
            .sum();
        assert_eq!(total, 256, "Total patterns across all orbits must be 256");
    }

    #[test]
    fn test_hensel_orbit_code_letters() {
        let orbits = get_hensel_orbits();
        // count=1: 代码为 c, e
        assert_eq!(orbits[1][0].0, 'c');
        assert_eq!(orbits[1][1].0, 'e');
        // count=7: 代码为 c, e
        assert_eq!(orbits[7][0].0, 'c');
        assert_eq!(orbits[7][1].0, 'e');
    }

    // ---- Hensel 解析测试 ----

    #[test]
    fn test_hensel_parse_conway() {
        // Conway's Life: B3/S23（全轨道）
        let table = parse_hensel_notation("B3/S23").unwrap();
        // 验证: count=3 的所有模式在 B 部为 true
        let orbits = get_hensel_orbits();
        for (_, patterns) in &orbits[3] {
            for &p in patterns {
                assert!(table[p as usize], "B3: pattern {} should be true", p);
            }
        }
        // 验证: count=2 的所有模式在 S 部为 true
        for (_, patterns) in &orbits[2] {
            for &p in patterns {
                assert!(
                    table[256 + p as usize],
                    "S2: pattern {} should be true",
                    p
                );
            }
        }
        // 验证: count=3 的所有模式在 S 部为 true
        for (_, patterns) in &orbits[3] {
            for &p in patterns {
                assert!(
                    table[256 + p as usize],
                    "S3: pattern {} should be true",
                    p
                );
            }
        }
    }

    #[test]
    fn test_hensel_parse_equals_lifelike() {
        // B3/S23 的 Hensel 解析应与 LifeLike B3/S23 产生相同的演化结果
        let hensel_table = parse_hensel_notation("B3/S23").unwrap();

        // 用 LifeLike LUT 构建对应的 512 项表
        let mut ll_table = [false; 512];
        let birth_lut = [false, false, false, true, false, false, false, false, false];
        let survival_lut = [false, false, true, true, false, false, false, false, false];
        for pattern in 0u8..=255 {
            let count = pattern.count_ones() as usize;
            if birth_lut[count] {
                ll_table[pattern as usize] = true;
            }
            if survival_lut[count] {
                ll_table[256 + pattern as usize] = true;
            }
        }

        // 两种方式产生的表应一致
        assert_eq!(hensel_table, ll_table);
    }

    #[test]
    fn test_hensel_parse_ameyalli() {
        // Ameyalli 规则应能成功解析
        let table =
            parse_hensel_notation("B2ci3ar4krtz5cq6c7ce/S01e2ek3qj4kt5ceayq6cki7c8").unwrap();
        // 验证表非全零
        assert!(table.iter().any(|&b| b));
    }

    #[test]
    fn test_hensel_parse_invalid_format() {
        assert!(parse_hensel_notation("B3S23").is_err()); // 缺少 /
        assert!(parse_hensel_notation("X3/S23").is_err()); // X 不是 B
        assert!(parse_hensel_notation("B3/X23").is_err()); // X 不是 S
    }

    // ---- 查找表解析测试 ----

    #[test]
    fn test_parse_lookup_hex_roundtrip() {
        let mut original = [false; 512];
        for i in 0..512 {
            original[i] = i % 3 == 0;
        }

        // 转为 hex
        let mut hex = String::with_capacity(128);
        for chunk in original.chunks(4) {
            let mut digit = 0u8;
            for (bit, &val) in chunk.iter().enumerate() {
                if val {
                    digit |= 1 << (3 - bit);
                }
            }
            hex.push_str(&format!("{:X}", digit));
        }

        let parsed = parse_lookup_hex(&hex).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_parse_lookup_hex_wrong_length() {
        assert!(parse_lookup_hex("FF00").is_err());
    }

    #[test]
    fn test_parse_lookup_array() {
        let arr: Vec<u8> = (0..512).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
        let table = parse_lookup_array(&arr).unwrap();
        for i in 0..512 {
            assert_eq!(table[i], i % 2 == 0);
        }
    }

    #[test]
    fn test_parse_lookup_array_wrong_length() {
        assert!(parse_lookup_array(&[0, 1]).is_err());
    }

    #[test]
    fn test_parse_lookup_array_invalid_value() {
        let mut arr = vec![0u8; 512];
        arr[100] = 5;
        assert!(parse_lookup_array(&arr).is_err());
    }

    // ---- 预设解析测试 ----

    #[test]
    fn test_preset_1d_all() {
        for name in &["rule30", "rule54", "rule90", "rule110", "rule184"] {
            let result = resolve_preset_1d(name);
            assert!(result.is_ok(), "1D preset '{}' should resolve", name);
            let map = result.unwrap();
            assert!(map.contains_key("rule"), "1D preset '{}' should set rule", name);
        }
    }

    #[test]
    fn test_preset_2d_lifelike() {
        let map = resolve_preset_2d("game_of_life").unwrap();
        assert_eq!(map["rule_type"], json!("lifelike"));
        assert_eq!(map["birth"], json!([3]));
        assert_eq!(map["survival"], json!([2, 3]));
    }

    #[test]
    fn test_preset_2d_wireworld() {
        let map = resolve_preset_2d("wireworld").unwrap();
        assert_eq!(map["rule_type"], json!("wireworld"));
        assert_eq!(map["d_state"], json!(4));
    }

    #[test]
    fn test_preset_2d_cyclic() {
        let map = resolve_preset_2d("cyclic").unwrap();
        assert_eq!(map["rule_type"], json!("cyclic"));
        assert_eq!(map["n_states"], json!(14));
    }

    #[test]
    fn test_preset_2d_ameyalli() {
        let map = resolve_preset_2d("ameyalli").unwrap();
        assert_eq!(map["rule_type"], json!("hensel"));
        assert!(map.contains_key("hensel_notation"));
    }

    #[test]
    fn test_preset_3d_all() {
        for name in &["life_3d", "cyclic_3d", "fredkin_3d"] {
            let result = resolve_preset_3d(name);
            assert!(result.is_ok(), "3D preset '{}' should resolve", name);
        }
    }

    #[test]
    fn test_unknown_preset_error() {
        assert!(resolve_preset_1d("rule999").is_err());
        assert!(resolve_preset_2d("nonexistent").is_err());
        assert!(resolve_preset_3d("unknown").is_err());
    }

    // ---- 预设合并测试 ----

    #[test]
    fn test_apply_preset_merges_rule_params() {
        let mut ext = HashMap::new();
        ext.insert("preset".into(), json!("game_of_life"));
        ext.insert("rows".into(), json!(32));
        ext.insert("cols".into(), json!(32));

        let result = apply_preset_to_extensions(&ext, 2).unwrap();
        assert_eq!(result["rule_type"], json!("lifelike"));
        assert_eq!(result["birth"], json!([3]));
        assert_eq!(result["survival"], json!([2, 3]));
        // 网格参数应保留
        assert_eq!(result["rows"], json!(32));
        assert_eq!(result["cols"], json!(32));
        // preset 键应被移除
        assert!(!result.contains_key("preset"));
    }

    #[test]
    fn test_apply_preset_overrides_rule_params() {
        let mut ext = HashMap::new();
        ext.insert("preset".into(), json!("highlife"));
        ext.insert("birth".into(), json!([3])); // 应被预设覆盖
        ext.insert("rows".into(), json!(16)); // 应保留

        let result = apply_preset_to_extensions(&ext, 2).unwrap();
        // 预设覆盖 birth
        assert_eq!(result["birth"], json!([3, 6]));
        // 网格参数保留
        assert_eq!(result["rows"], json!(16));
    }

    #[test]
    fn test_apply_preset_no_preset_key() {
        let mut ext = HashMap::new();
        ext.insert("rule".into(), json!(30));
        let result = apply_preset_to_extensions(&ext, 1).unwrap();
        assert_eq!(result["rule"], json!(30));
    }
}
