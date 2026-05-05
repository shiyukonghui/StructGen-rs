use serde::Deserialize;

/// 元胞自动机边界条件类型
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Boundary {
    /// 周期边界：越界环绕（环面拓扑）
    #[default]
    Periodic,
    /// 固定边界：越界视为状态 0
    Fixed,
    /// 反射边界：越界镜像反射
    Reflective,
}

/// 邻域类型
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Neighborhood {
    /// 全邻接：2D=8邻域, 3D=26邻域
    #[default]
    Moore,
    /// 轴邻接：2D=4邻域, 3D=6邻域
    VonNeumann,
}

/// 初始化模式
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum InitMode {
    /// PRNG 均匀随机填充
    #[default]
    Random,
    /// 全零，仅中心格子为状态 1
    SingleCenter,
}

/// 2D 规则系统
#[derive(Clone)]
pub enum Rule2D {
    /// B/S 记法：birth/survival 包含触发转移的邻居计数
    /// 仅适用于 d_state == 2
    LifeLike {
        birth: Vec<u8>,
        survival: Vec<u8>,
    },
    /// 邻居状态总和 → 下一状态
    /// 适用于任意 d_state >= 2
    Totalistic {
        transition_table: Vec<u8>,
    },
}

/// 3D 规则系统
#[derive(Clone)]
pub enum Rule3D {
    /// B/S 记法（3D）
    LifeLike3D {
        birth: Vec<u8>,
        survival: Vec<u8>,
    },
    /// 邻居状态总和 → 下一状态（3D）
    Totalistic3D {
        transition_table: Vec<u8>,
    },
}

/// LifeLike 规则编译后的查找表
#[derive(Clone)]
pub struct LifeLikeLUT {
    /// birth_lut[count] = true 表示死格在 count 个活邻居时出生
    pub birth_lut: Vec<bool>,
    /// survival_lut[count] = true 表示活格在 count 个活邻居时存活
    pub survival_lut: Vec<bool>,
}

impl LifeLikeLUT {
    /// 从 birth/survival 列表构建查找表
    ///
    /// `max_neighbors` 为当前邻域的最大邻居数：
    /// - 2D Moore=8, 2D VonNeumann=4
    /// - 3D Moore=26, 3D VonNeumann=6
    pub fn from_birth_survival(birth: &[u8], survival: &[u8], max_neighbors: usize) -> Self {
        let mut birth_lut = vec![false; max_neighbors + 1];
        let mut survival_lut = vec![false; max_neighbors + 1];
        for &b in birth {
            if (b as usize) <= max_neighbors {
                birth_lut[b as usize] = true;
            }
        }
        for &s in survival {
            if (s as usize) <= max_neighbors {
                survival_lut[s as usize] = true;
            }
        }
        LifeLikeLUT {
            birth_lut,
            survival_lut,
        }
    }
}

// ---- 邻域偏移常量 ----

/// 2D Moore 邻域偏移（8 个）
pub const MOORE_2D_OFFSETS: [(i32, i32); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

/// 2D VonNeumann 邻域偏移（4 个）
pub const VONNEUMANN_2D_OFFSETS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// 3D Moore 邻域偏移（26 个，排除 (0,0,0)）
pub const MOORE_3D_OFFSETS: [(i32, i32, i32); 26] = [
    (-1, -1, -1),
    (-1, -1, 0),
    (-1, -1, 1),
    (-1, 0, -1),
    (-1, 0, 0),
    (-1, 0, 1),
    (-1, 1, -1),
    (-1, 1, 0),
    (-1, 1, 1),
    (0, -1, -1),
    (0, -1, 0),
    (0, -1, 1),
    (0, 0, -1),
    (0, 0, 1),
    (0, 1, -1),
    (0, 1, 0),
    (0, 1, 1),
    (1, -1, -1),
    (1, -1, 0),
    (1, -1, 1),
    (1, 0, -1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, -1),
    (1, 1, 0),
    (1, 1, 1),
];

/// 3D VonNeumann 邻域偏移（6 个）
pub const VONNEUMANN_3D_OFFSETS: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// 周期环绕索引计算
#[inline]
pub fn wrap_index(coord: i32, offset: i32, size: usize) -> usize {
    let shifted = coord + offset;
    let wrapped = ((shifted % size as i32) + size as i32) % size as i32;
    wrapped as usize
}

/// 边界感知的 2D 邻居读取
#[inline]
pub fn get_neighbor_2d(
    grid: &[u8],
    r: usize,
    c: usize,
    rows: usize,
    cols: usize,
    dr: i32,
    dc: i32,
    boundary: &Boundary,
) -> u8 {
    let nr = r as i32 + dr;
    let nc = c as i32 + dc;

    match boundary {
        Boundary::Periodic => {
            let wr = wrap_index(r as i32, dr, rows);
            let wc = wrap_index(c as i32, dc, cols);
            grid[wr * cols + wc]
        }
        Boundary::Fixed => {
            if nr < 0 || nr >= rows as i32 || nc < 0 || nc >= cols as i32 {
                0
            } else {
                grid[nr as usize * cols + nc as usize]
            }
        }
        Boundary::Reflective => {
            let wr = reflect_index(nr, rows);
            let wc = reflect_index(nc, cols);
            grid[wr * cols + wc]
        }
    }
}

/// 边界感知的 3D 邻居读取
#[inline]
pub fn get_neighbor_3d(
    grid: &[u8],
    d: usize,
    r: usize,
    c: usize,
    depth: usize,
    rows: usize,
    cols: usize,
    dd: i32,
    dr: i32,
    dc: i32,
    boundary: &Boundary,
) -> u8 {
    let nd = d as i32 + dd;
    let nr = r as i32 + dr;
    let nc = c as i32 + dc;

    match boundary {
        Boundary::Periodic => {
            let wd = wrap_index(d as i32, dd, depth);
            let wr = wrap_index(r as i32, dr, rows);
            let wc = wrap_index(c as i32, dc, cols);
            grid[wd * (rows * cols) + wr * cols + wc]
        }
        Boundary::Fixed => {
            if nd < 0
                || nd >= depth as i32
                || nr < 0
                || nr >= rows as i32
                || nc < 0
                || nc >= cols as i32
            {
                0
            } else {
                grid[nd as usize * (rows * cols) + nr as usize * cols + nc as usize]
            }
        }
        Boundary::Reflective => {
            let wd = reflect_index(nd, depth);
            let wr = reflect_index(nr, rows);
            let wc = reflect_index(nc, cols);
            grid[wd * (rows * cols) + wr * cols + wc]
        }
    }
}

/// 反射索引：越界时取边界值
#[inline]
fn reflect_index(coord: i32, size: usize) -> usize {
    if coord < 0 {
        0
    } else if coord >= size as i32 {
        size - 1
    } else {
        coord as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_boundary_deserialization() {
        let b: Boundary = serde_json::from_value(serde_json::json!("periodic")).unwrap();
        assert_eq!(b, Boundary::Periodic);
        let b: Boundary = serde_json::from_value(serde_json::json!("fixed")).unwrap();
        assert_eq!(b, Boundary::Fixed);
        let b: Boundary = serde_json::from_value(serde_json::json!("reflective")).unwrap();
        assert_eq!(b, Boundary::Reflective);
    }

    #[test]
    fn test_boundary_default() {
        assert_eq!(Boundary::default(), Boundary::Periodic);
    }

    #[test]
    fn test_neighborhood_deserialization() {
        let n: Neighborhood = serde_json::from_value(serde_json::json!("moore")).unwrap();
        assert_eq!(n, Neighborhood::Moore);
        let n: Neighborhood = serde_json::from_value(serde_json::json!("vonneumann")).unwrap();
        assert_eq!(n, Neighborhood::VonNeumann);
    }

    #[test]
    fn test_neighborhood_default() {
        assert_eq!(Neighborhood::default(), Neighborhood::Moore);
    }

    #[test]
    fn test_init_mode_deserialization() {
        let m: InitMode = serde_json::from_value(serde_json::json!("random")).unwrap();
        assert_eq!(m, InitMode::Random);
        let m: InitMode = serde_json::from_value(serde_json::json!("singlecenter")).unwrap();
        assert_eq!(m, InitMode::SingleCenter);
    }

    #[test]
    fn test_init_mode_default() {
        assert_eq!(InitMode::default(), InitMode::Random);
    }

    #[test]
    fn test_lifelike_lut_construction() {
        // B3/S23 (Conway's Game of Life)
        let lut = LifeLikeLUT::from_birth_survival(&[3], &[2, 3], 8);
        assert_eq!(lut.birth_lut.len(), 9);
        assert_eq!(lut.survival_lut.len(), 9);
        assert!(lut.birth_lut[3]);
        assert!(!lut.birth_lut[2]);
        assert!(lut.survival_lut[2]);
        assert!(lut.survival_lut[3]);
        assert!(!lut.survival_lut[1]);
    }

    #[test]
    fn test_lifelike_lut_ignores_out_of_range() {
        // 值超过 max_neighbors 应被忽略（不会越界设置）
        let lut = LifeLikeLUT::from_birth_survival(&[3, 9], &[2, 3], 8);
        assert!(lut.birth_lut[3]);
        // 9 超过 max_neighbors=8，不会被设置；birth_lut 长度为 9（索引 0..=8）
        assert_eq!(lut.birth_lut.len(), 9);
        // 验证索引 8 未被设置为 true（因为传入的是 9 不是 8）
        assert!(!lut.birth_lut[8]);
    }

    #[test]
    fn test_moore_2d_offsets() {
        assert_eq!(MOORE_2D_OFFSETS.len(), 8);
        // 不包含 (0,0)
        assert!(!MOORE_2D_OFFSETS.iter().any(|&(dr, dc)| dr == 0 && dc == 0));
    }

    #[test]
    fn test_vonneumann_2d_offsets() {
        assert_eq!(VONNEUMANN_2D_OFFSETS.len(), 4);
        assert!(!VONNEUMANN_2D_OFFSETS.iter().any(|&(dr, dc)| dr == 0 && dc == 0));
    }

    #[test]
    fn test_moore_3d_offsets() {
        assert_eq!(MOORE_3D_OFFSETS.len(), 26);
        assert!(!MOORE_3D_OFFSETS.iter().any(|&(dd, dr, dc)| dd == 0 && dr == 0 && dc == 0));
    }

    #[test]
    fn test_vonneumann_3d_offsets() {
        assert_eq!(VONNEUMANN_3D_OFFSETS.len(), 6);
        assert!(!VONNEUMANN_3D_OFFSETS.iter().any(|&(dd, dr, dc)| dd == 0 && dr == 0 && dc == 0));
    }

    #[test]
    fn test_wrap_index() {
        // 正常情况
        assert_eq!(wrap_index(2, 1, 5), 3);
        // 越界左
        assert_eq!(wrap_index(0, -1, 5), 4);
        // 越界右
        assert_eq!(wrap_index(4, 1, 5), 0);
        // 大偏移
        assert_eq!(wrap_index(0, -6, 5), 4);
    }

    #[test]
    fn test_get_neighbor_2d_periodic() {
        // 3×3 网格
        let grid: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        // (0,0) 的左上邻居 → 周期环绕到 (2,2)
        assert_eq!(
            get_neighbor_2d(&grid, 0, 0, 3, 3, -1, -1, &Boundary::Periodic),
            9
        );
        // (0,0) 的右下邻居 → (1,1)
        assert_eq!(
            get_neighbor_2d(&grid, 0, 0, 3, 3, 1, 1, &Boundary::Periodic),
            5
        );
    }

    #[test]
    fn test_get_neighbor_2d_fixed() {
        let grid: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        // (0,0) 的左上邻居 → 越界，返回 0
        assert_eq!(
            get_neighbor_2d(&grid, 0, 0, 3, 3, -1, -1, &Boundary::Fixed),
            0
        );
        // (1,1) 的右下邻居 → (2,2) = 9
        assert_eq!(
            get_neighbor_2d(&grid, 1, 1, 3, 3, 1, 1, &Boundary::Fixed),
            9
        );
    }

    #[test]
    fn test_get_neighbor_2d_reflective() {
        let grid: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        // (0,0) 的左上邻居 → 反射到 (0,0) = 1
        assert_eq!(
            get_neighbor_2d(&grid, 0, 0, 3, 3, -1, -1, &Boundary::Reflective),
            1
        );
        // (2,2) 的右下邻居 → 反射到 (2,2) = 9
        assert_eq!(
            get_neighbor_2d(&grid, 2, 2, 3, 3, 1, 1, &Boundary::Reflective),
            9
        );
    }

    #[test]
    fn test_get_neighbor_3d_periodic() {
        // 2×2×2 网格
        let grid: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        // (0,0,0) 的 (-1,-1,-1) 邻居 → 周期环绕到 (1,1,1) = 8
        assert_eq!(
            get_neighbor_3d(&grid, 0, 0, 0, 2, 2, 2, -1, -1, -1, &Boundary::Periodic),
            8
        );
        // (0,0,0) 的 (1,0,0) 邻居 → (1,0,0) = 5
        assert_eq!(
            get_neighbor_3d(&grid, 0, 0, 0, 2, 2, 2, 1, 0, 0, &Boundary::Periodic),
            5
        );
    }
}
