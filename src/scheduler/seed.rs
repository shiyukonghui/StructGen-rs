//! 确定性的种子派生算法
//!
//! 将用户指定的基础种子与分片编号/样本编号组合，产生唯一种子。
//! 使用 SplitMix64 风格的混合函数，确保良好的雪崩效应和均匀分布。

/// SplitMix64 风格的混合函数
///
/// 对输入值进行多轮移位和乘法混合，产生良好的雪崩效应：
/// 输入的微小变化会导致输出的巨大变化。
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// 根据基础种子和分片序号派生唯一种子
///
/// 使用 SplitMix64 风格的混合函数，将基础种子与分片序号组合，
/// 确保不同分片之间的种子具有良好分布和雪崩效应。
///
/// # Arguments
/// * `base_seed` - 用户在清单中为每个任务指定的基础种子
/// * `shard_idx` - 分片编号，从 0 开始
///
/// # Returns
/// 派生后的唯一种子
pub fn derive_seed(base_seed: u64, shard_idx: usize) -> u64 {
    // 使用黄金比例常量将 base_seed 和 shard_idx 混合
    let z = base_seed ^ ((shard_idx as u64).wrapping_mul(0x9E3779B97F4A7C15));
    mix(z)
}

/// 根据分片种子和样本序号派生唯一种子
///
/// 在 executor 的多样本循环中，为每个样本派生独立的种子。
/// 使用与 `derive_seed` 不同的混合常量，确保两个命名空间
/// 不会产生碰撞。
///
/// # Arguments
/// * `shard_seed` - 分片的派生种子（由 `derive_seed` 产生）
/// * `sample_idx` - 样本编号，从 0 开始
///
/// # Returns
/// 派生后的唯一种子
pub fn derive_sample_seed(shard_seed: u64, sample_idx: usize) -> u64 {
    // 使用不同于 derive_seed 的常量，保证命名空间独立
    let z = shard_seed.wrapping_add((sample_idx as u64).wrapping_mul(0x517CC1B727220A95));
    mix(z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_seed_deterministic() {
        // 同样的输入必须产生同样的输出
        for base in [0u64, 42, 0xDEAD_BEEF, u64::MAX] {
            for shard in 0usize..10 {
                assert_eq!(derive_seed(base, shard), derive_seed(base, shard));
            }
        }
    }

    #[test]
    fn test_derive_seed_different_shards_unique() {
        let seeds: Vec<u64> = (0..100).map(|i| derive_seed(1000, i)).collect();
        let mut unique = seeds.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(seeds.len(), unique.len(), "所有分片种子应唯一");
    }

    #[test]
    fn test_derive_seed_avalanche() {
        // 相邻的 shard_idx 应产生差异巨大的种子
        let s0 = derive_seed(42, 0);
        let s1 = derive_seed(42, 1);
        // 汉明距离应该很大（至少 16 位不同）
        let diff = s0 ^ s1;
        let hamming = diff.count_ones();
        assert!(hamming >= 16, "相邻种子的汉明距离应 >= 16，实际为 {}", hamming);
    }

    #[test]
    fn test_derive_sample_seed_deterministic() {
        for seed in [0u64, 42, u64::MAX] {
            for idx in 0usize..10 {
                assert_eq!(
                    derive_sample_seed(seed, idx),
                    derive_sample_seed(seed, idx)
                );
            }
        }
    }

    #[test]
    fn test_derive_sample_seed_different_indices_unique() {
        let seeds: Vec<u64> = (0..1000).map(|i| derive_sample_seed(42, i)).collect();
        let mut unique = seeds.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(seeds.len(), unique.len(), "所有样本种子应唯一");
    }

    #[test]
    fn test_derive_sample_seed_namespace_independent() {
        // derive_sample_seed(shard_seed, sample_idx) 应与 derive_seed(base_seed, shard_idx)
        // 在实际使用场景中不冲突。模拟真实场景：base_seed=42, shard_idx=0..5, sample_idx=0..10
        let base_seed = 42u64;
        let mut shard_seeds = Vec::new();
        let mut sample_seeds = Vec::new();

        for shard_idx in 0..5 {
            let shard_seed = derive_seed(base_seed, shard_idx);
            shard_seeds.push(shard_seed);
            for sample_idx in 0..10 {
                sample_seeds.push(derive_sample_seed(shard_seed, sample_idx));
            }
        }

        // 所有 shard 级种子应互不相同
        let mut unique_shard = shard_seeds.clone();
        unique_shard.sort();
        unique_shard.dedup();
        assert_eq!(shard_seeds.len(), unique_shard.len(), "shard 种子应唯一");

        // 所有 sample 级种子应互不相同
        let mut unique_sample = sample_seeds.clone();
        unique_sample.sort();
        unique_sample.dedup();
        assert_eq!(sample_seeds.len(), unique_sample.len(), "sample 种子应唯一");

        // shard 级种子不应出现在 sample 级种子中（两个命名空间无交叉）
        for shard_seed in &shard_seeds {
            assert!(
                !sample_seeds.contains(shard_seed),
                "shard 种子 {} 不应出现在 sample 种子中", shard_seed
            );
        }
    }

    #[test]
    fn test_derive_sample_seed_avalanche() {
        let s0 = derive_sample_seed(42, 0);
        let s1 = derive_sample_seed(42, 1);
        let diff = s0 ^ s1;
        let hamming = diff.count_ones();
        assert!(hamming >= 16, "相邻样本种子的汉明距离应 >= 16，实际为 {}", hamming);
    }
}
