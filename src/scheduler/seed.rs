//! 确定性的种子派生算法
//!
//! 将用户指定的基础种子与分片编号组合，产生每个分片的唯一种子。
//! 使用 wrapping_add 确保不同分片、不同任务之间的种子无冲突。

/// 根据基础种子和分片序号派生唯一种子
///
/// 派生规则：`seed = base_seed.wrapping_add(shard_idx as u64)`
/// 简单的包装加法保证不同分片之间的种子无冲突。
///
/// # Arguments
/// * `base_seed` - 用户在清单中为每个任务指定的基础种子
/// * `shard_idx` - 分片编号，从 0 开始
///
/// # Returns
/// 派生后的唯一种子
///
/// # Examples
/// ```
/// use scheduler::seed::derive_seed;
/// assert_eq!(derive_seed(42, 0), 42);
/// assert_eq!(derive_seed(42, 1), 43);
/// assert_eq!(derive_seed(u64::MAX, 1), 0); // wrapping
/// ```
pub fn derive_seed(base_seed: u64, shard_idx: usize) -> u64 {
    base_seed.wrapping_add(shard_idx as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_seed_deterministic() {
        assert_eq!(derive_seed(42, 0), 42);
        assert_eq!(derive_seed(42, 1), 43);
        assert_eq!(derive_seed(0, 0), 0);
        assert_eq!(derive_seed(0, 5), 5);
    }

    #[test]
    fn test_derive_seed_wrapping() {
        assert_eq!(derive_seed(u64::MAX, 1), 0);
        assert_eq!(derive_seed(u64::MAX, 2), 1);
        assert_eq!(derive_seed(u64::MAX - 1, 3), 1);
    }

    #[test]
    fn test_derive_seed_idempotent() {
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
}
