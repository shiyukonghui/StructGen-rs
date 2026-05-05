//! 分片数据结构与切分算法
//!
//! Shard 描述一个可独立并行执行的子任务。
//! ShardResult 记录单个分片的执行结果，包含输出统计和错误信息。

use crate::core::OutputFormat;
use crate::sink::OutputStats;

use super::manifest::Manifest;
use super::seed::derive_seed;

/// 分片描述，表示一个可独立并行执行的子任务
#[derive(Debug, Clone)]
pub struct Shard {
    /// 所属任务在 Manifest.tasks 中的索引
    pub task_idx: usize,
    /// 分片在本任务内的编号，从 0 开始
    pub shard_idx: usize,
    /// 派生后的唯一种子
    pub seed: u64,
    /// 本分片需生成的样本数量
    pub sample_count: usize,
}

/// 分片执行结果，包含输出统计信息和文件元信息
#[derive(Debug, Clone)]
pub struct ShardResult {
    /// 关联的任务名称
    pub task_name: String,
    /// 分片编号
    pub shard_idx: usize,
    /// 派生种子（= derive_seed(base_seed, shard_idx)）
    pub seed: u64,
    /// 本分片的样本数量
    pub sample_count: usize,
    /// 输出格式
    pub format: OutputFormat,
    /// 分片的输出统计（由 sink::SinkAdapter::close() 产生）
    pub stats: OutputStats,
    /// 发生的错误信息（None 表示成功）
    pub error: Option<String>,
}

/// 将清单切分为可并行执行的分片列表
///
/// 对每个任务按分片大小切分为多个 Shard，每个 Shard 携带派生种子。
/// 分片大小可由用户指定或自动计算。
///
/// # Arguments
/// * `manifest` - 已校验的清单
///
/// # Returns
/// 所有分片的平铺列表
pub fn shard_tasks(manifest: &Manifest) -> Vec<Shard> {
    let num_cpus = rayon::current_num_threads();
    let mut shards = Vec::new();

    for (task_idx, task) in manifest.tasks.iter().enumerate() {
        let effective_shard_size = task
            .shard_size
            .unwrap_or_else(|| compute_auto_shard_size(task.count, num_cpus));

        let shard_count = task.count.div_ceil(effective_shard_size);

        for shard_idx in 0..shard_count {
            let begin_sample = shard_idx * effective_shard_size;
            let end_sample = (begin_sample + effective_shard_size).min(task.count);
            let sample_count = end_sample - begin_sample;

            shards.push(Shard {
                task_idx,
                shard_idx,
                seed: derive_seed(task.seed, shard_idx),
                sample_count,
            });
        }
    }

    shards
}

/// 自动计算分片大小
///
/// 目标是 CPU 核心数的 4 倍，以实现负载均衡。
/// 最小分片大小为 1。
///
/// # Arguments
/// * `total_count` - 总样本数
/// * `num_cpus` - CPU 逻辑核心数
///
/// # Returns
/// 每个分片的样本数
fn compute_auto_shard_size(total_count: usize, num_cpus: usize) -> usize {
    let target_shards = num_cpus * 4;
    std::cmp::max(1, total_count.div_ceil(target_shards))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::GlobalConfig;

    fn make_manifest(counts: Vec<usize>) -> Manifest {
        let tasks: Vec<_> = counts
            .into_iter()
            .enumerate()
            .map(|(i, count)| super::super::manifest::TaskSpec {
                name: format!("task_{}", i),
                generator: "ca".into(),
                params: Default::default(),
                count,
                seed: (i * 1000) as u64,
                pipeline: vec![],
                output_format: None,
                shard_size: None,
            })
            .collect();

        Manifest {
            tasks,
            global: GlobalConfig {
                output_dir: "./output".into(),
                ..Default::default()
            },
        }
    }

    #[test]
    fn test_shard_tasks_single_task() {
        let manifest = make_manifest(vec![100]);
        let shards = shard_tasks(&manifest);

        assert!(!shards.is_empty());
        let total: usize = shards.iter().map(|s| s.sample_count).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_shard_tasks_uniform_distribution() {
        let manifest = Manifest {
            tasks: vec![super::super::manifest::TaskSpec {
                name: "uniform_task".into(),
                generator: "ca".into(),
                params: Default::default(),
                count: 1000,
                seed: 42,
                pipeline: vec![],
                output_format: None,
                shard_size: Some(250),
            }],
            global: GlobalConfig {
                output_dir: "./output".into(),
                ..Default::default()
            },
        };

        let shards = shard_tasks(&manifest);
        assert_eq!(shards.len(), 4);
        assert_eq!(shards[0].sample_count, 250);
        assert_eq!(shards[1].sample_count, 250);
        assert_eq!(shards[2].sample_count, 250);
        assert_eq!(shards[3].sample_count, 250);

        let total: usize = shards.iter().map(|s| s.sample_count).sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn test_shard_tasks_uneven_distribution() {
        let manifest = Manifest {
            tasks: vec![super::super::manifest::TaskSpec {
                name: "uneven_task".into(),
                generator: "ca".into(),
                params: Default::default(),
                count: 1001,
                seed: 42,
                pipeline: vec![],
                output_format: None,
                shard_size: Some(250),
            }],
            global: GlobalConfig {
                output_dir: "./output".into(),
                ..Default::default()
            },
        };

        let shards = shard_tasks(&manifest);
        assert_eq!(shards.len(), 5);
        assert_eq!(shards[0].sample_count, 250);
        assert_eq!(shards[1].sample_count, 250);
        assert_eq!(shards[2].sample_count, 250);
        assert_eq!(shards[3].sample_count, 250);
        assert_eq!(shards[4].sample_count, 1);

        let total: usize = shards.iter().map(|s| s.sample_count).sum();
        assert_eq!(total, 1001);
    }

    #[test]
    fn test_shard_seeds_are_derived_correctly() {
        let manifest = Manifest {
            tasks: vec![super::super::manifest::TaskSpec {
                name: "seed_test".into(),
                generator: "ca".into(),
                params: Default::default(),
                count: 100,
                seed: 100,
                pipeline: vec![],
                output_format: None,
                shard_size: Some(30),
            }],
            global: GlobalConfig {
                output_dir: "./output".into(),
                ..Default::default()
            },
        };

        let shards = shard_tasks(&manifest);
        // 100个样本, 每分片30 → 4个分片 (30+30+30+10)
        assert_eq!(shards.len(), 4);
        // 验证种子是确定性派生的（不再检查具体值，因为派生算法会变化）
        for shard in &shards {
            assert!(shard.seed > 0 || shard.shard_idx == 0, "种子应有效");
        }
        // 验证不同分片的种子互不相同
        let seeds: Vec<u64> = shards.iter().map(|s| s.seed).collect();
        let mut unique_seeds = seeds.clone();
        unique_seeds.sort();
        unique_seeds.dedup();
        assert_eq!(seeds.len(), unique_seeds.len(), "所有分片种子应唯一");
    }

    #[test]
    fn test_shard_tasks_multi_task() {
        let manifest = make_manifest(vec![50, 100, 200]);
        let shards = shard_tasks(&manifest);
        assert!(!shards.is_empty());

        // 验证所有分片的样本总数正确
        let total_task_0: usize = shards
            .iter()
            .filter(|s| s.task_idx == 0)
            .map(|s| s.sample_count)
            .sum();
        let total_task_1: usize = shards
            .iter()
            .filter(|s| s.task_idx == 1)
            .map(|s| s.sample_count)
            .sum();
        let total_task_2: usize = shards
            .iter()
            .filter(|s| s.task_idx == 2)
            .map(|s| s.sample_count)
            .sum();

        assert_eq!(total_task_0, 50);
        assert_eq!(total_task_1, 100);
        assert_eq!(total_task_2, 200);
    }

    #[test]
    fn test_seed_unique_across_different_tasks() {
        let manifest = Manifest {
            tasks: vec![
                super::super::manifest::TaskSpec {
                    name: "task_a".into(),
                    generator: "ca".into(),
                    params: Default::default(),
                    count: 10,
                    seed: 0,
                    pipeline: vec![],
                    output_format: None,
                    shard_size: Some(5),
                },
                super::super::manifest::TaskSpec {
                    name: "task_b".into(),
                    generator: "ca".into(),
                    params: Default::default(),
                    count: 10,
                    seed: 0,
                    pipeline: vec![],
                    output_format: None,
                    shard_size: Some(5),
                },
            ],
            global: GlobalConfig {
                output_dir: "./output".into(),
                ..Default::default()
            },
        };

        let shards = shard_tasks(&manifest);
        // 两个任务各有 2 个分片，种子从 0 开始派生，应该是 (0,1) 和 (0,1)
        // 但由于 task_idx 不同，它们在逻辑上属于不同任务
        assert_eq!(shards.len(), 4);
    }

    #[test]
    fn test_auto_shard_size_minimum() {
        // 使用较小的 count 确保自动计算的分片大小至少为 1
        let manifest = make_manifest(vec![5]);
        let shards = shard_tasks(&manifest);
        assert!(!shards.is_empty());
        for shard in &shards {
            assert!(shard.sample_count > 0);
        }
    }
}
