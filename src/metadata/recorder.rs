//! 元数据记录器，负责收集 ShardResult 并生成 metadata.json
//!
//! 本模块实现 write_metadata() 核心逻辑：
//! 接收 Manifest 和所有分片执行结果，汇总统计信息，
//! 序列化为格式化的 JSON 并写入输出目录。

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::core::{CoreError, CoreResult};
use crate::scheduler::manifest::Manifest;
use crate::scheduler::shard::ShardResult;

use super::types::{
    FailedShardRecord, FileRecord, GlobalConfigSnapshot, RunMetadata, RunSummary, TaskMetadata,
};

/// 收集编译时依赖库版本信息
///
/// 从 Cargo.toml 已知的主要依赖库版本，
/// StructGen-rs 自身版本通过 CARGO_PKG_VERSION 环境变量获取。
fn collect_dep_versions() -> HashMap<String, String> {
    let mut versions = HashMap::new();
    versions.insert("StructGen-rs".to_string(), env!("CARGO_PKG_VERSION").to_string());
    versions.insert("serde".to_string(), "1".to_string());
    versions.insert("serde_json".to_string(), "1".to_string());
    versions.insert("thiserror".to_string(), "1".to_string());
    versions.insert("arrow".to_string(), "53".to_string());
    versions.insert("parquet".to_string(), "53".to_string());
    versions.insert("rayon".to_string(), "1".to_string());
    versions.insert("serde_yaml".to_string(), "0.9".to_string());
    versions.insert("sha2".to_string(), "0.10".to_string());
    versions.insert("tracing".to_string(), "0.1".to_string());
    versions.insert("tracing-subscriber".to_string(), "0.3".to_string());
    versions.insert("chrono".to_string(), "0.4".to_string());
    versions
}

/// 将元数据写入 JSON 文件到输出目录
///
/// # 参数
/// * `output_dir` - 输出根目录（metadata.json 将写入此目录下）
/// * `manifest` - 原始清单，用于记录配置快照
/// * `results` - 所有分片的执行结果
/// * `start_time` - 运行开始时间
/// * `end_time` - 运行结束时间
///
/// # 错误
/// 当 JSON 序列化或文件写入失败时返回 CoreError
pub fn write_metadata(
    output_dir: &Path,
    manifest: &Manifest,
    results: &[ShardResult],
    start_time: chrono::DateTime<chrono::Utc>,
    end_time: chrono::DateTime<chrono::Utc>,
) -> CoreResult<()> {
    // 构建全局配置快照
    let global_snapshot = GlobalConfigSnapshot {
        output_dir: manifest.global.output_dir.clone(),
        default_format: format!("{:?}", manifest.global.default_format),
        num_threads: manifest
            .global
            .num_threads
            .unwrap_or_else(rayon::current_num_threads),
        log_level: format!("{:?}", manifest.global.log_level).to_lowercase(),
    };

    // 按任务名称将 ShardResult 分组并构建 TaskMetadata
    let mut task_metadatas: Vec<TaskMetadata> = Vec::new();

    for task in &manifest.tasks {
        // 收集匹配当前任务名称的所有结果
        let task_results: Vec<&ShardResult> = results
            .iter()
            .filter(|r| r.task_name == task.name)
            .collect();

        let mut output_files: Vec<FileRecord> = Vec::new();
        let mut failed_shards: Vec<FailedShardRecord> = Vec::new();
        let mut total_frames: u64 = 0;
        let mut total_bytes: u64 = 0;

        for result in &task_results {
            total_frames += result.stats.frames_written;
            total_bytes += result.stats.bytes_written;

            if result.error.is_none() {
                // 成功分片 → 构建 FileRecord
                let filename = result
                    .stats
                    .output_path
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();

                output_files.push(FileRecord {
                    filename,
                    format: format!("{:?}", result.format),
                    shard_id: result.shard_idx,
                    seed: result.seed,
                    frames: result.stats.frames_written,
                    file_size: result.stats.bytes_written,
                    sha256: result.stats.file_hash.clone(),
                });
            } else {
                // 失败分片 → 构建 FailedShardRecord
                failed_shards.push(FailedShardRecord {
                    shard_id: result.shard_idx,
                    seed: result.seed,
                    error_message: result.error.clone().unwrap_or_default(),
                });
            }
        }

        let sample_count = task.count;
        let avg_frames_per_sample = if sample_count > 0 {
            total_frames as f64 / sample_count as f64
        } else {
            0.0
        };

        task_metadatas.push(TaskMetadata {
            task_name: task.name.clone(),
            generator: task.generator.clone(),
            params_snapshot: serde_json::to_value(&task.params).unwrap_or(Value::Null),
            base_seed: task.seed,
            sample_count,
            shard_count: task_results.len(),
            total_frames,
            total_bytes,
            avg_frames_per_sample,
            output_files,
            elapsed_ms: 0, // 暂时置 0，后续由 CLI 传入
            failed_shards,
        });
    }

    // 构建 RunSummary 汇总统计
    let total_tasks = task_metadatas.len();
    let total_samples: usize = task_metadatas.iter().map(|t| t.sample_count).sum();
    let total_frames: u64 = task_metadatas.iter().map(|t| t.total_frames).sum();
    let total_bytes: u64 = task_metadatas.iter().map(|t| t.total_bytes).sum();
    let total_files: usize = task_metadatas.iter().map(|t| t.output_files.len()).sum();
    let failed_shard_count: usize = task_metadatas.iter().map(|t| t.failed_shards.len()).sum();

    // 构建完整 RunMetadata
    let run_metadata = RunMetadata {
        software_version: env!("CARGO_PKG_VERSION").to_string(),
        dependency_versions: collect_dep_versions(),
        start_time: start_time.to_rfc3339(),
        end_time: end_time.to_rfc3339(),
        elapsed_ms: (end_time - start_time).num_milliseconds() as u64,
        global_config: global_snapshot,
        tasks: task_metadatas,
        summary: RunSummary {
            total_tasks,
            total_samples,
            total_frames,
            total_bytes,
            total_files,
            failed_shard_count,
        },
    };

    // 序列化为格式化的 JSON 并写入文件
    let json = serde_json::to_string_pretty(&run_metadata)
        .map_err(|e| CoreError::SerializationError(e.to_string()))?;
    let path = output_dir.join("metadata.json");
    std::fs::write(&path, &json)?;
    tracing::info!("元数据已写入: {}", path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use chrono::Utc;
    use tempfile::TempDir;

    use crate::core::params::{GlobalConfig, OutputFormat};
    use crate::scheduler::manifest::TaskSpec;
    use crate::sink::adapter::OutputStats;

    /// 创建包含 2 个任务的模拟清单
    fn make_mock_manifest() -> Manifest {
        Manifest {
            tasks: vec![
                TaskSpec {
                    name: "task_a".to_string(),
                    generator: "ca".to_string(),
                    params: Default::default(),
                    count: 10,
                    seed: 100,
                    pipeline: vec![],
                    output_format: None,
                    shard_size: None,
                },
                TaskSpec {
                    name: "task_b".to_string(),
                    generator: "lorenz".to_string(),
                    params: Default::default(),
                    count: 20,
                    seed: 200,
                    pipeline: vec![],
                    output_format: None,
                    shard_size: None,
                },
            ],
            global: GlobalConfig {
                output_dir: "./test_output".to_string(),
                ..Default::default()
            },
        }
    }

    /// 创建模拟分片结果
    /// task_a: 2 个成功分片，task_b: 1 个成功分片
    fn make_mock_results() -> Vec<ShardResult> {
        vec![
            ShardResult {
                task_name: "task_a".to_string(),
                shard_idx: 0,
                seed: 100,
                sample_count: 5,
                format: OutputFormat::Parquet,
                stats: OutputStats {
                    frames_written: 50,
                    bytes_written: 1000,
                    output_path: Some(PathBuf::from("/tmp/task_a_00000_0000000000000064.parquet")),
                    file_hash: Some("aaaa1111".to_string()),
                },
                error: None,
            },
            ShardResult {
                task_name: "task_a".to_string(),
                shard_idx: 1,
                seed: 101,
                sample_count: 5,
                format: OutputFormat::Parquet,
                stats: OutputStats {
                    frames_written: 50,
                    bytes_written: 1000,
                    output_path: Some(PathBuf::from("/tmp/task_a_00001_0000000000000065.parquet")),
                    file_hash: Some("bbbb2222".to_string()),
                },
                error: None,
            },
            ShardResult {
                task_name: "task_b".to_string(),
                shard_idx: 0,
                seed: 200,
                sample_count: 20,
                format: OutputFormat::Text,
                stats: OutputStats {
                    frames_written: 200,
                    bytes_written: 4000,
                    output_path: Some(PathBuf::from("/tmp/task_b_00000_00000000000000c8.txt")),
                    file_hash: Some("cccc3333".to_string()),
                },
                error: None,
            },
        ]
    }

    #[test]
    fn test_write_metadata_creates_valid_json() {
        let tmp = TempDir::new().unwrap();
        let manifest = make_mock_manifest();
        let results = make_mock_results();
        let now = Utc::now();

        write_metadata(tmp.path(), &manifest, &results, now, now).unwrap();

        let content = std::fs::read_to_string(tmp.path().join("metadata.json")).unwrap();
        let parsed: RunMetadata = serde_json::from_str(&content).unwrap();

        assert_eq!(parsed.tasks.len(), 2);
        assert!(!parsed.software_version.is_empty());
    }

    #[test]
    fn test_write_metadata_records_failed_shards() {
        let tmp = TempDir::new().unwrap();
        let manifest = make_mock_manifest();
        let mut results = make_mock_results();
        // 标记第一个分片为失败
        results[0].error = Some("模拟生成错误".to_string());

        let now = Utc::now();
        write_metadata(tmp.path(), &manifest, &results, now, now).unwrap();

        let content = std::fs::read_to_string(tmp.path().join("metadata.json")).unwrap();
        let parsed: RunMetadata = serde_json::from_str(&content).unwrap();

        assert_eq!(parsed.summary.failed_shard_count, 1);

        // 验证 task_a 有失败分片记录
        let task_a = parsed
            .tasks
            .iter()
            .find(|t| t.task_name == "task_a")
            .unwrap();
        assert!(!task_a.failed_shards.is_empty());
        assert_eq!(
            task_a.failed_shards[0].error_message,
            "模拟生成错误"
        );
    }

    #[test]
    fn test_write_metadata_empty_results() {
        let tmp = TempDir::new().unwrap();
        let manifest = make_mock_manifest();
        let results: Vec<ShardResult> = vec![];
        let now = Utc::now();

        // 空结果不应 panic
        write_metadata(tmp.path(), &manifest, &results, now, now).unwrap();

        let content = std::fs::read_to_string(tmp.path().join("metadata.json")).unwrap();
        let parsed: RunMetadata = serde_json::from_str(&content).unwrap();

        // 汇总各项应为 0
        assert_eq!(parsed.summary.total_frames, 0);
        assert_eq!(parsed.summary.total_bytes, 0);
        assert_eq!(parsed.summary.total_files, 0);
        assert_eq!(parsed.summary.failed_shard_count, 0);
    }
}
