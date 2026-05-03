//! 元数据类型定义模块
//!
//! 定义 metadata.json 的 Rust 数据结构，包含运行元数据、任务元数据、
//! 输出文件记录和汇总统计等类型。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::core::GlobalConfig;

/// 全局配置快照，记录运行时的全局配置状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfigSnapshot {
    /// 输出根目录
    pub output_dir: String,
    /// 默认输出格式
    pub default_format: String,
    /// 并行线程数
    pub num_threads: usize,
    /// 日志级别
    pub log_level: String,
}

impl From<&GlobalConfig> for GlobalConfigSnapshot {
    fn from(config: &GlobalConfig) -> Self {
        let num_threads = config
            .num_threads
            .unwrap_or_else(rayon::current_num_threads);
        GlobalConfigSnapshot {
            output_dir: config.output_dir.clone(),
            default_format: format!("{:?}", config.default_format),
            num_threads,
            log_level: format!("{:?}", config.log_level).to_lowercase(),
        }
    }
}

/// 单个输出文件的记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    /// 文件名（不含目录路径）
    pub filename: String,
    /// 输出格式
    pub format: String,
    /// 分片编号
    pub shard_id: usize,
    /// 派生种子
    pub seed: u64,
    /// 该文件中的帧数
    pub frames: u64,
    /// 文件大小（字节）
    pub file_size: u64,
    /// SHA-256 哈希（可选，若配置启用）
    pub sha256: Option<String>,
}

/// 失败的分片记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedShardRecord {
    /// 分片编号
    pub shard_id: usize,
    /// 派生种子
    pub seed: u64,
    /// 错误信息
    pub error_message: String,
}

/// 运行汇总统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    /// 任务总数
    pub total_tasks: usize,
    /// 总样本数
    pub total_samples: usize,
    /// 总帧数
    pub total_frames: u64,
    /// 总输出字节数
    pub total_bytes: u64,
    /// 总输出文件数
    pub total_files: usize,
    /// 失败分片数
    pub failed_shard_count: usize,
}

/// 单个任务的元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetadata {
    /// 任务名称（来自清单）
    pub task_name: String,
    /// 生成器名称
    pub generator: String,
    /// 任务配置快照
    pub params_snapshot: serde_json::Value,
    /// 基础随机种子
    pub base_seed: u64,
    /// 总样本数量
    pub sample_count: usize,
    /// 分片总数
    pub shard_count: usize,
    /// 实际生成的总帧数
    pub total_frames: u64,
    /// 输出的总物理字节数
    pub total_bytes: u64,
    /// 平均每样本帧数
    pub avg_frames_per_sample: f64,
    /// 输出文件列表及其元信息
    pub output_files: Vec<FileRecord>,
    /// 任务级别耗时（毫秒）
    pub elapsed_ms: u64,
    /// 失败的分片及其错误信息
    pub failed_shards: Vec<FailedShardRecord>,
}

/// 完整运行元数据，对应 metadata.json 文件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    /// StructGen-rs 软件版本
    pub software_version: String,
    /// Cargo 依赖库版本摘要
    pub dependency_versions: HashMap<String, String>,
    /// 运行开始时间（ISO 8601 格式）
    pub start_time: String,
    /// 运行结束时间（ISO 8601 格式）
    pub end_time: String,
    /// 总耗时（毫秒）
    pub elapsed_ms: u64,
    /// 全局配置快照
    pub global_config: GlobalConfigSnapshot,
    /// 各任务的元数据
    pub tasks: Vec<TaskMetadata>,
    /// 汇总统计
    pub summary: RunSummary,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_global_config() -> GlobalConfig {
        GlobalConfig {
            num_threads: Some(4),
            output_dir: "./test_output".into(),
            ..Default::default()
        }
    }

    #[test]
    fn test_global_config_snapshot_from_global_config() {
        let config = make_global_config();
        let snapshot = GlobalConfigSnapshot::from(&config);
        assert_eq!(snapshot.output_dir, "./test_output");
        assert_eq!(snapshot.num_threads, 4);
        assert_eq!(snapshot.default_format, "Parquet");
        assert_eq!(snapshot.log_level, "info");
    }

    #[test]
    fn test_global_config_snapshot_serialization() {
        let snapshot = GlobalConfigSnapshot {
            output_dir: "./output".into(),
            default_format: "Parquet".into(),
            num_threads: 8,
            log_level: "info".into(),
        };
        let json = serde_json::to_string(&snapshot).unwrap();
        let restored: GlobalConfigSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.output_dir, snapshot.output_dir);
        assert_eq!(restored.default_format, snapshot.default_format);
        assert_eq!(restored.num_threads, snapshot.num_threads);
        assert_eq!(restored.log_level, snapshot.log_level);
    }

    #[test]
    fn test_global_config_snapshot_auto_threads() {
        let config = GlobalConfig {
            num_threads: None,
            output_dir: "./output".into(),
            ..Default::default()
        };
        let snapshot = GlobalConfigSnapshot::from(&config);
        assert!(snapshot.num_threads > 0);
    }

    #[test]
    fn test_file_record_serialization_with_sha256() {
        let record = FileRecord {
            filename: "task_01_shard_000.parquet".into(),
            format: "Parquet".into(),
            shard_id: 0,
            seed: 12345,
            frames: 1000,
            file_size: 4096,
            sha256: Some("abcdef1234567890".into()),
        };
        let json = serde_json::to_string(&record).unwrap();
        let restored: FileRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.filename, record.filename);
        assert_eq!(restored.format, record.format);
        assert_eq!(restored.shard_id, record.shard_id);
        assert_eq!(restored.seed, record.seed);
        assert_eq!(restored.frames, record.frames);
        assert_eq!(restored.file_size, record.file_size);
        assert_eq!(restored.sha256, Some("abcdef1234567890".into()));
    }

    #[test]
    fn test_file_record_serialization_without_sha256() {
        let record = FileRecord {
            filename: "task_02_shard_001.bin".into(),
            format: "Binary".into(),
            shard_id: 1,
            seed: 67890,
            frames: 500,
            file_size: 2048,
            sha256: None,
        };
        let json = serde_json::to_string(&record).unwrap();
        let restored: FileRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.sha256, None);
        assert_eq!(restored.filename, "task_02_shard_001.bin");
    }

    #[test]
    fn test_failed_shard_record_serialization() {
        let record = FailedShardRecord {
            shard_id: 3,
            seed: 11111,
            error_message: "生成器初始化失败: 无效参数".into(),
        };
        let json = serde_json::to_string(&record).unwrap();
        let restored: FailedShardRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.shard_id, 3);
        assert_eq!(restored.seed, 11111);
        assert_eq!(restored.error_message, "生成器初始化失败: 无效参数");
    }

    #[test]
    fn test_run_summary_serialization() {
        let summary = RunSummary {
            total_tasks: 2,
            total_samples: 200,
            total_frames: 50000,
            total_bytes: 1024000,
            total_files: 10,
            failed_shard_count: 1,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let restored: RunSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.total_tasks, 2);
        assert_eq!(restored.total_samples, 200);
        assert_eq!(restored.total_frames, 50000);
        assert_eq!(restored.total_bytes, 1024000);
        assert_eq!(restored.total_files, 10);
        assert_eq!(restored.failed_shard_count, 1);
    }

    fn make_file_records() -> Vec<FileRecord> {
        vec![
            FileRecord {
                filename: "task_01_shard_000.parquet".into(),
                format: "Parquet".into(),
                shard_id: 0,
                seed: 100,
                frames: 1000,
                file_size: 4096,
                sha256: Some("hash1".into()),
            },
            FileRecord {
                filename: "task_01_shard_001.parquet".into(),
                format: "Parquet".into(),
                shard_id: 1,
                seed: 200,
                frames: 800,
                file_size: 3200,
                sha256: Some("hash2".into()),
            },
        ]
    }

    fn make_failed_shards() -> Vec<FailedShardRecord> {
        vec![FailedShardRecord {
            shard_id: 5,
            seed: 999,
            error_message: "I/O 错误".into(),
        }]
    }

    #[test]
    fn test_task_metadata_serialization() {
        let task = TaskMetadata {
            task_name: "test_task".into(),
            generator: "ca".into(),
            params_snapshot: json!({"rule": 30, "seq_length": 256}),
            base_seed: 42,
            sample_count: 100,
            shard_count: 2,
            total_frames: 1800,
            total_bytes: 7296,
            avg_frames_per_sample: 18.0,
            output_files: make_file_records(),
            elapsed_ms: 1500,
            failed_shards: make_failed_shards(),
        };
        let json = serde_json::to_string(&task).unwrap();
        let restored: TaskMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.task_name, "test_task");
        assert_eq!(restored.generator, "ca");
        assert_eq!(restored.base_seed, 42);
        assert_eq!(restored.sample_count, 100);
        assert_eq!(restored.shard_count, 2);
        assert_eq!(restored.total_frames, 1800);
        assert_eq!(restored.total_bytes, 7296);
        assert!((restored.avg_frames_per_sample - 18.0).abs() < 1e-10);
        assert_eq!(restored.output_files.len(), 2);
        assert_eq!(restored.elapsed_ms, 1500);
        assert_eq!(restored.failed_shards.len(), 1);
        assert_eq!(
            restored.params_snapshot,
            json!({"rule": 30, "seq_length": 256})
        );
    }

    #[test]
    fn test_task_metadata_empty_failed_shards() {
        let task = TaskMetadata {
            task_name: "perfect_task".into(),
            generator: "lorenz".into(),
            params_snapshot: json!({"sigma": 10.0}),
            base_seed: 7,
            sample_count: 50,
            shard_count: 1,
            total_frames: 5000,
            total_bytes: 20000,
            avg_frames_per_sample: 100.0,
            output_files: vec![FileRecord {
                filename: "perfect.parquet".into(),
                format: "Parquet".into(),
                shard_id: 0,
                seed: 7,
                frames: 5000,
                file_size: 20000,
                sha256: None,
            }],
            elapsed_ms: 3000,
            failed_shards: vec![],
        };
        let json = serde_json::to_string(&task).unwrap();
        let restored: TaskMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.failed_shards.len(), 0);
        assert_eq!(restored.output_files.len(), 1);
    }

    #[test]
    fn test_run_metadata_serialization() {
        let snapshot = GlobalConfigSnapshot {
            output_dir: "./output".into(),
            default_format: "Parquet".into(),
            num_threads: 4,
            log_level: "info".into(),
        };
        let mut dep_versions = HashMap::new();
        dep_versions.insert("serde".into(), "1.0".into());
        dep_versions.insert("rayon".into(), "1.10".into());

        let summary = RunSummary {
            total_tasks: 2,
            total_samples: 200,
            total_frames: 6800,
            total_bytes: 27296,
            total_files: 4,
            failed_shard_count: 1,
        };

        let task = TaskMetadata {
            task_name: "task_a".into(),
            generator: "ca".into(),
            params_snapshot: json!({"rule": 30}),
            base_seed: 42,
            sample_count: 100,
            shard_count: 2,
            total_frames: 1800,
            total_bytes: 7296,
            avg_frames_per_sample: 18.0,
            output_files: make_file_records(),
            elapsed_ms: 1500,
            failed_shards: make_failed_shards(),
        };

        let metadata = RunMetadata {
            software_version: "0.1.0".into(),
            dependency_versions: dep_versions,
            start_time: "2026-05-03T10:00:00Z".into(),
            end_time: "2026-05-03T10:05:00Z".into(),
            elapsed_ms: 300000,
            global_config: snapshot,
            tasks: vec![task],
            summary,
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let restored: RunMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.software_version, "0.1.0");
        assert!(!restored.software_version.is_empty(), "software_version 不应为空");
        assert_eq!(restored.dependency_versions.len(), 2);
        assert_eq!(restored.start_time, "2026-05-03T10:00:00Z");
        assert!(!restored.start_time.is_empty(), "start_time 不应为空");
        assert_eq!(restored.end_time, "2026-05-03T10:05:00Z");
        assert!(!restored.end_time.is_empty(), "end_time 不应为空");
        assert_eq!(restored.elapsed_ms, 300000);
        assert_eq!(restored.global_config.output_dir, "./output");
        assert_eq!(restored.tasks.len(), 1);
        assert_eq!(restored.tasks[0].task_name, "task_a");
        assert_eq!(restored.summary.total_tasks, 2);
        assert_eq!(restored.summary.failed_shard_count, 1);
    }

    #[test]
    fn test_run_metadata_json_contains_key_fields() {
        let metadata = RunMetadata {
            software_version: "0.1.0".into(),
            dependency_versions: HashMap::new(),
            start_time: "2026-01-01T00:00:00Z".into(),
            end_time: "2026-01-01T00:01:00Z".into(),
            elapsed_ms: 60000,
            global_config: GlobalConfigSnapshot {
                output_dir: "./out".into(),
                default_format: "Text".into(),
                num_threads: 1,
                log_level: "debug".into(),
            },
            tasks: vec![],
            summary: RunSummary {
                total_tasks: 0,
                total_samples: 0,
                total_frames: 0,
                total_bytes: 0,
                total_files: 0,
                failed_shard_count: 0,
            },
        };

        let json = serde_json::to_string_pretty(&metadata).unwrap();
        assert!(json.contains("\"software_version\""));
        assert!(json.contains("\"start_time\""));
        assert!(json.contains("\"end_time\""));
        assert!(json.contains("\"global_config\""));
        assert!(json.contains("\"tasks\""));
        assert!(json.contains("\"summary\""));
    }
}
