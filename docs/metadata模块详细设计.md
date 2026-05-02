# StructGen-rs 元数据与监控层 (metadata) 详细设计规格

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| 1.0 | 2026-05-01 | - | 初始版本，定义元数据记录、日志服务和进度上报逻辑 |

## 1. 模块概述

元数据与监控层（metadata）是 StructGen-rs 系统的可追溯性与可观测性基础。它负责记录每次生成运行的完整参数快照、数据统计、完整性校验信息以及实时进度，确保生成的数据集可以完整复现和审计。

该模块的职责包括：
- 在每次运行结束后，生成一个独立的 JSON 格式元数据文件（`metadata.json`），记录任务配置、输出文件清单、哈希校验和数据统计。
- 管理结构化日志输出（控制台/文件/同时），记录生成进度、耗时、错误信息和警告。
- 提供实时进度上报接口，供调度器和 CLI 层在长时间运行中更新进度条。
- 记录软件版本信息和依赖库版本摘要，确保环境可复现。

**核心原则**：元数据层是一个"观察者"而非"参与者"——它不修改数据流，只记录和报告。它与数据生成管线松耦合，通过接收来自调度器的统计信息来构建完整的运行记录。

## 2. 设计目标与原则

- **完整可复现**：元数据记录必须包含完整的信息集（配置、种子、版本、哈希），确保任意第三方可以在相同环境下精确重构数据集。
- **非侵入性**：元数据收集不应影响生成管线的性能。统计信息的计算尽量利用已有的中间结果（如 ShardResult 中的统计），而非重复扫描。
- **结构化**：所有元数据以 JSON 格式存储，便于程序化查询、索引和跨语言消费。
- **日志分级**：支持 Trace / Debug / Info / Warn / Error 五个级别，可由用户通过 `--log-level` 控制。
- **实时进度**：在长时间运行中，通过日志或进度回调通知用户当前完成百分比和预计剩余时间。

## 3. 模块内部结构组织

```
src/metadata/
├── mod.rs        # 模块根，暴露 write_metadata()、init_logger()
├── types.rs      # RunMetadata、TaskMetadata、FileRecord 等类型定义
├── recorder.rs   # write_metadata() 逻辑：收集 ShardResult 并写入 JSON
├── logger.rs     # 日志初始化与配置（基于 tracing/log crate）
└── progress.rs   # ProgressTracker：实时进度计算与上报
```

| 文件 | 职责 |
|------|------|
| `types.rs` | 定义元数据 JSON 结构的 Rust 类型（带 serde 序列化） |
| `recorder.rs` | 实现 `write_metadata()`：接收 Manifest + Vec\<ShardResult\>，汇总并写入 JSON |
| `logger.rs` | 封装 `tracing-subscriber`，支持控制台和文件两种输出 |
| `progress.rs` | `ProgressTracker` 结构体，接收事件并计算进度百分比和 ETA |
| `mod.rs` | 重导出所有公开类型，暴露 `write_metadata()` 和 `init_logger()` |

## 4. 公开接口定义

### 4.1 元数据类型

```rust
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use crate::sink::OutputStats;    // OutputStats 定义在 sink 模块

/// 完整运行元数据，对应 metadata.json 文件。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    /// StructGen-rs 软件版本。
    pub software_version: String,
    /// Cargo 依赖库版本摘要（主要库名 → 版本）。
    pub dependency_versions: HashMap<String, String>,
    /// 运行开始时间（ISO 8601 格式）。
    pub start_time: String,
    /// 运行结束时间（ISO 8601 格式）。
    pub end_time: String,
    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
    /// 全局配置快照。
    pub global_config: GlobalConfigSnapshot,
    /// 各任务的元数据。
    pub tasks: Vec<TaskMetadata>,
    /// 汇总统计。
    pub summary: RunSummary,
}

/// 单个任务的元数据。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetadata {
    /// 任务名称（来自清单）。
    pub task_name: String,
    /// 生成器名称。
    pub generator: String,
    /// 任务配置快照（JSON 字符串，保留原始参数形态）。
    pub params_snapshot: serde_json::Value,
    /// 基础随机种子。
    pub base_seed: u64,
    /// 总样本数量（清单中指定的）。
    pub sample_count: usize,
    /// 分片总数。
    pub shard_count: usize,
    /// 实际生成的总帧数。
    pub total_frames: u64,
    /// 输出的总物理字节数。
    pub total_bytes: u64,
    /// 平均每样本帧数（= total_frames / sample_count）。
    pub avg_frames_per_sample: f64,
    /// 输出文件列表及其元信息。
    pub output_files: Vec<FileRecord>,
    /// 任务级别耗时（毫秒）。
    pub elapsed_ms: u64,
    /// 失败的分片及其错误信息。
    pub failed_shards: Vec<FailedShardRecord>,
}

/// 单个输出文件的记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    /// 文件名（不含目录路径）。
    pub filename: String,
    /// 输出格式。
    pub format: String,
    /// 分片编号。
    pub shard_id: usize,
    /// 派生种子。
    pub seed: u64,
    /// 该文件中的帧数。
    pub frames: u64,
    /// 文件大小（字节）。
    pub file_size: u64,
    /// SHA-256 哈希（可选，若配置启用）。
    pub sha256: Option<String>,
}

/// 失败的分片记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedShardRecord {
    pub shard_id: usize,
    pub seed: u64,
    pub error_message: String,
}

/// 全局配置快照。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfigSnapshot {
    pub output_dir: String,
    pub default_format: String,
    pub num_threads: usize,
    pub log_level: String,
}

/// 运行汇总。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub total_tasks: usize,
    pub total_samples: usize,
    pub total_frames: u64,
    pub total_bytes: u64,
    pub total_files: usize,
    pub failed_shard_count: usize,
}
```

### 4.2 公开函数

```rust
use crate::core::CoreResult;
use crate::scheduler::{Manifest, ShardResult};
use crate::sink::OutputStats;    // OutputStats 定义在 sink 模块中
use std::path::Path;

/// 写入元数据 JSON 文件到输出目录。
///
/// # Arguments
/// * `output_dir` - 输出根目录（metadata.json 将写入此目录下）。
/// * `manifest` - 原始清单（用于记录配置快照）。
/// * `results` - 所有分片的执行结果。
/// * `start_time` - 运行开始时间。
/// * `end_time` - 运行结束时间。
///
/// # Errors
/// 当文件写入失败时返回 CoreError。
pub fn write_metadata(
    output_dir: &Path,
    manifest: &Manifest,
    results: &[ShardResult],
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
) -> CoreResult<()> { /* ... */ }

/// 初始化日志系统。
///
/// # Arguments
/// * `log_level` - 日志级别字符串（"trace"/"debug"/"info"/"warn"/"error"）。
/// * `log_file` - 若为 Some(path)，同时将日志写入该文件。
///
/// # Returns
/// 成功初始化后返回 Ok(())。
///
/// # Panics
/// 若日志系统已被初始化（通常仅在 main 中调用一次）。
pub fn init_logger(log_level: &str, log_file: Option<&Path>) -> CoreResult<()> { /* ... */ }
```

### 4.3 ProgressTracker

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// 实时进度追踪器，线程安全。
///
/// 调度器在每个分片完成时调用 `report_completed(count)`，
/// CLI 层定期调用 `progress()` 获取当前进度用于显示。
#[derive(Clone)]
pub struct ProgressTracker {
    total_samples: u64,
    completed_samples: Arc<AtomicU64>,
    total_frames: Arc<AtomicU64>,
    start_instant: std::time::Instant,
}

impl ProgressTracker {
    /// 创建新的进度追踪器。
    ///
    /// # Arguments
    /// * `total_samples` - 全部样本总数（所有分片之和）。
    pub fn new(total_samples: usize) -> Self { /* ... */ }

    /// 报告一批样本已完成生成。
    pub fn report_completed(&self, samples: usize, frames: u64) {
        self.completed_samples.fetch_add(samples as u64, Ordering::Relaxed);
        self.total_frames.fetch_add(frames, Ordering::Relaxed);
    }

    /// 获取当前进度信息。
    pub fn progress(&self) -> ProgressInfo {
        let completed = self.completed_samples.load(Ordering::Relaxed);
        let pct = if self.total_samples > 0 {
            (completed as f64 / self.total_samples as f64) * 100.0
        } else { 0.0 };
        let elapsed = self.start_instant.elapsed();
        let eta = if completed > 0 {
            let rate = completed as f64 / elapsed.as_secs_f64();
            let remaining = self.total_samples - completed;
            std::time::Duration::from_secs_f64(remaining as f64 / rate)
        } else {
            std::time::Duration::from_secs(0)
        };
        ProgressInfo {
            completed_samples: completed,
            total_samples: self.total_samples,
            percent: pct,
            total_frames: self.total_frames.load(Ordering::Relaxed),
            elapsed_secs: elapsed.as_secs_f64(),
            eta_secs: eta.as_secs_f64(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProgressInfo {
    pub completed_samples: u64,
    pub total_samples: u64,
    pub percent: f64,
    pub total_frames: u64,
    pub elapsed_secs: f64,
    pub eta_secs: f64,
}
```

## 5. 核心逻辑详解

### 5.1 metadata.json 生成流程

```
write_metadata(output_dir, manifest, results, start_time, end_time):
  ↓
1. 按任务名称将 ShardResult 分组（一个任务可能产生多个分片结果）：
   for task in manifest.tasks:
       task_results = results.filter(|r| r.task_name == task.name)

2. 对每个任务汇总：
   TaskMetadata {
       task_name:           task.name,
       generator:           task.generator,
       params_snapshot:     serde_json::to_value(&task.params),
       base_seed:           task.seed,
       sample_count:        task.count,
       shard_count:         task_results.len(),
       total_frames:        sum(task_results.stats.frames_written),
       total_bytes:         sum(task_results.stats.bytes_written),
       avg_frames_per_sample: total_frames / sample_count as f64,
       output_files:        for each r in task_results (where r.error.is_none()):
                                FileRecord {
                                    filename:  r.stats.output_path.file_name(),
                                    format:    r.format.to_string(),
                                    shard_id:  r.shard_idx,
                                    seed:      r.seed,
                                    frames:    r.stats.frames_written,
                                    file_size: r.stats.bytes_written,
                                    sha256:    r.stats.file_hash.clone(),
                                },
       failed_shards:       for each r in task_results (where r.error.is_some()):
                                FailedShardRecord {
                                    shard_id: r.shard_idx,
                                    seed:     r.seed,
                                    error_message: r.error.clone().unwrap(),
                                },
       elapsed_ms:          (来自 ShardResult 的计时或估算),
   }

3. 构建完整 RunMetadata：
   RunMetadata {
       software_version:    env!("CARGO_PKG_VERSION"),
       dependency_versions: collect_dep_versions(),  // 从 Cargo.toml 编译时嵌入
       start_time:          start_time.to_rfc3339(),
       end_time:            end_time.to_rfc3339(),
       elapsed_ms:          (end_time - start_time).as_millis(),
       global_config:       GlobalConfigSnapshot { ... },
       tasks:               task_metadatas,
       summary:             RunSummary { ... },
   }

4. 序列化为格式化的 JSON（pretty print）：
   let json = serde_json::to_string_pretty(&run_metadata)?;
   let path = output_dir.join("metadata.json");
   std::fs::write(&path, json)?;

5. 日志记录：info!("Metadata written to {}", path.display());
```

### 5.2 日志系统初始化

```
init_logger(log_level, log_file):
  ↓
1. 解析 log_level 为 tracing_subscriber::EnvFilter
   支持 RUST_LOG 环境变量覆盖

2. 创建控制台 Layer：
   - 使用带颜色的 compact 格式
   - 时间戳精确到毫秒
   - 包含模块路径和目标

3. 若 log_file 为 Some：
   - 创建文件 Layer
   - 使用 JSON 格式（便于日志聚合工具解析）
   - 创建父目录（若不存在）

4. 组合 Layer 并注册为全局 subscriber

5. 记录启动日志：
   info!("StructGen-rs v{} starting", env!("CARGO_PKG_VERSION"));
```

### 5.3 进度计算算法

```
ProgressTracker 内部状态:
    total_samples: u64         -- 在构造时设定，不变
    completed_samples: AtomicU64  -- 原子递增
    total_frames: AtomicU64       -- 原子递增
    start_instant: Instant        -- 单调时钟

report_completed(samples, frames):
    completed_samples += samples
    total_frames += frames

progress():
    completed = completed_samples.load()
    percent = completed / total_samples * 100
    elapsed = start_instant.elapsed()
    rate = completed / elapsed_secs
    eta = (total_samples - completed) / rate
    return ProgressInfo { completed_samples, percent, elapsed_secs, eta_secs, ... }
```

## 6. 与其他模块的交互

### 6.1 依赖关系

```
metadata
  ├── 依赖 core::{CoreResult, CoreError}
  ├── 依赖 scheduler::{Manifest, ShardResult}（仅类型引用，不调用调度器方法）
  ├── 被 CLI 调用（write_metadata 在运行结束后调用）
  └── 被 scheduler 间接使用（通过 ProgressTracker 报告进度）
```

### 6.2 调用时序

```
CLI main()
  │
  ├→ metadata::init_logger(log_level, log_file)    // 1. 初始化日志
  │
  ├→ let progress = ProgressTracker::new(total)    // 2. 创建进度追踪器
  │
  ├→ scheduler::run_manifest(manifest, registry)    // 3. 运行任务
  │     │
  │     └→ [每个分片完成时]
  │           progress.report_completed(samples, frames)  // 4. 实时报告进度
  │           log::info!("Shard {}/{} completed", ...)
  │
  ├→ metadata::write_metadata(                      // 5. 写入元数据
  │       output_dir, manifest, &results,
  │       start_time, end_time
  │   )
  │
  └→ log::info!("Done. Metadata written to {}/metadata.json", output_dir)
```

### 6.3 与其他模块的数据约定

| 约定 | 说明 |
|------|------|
| ShardResult → TaskMetadata | metadata 模块从 ShardResult 中提取 stats 构造 TaskMetadata |
| Manifest → RunMetadata | metadata 模块从 Manifest 中复制任务配置快照 |
| 文件名关联 | metadata.json 中的 FileRecord.filename 与 sink 模块的实际输出文件名一致 |
| 不重复计算 | metadata 模块使用 ShardResult 中已有的帧数和字节数，不重新扫描输出文件 |

## 7. 错误处理策略

| 错误情景 | 处理 |
|----------|------|
| 日志文件路径的父目录不存在 | `init_logger` 尝试创建父目录 |
| 日志文件无写入权限 | `init_logger` 降级为仅控制台输出，并打印警告 |
| metadata.json 写入失败 | 返回 `CoreError::IoError`，CLI 报告错误但运行结果仍有效 |
| SHA-256 计算失败 | 文件记录中 sha256 字段置为 None，不阻断流程 |
| 重复初始化日志 | `init_logger` 返回 `Err`，避免全局状态混乱 |
| 进度追踪器除零（total_samples=0） | `progress()` 返回 0% 和 0 ETA |

## 8. 性能考量

- **原子操作追踪进度**：`ProgressTracker` 使用 `AtomicU64` 和 `Relaxed` 内存序，极低开销。
- **日志异步写入**：`tracing-subscriber` 支持非阻塞写入器，日志 I/O 不阻塞生成线程。
- **metadata.json 一次写入**：所有统计信息在内存中汇总后一次性写出，不产生中间临时文件。
- **避免输出文件扫描**：帧数和字节数来自 `ShardResult`，而非 `fs::metadata()` 调用，减少系统调用。

## 9. 可测试性设计

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use chrono::Utc;

    fn make_mock_manifest() -> Manifest { /* ... */ }
    fn make_mock_results() -> Vec<ShardResult> { /* ... */ }

    #[test]
    fn test_write_metadata_creates_valid_json() {
        let tmp = TempDir::new().unwrap();
        let manifest = make_mock_manifest();
        let results = make_mock_results();

        write_metadata(tmp.path(), &manifest, &results, Utc::now(), Utc::now()).unwrap();

        let content = std::fs::read_to_string(tmp.path().join("metadata.json")).unwrap();
        let parsed: RunMetadata = serde_json::from_str(&content).unwrap();

        assert_eq!(parsed.tasks.len(), manifest.tasks.len());
        assert!(parsed.software_version.len() > 0);
    }

    #[test]
    fn test_progress_tracker_basic() {
        let tracker = ProgressTracker::new(100);
        assert_eq!(tracker.progress().percent, 0.0);

        tracker.report_completed(25, 250);
        let info = tracker.progress();
        assert!(info.percent > 20.0 && info.percent < 30.0);
    }

    #[test]
    fn test_progress_tracker_eta_decreases() {
        let tracker = ProgressTracker::new(100);
        tracker.report_completed(50, 500);
        // 等待一小段时间
        std::thread::sleep(std::time::Duration::from_millis(10));
        tracker.report_completed(25, 250);
        let info = tracker.progress();
        assert!(info.eta_secs >= 0.0);  // ETA 应为非负
    }

    #[test]
    fn test_write_metadata_records_failed_shards() {
        let tmp = TempDir::new().unwrap();
        let manifest = make_mock_manifest();
        let mut results = make_mock_results();
        results[0].error = Some("test error".to_string());

        write_metadata(tmp.path(), &manifest, &results, Utc::now(), Utc::now()).unwrap();

        let content = std::fs::read_to_string(tmp.path().join("metadata.json")).unwrap();
        let parsed: RunMetadata = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed.summary.failed_shard_count, 1);
    }
}
```

## 10. 配置与参数

| 参数 | 类型 | 来源 | 默认值 | 说明 |
|------|------|------|--------|------|
| `log_level` | String | CLI `--log-level` / GlobalConfig | "info" | trace/debug/info/warn/error |
| `log_file` | Option\<Path\> | GlobalConfig 或 CLI | None | 日志文件路径 |

元数据文件固定命名为 `metadata.json`，写入到 `output_dir` 根目录下。不提供自定义命名以避免 CLI 参数膨胀。

---

通过以上设计，元数据与监控层以最小的性能代价提供了完整的可追溯性和实时可观测性，确保每次 StructGen-rs 运行都留下完整、可审计的运行记录。
