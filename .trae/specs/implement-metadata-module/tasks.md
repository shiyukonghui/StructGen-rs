# Tasks

- [x] Task 0: 添加 Cargo.toml 依赖和模块声明
  - [x] 在 Cargo.toml 中新增 `tracing`、`tracing-subscriber`、`chrono` 依赖
  - [x] 在 `src/main.rs` 中增加 `mod metadata;` 声明

- [x] Task 1: 实现 types.rs —— 元数据类型定义
  - [x] 定义 `GlobalConfigSnapshot` 结构体（output_dir, default_format, num_threads, log_level），实现 Serialize/Deserialize
  - [x] 定义 `FileRecord` 结构体（filename, format, shard_id, seed, frames, file_size, sha256），实现 Serialize/Deserialize
  - [x] 定义 `FailedShardRecord` 结构体（shard_id, seed, error_message），实现 Serialize/Deserialize
  - [x] 定义 `RunSummary` 结构体（total_tasks, total_samples, total_frames, total_bytes, total_files, failed_shard_count），实现 Serialize/Deserialize
  - [x] 定义 `TaskMetadata` 结构体（task_name, generator, params_snapshot, base_seed, sample_count, shard_count, total_frames, total_bytes, avg_frames_per_sample, output_files, elapsed_ms, failed_shards），实现 Serialize/Deserialize
  - [x] 定义 `RunMetadata` 结构体（software_version, dependency_versions, start_time, end_time, elapsed_ms, global_config, tasks, summary），实现 Serialize/Deserialize
  - [x] 编写 types 模块的单元测试：验证各类型序列化/反序列化往返正确

- [x] Task 2: 实现 progress.rs —— ProgressTracker 进度追踪器
  - [x] 定义 `ProgressInfo` 结构体（completed_samples, total_samples, percent, total_frames, elapsed_secs, eta_secs）
  - [x] 定义 `ProgressTracker` 结构体（total_samples: u64, completed_samples: Arc<AtomicU64>, total_frames: Arc<AtomicU64>, start_instant: Instant），实现 Clone
  - [x] 实现 `ProgressTracker::new(total_samples: usize) -> Self`
  - [x] 实现 `ProgressTracker::report_completed(&self, samples: usize, frames: u64)`
  - [x] 实现 `ProgressTracker::progress(&self) -> ProgressInfo`，包含百分比计算与线性 ETA 估算
  - [x] 编写 progress 模块的单元测试：百分比范围、ETA 非负、零样本安全、多线程并发报告

- [x] Task 3: 实现 recorder.rs —— write_metadata() 元数据写入
  - [x] 实现 `collect_dep_versions() -> HashMap<String, String>`：编译时收集依赖库版本
  - [x] 实现 `write_metadata()`：按任务名分组 ShardResult → 汇总 TaskMetadata → 构建 RunMetadata → 序列化为格式化 JSON → 写入 metadata.json
  - [x] 正确处理失败分片：error 为 Some 的分片计入 failed_shards，不产生 FileRecord
  - [x] 错误处理：文件写入失败返回 CoreError::IoError
  - [x] 日志记录：写入成功后 info 级别记录文件路径
  - [x] 编写 recorder 模块的单元测试：产出合法 JSON、失败分片正确记录、空结果处理

- [x] Task 4: 实现 logger.rs —— init_logger() 日志系统
  - [x] 实现 `init_logger(log_level: &str, log_file: Option<&Path>) -> CoreResult<()>`
  - [x] 支持日志级别解析（trace/debug/info/warn/error），无效级别返回 Err
  - [x] 创建控制台 Layer：彩色 compact 格式，精确到毫秒
  - [x] 若 log_file 为 Some，创建文件 Layer：JSON 格式，自动创建父目录
  - [x] 组合 Layer 并注册为全局 subscriber
  - [x] 重复初始化检测：第二次调用返回 Err
  - [x] 记录启动日志：info!("StructGen-rs v{} starting", ...)
  - [x] 编写 logger 模块的单元测试：控制台初始化、文件输出、重复初始化报错、无效日志级别

- [x] Task 5: 实现 mod.rs —— 模块根
  - [x] 声明子模块：pub mod types、pub mod progress、pub mod recorder、pub mod logger
  - [x] 重导出所有公开类型和函数

- [x] Task 6: 最终验证
  - [x] 运行 `cargo test` 确保所有测试通过（260 passed, 0 failed）
  - [x] 运行 `cargo clippy` 确保无新增 lint 警告
  - [x] 运行 `cargo build` 确保编译成功

# Task Dependencies
- Task 0 必须在 Task 1-5 之前完成
- Task 1、Task 2、Task 3、Task 4 可并行开发
- Task 5 依赖 Task 1-4 完成
- Task 6 依赖 Task 0-5 完成
