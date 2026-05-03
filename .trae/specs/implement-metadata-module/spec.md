# Metadata 模块 Spec

## Why
元数据与监控层是 StructGen-rs 的可追溯性与可观测性基础。每次生成运行需要记录完整的参数快照、数据统计、完整性校验及实时进度，确保数据集可完整复现和审计。

## What Changes
- **BREAKING**: Cargo.toml 新增 `tracing`、`tracing-subscriber`、`chrono` 依赖
- 新增 `src/metadata/types.rs`：RunMetadata、TaskMetadata、FileRecord、FailedShardRecord、RunSummary、GlobalConfigSnapshot 类型定义
- 新增 `src/metadata/progress.rs`：ProgressTracker 实时进度追踪器及 ProgressInfo
- 新增 `src/metadata/recorder.rs`：write_metadata() 汇总 ShardResult → RunMetadata → JSON
- 新增 `src/metadata/logger.rs`：init_logger() 基于 tracing-subscriber 的日志系统
- 新增 `src/metadata/mod.rs`：模块根，公开重导出
- 修改 `src/main.rs`：增加 `mod metadata;` 声明

## Impact
- Affected specs: core、scheduler、sink（仅类型引用，不修改已有代码）
- Affected code: `src/main.rs`、`Cargo.toml`、新建 `src/metadata/` 目录下全部文件

## ADDED Requirements

### Requirement: 元数据类型定义
系统 SHALL 提供完整的元数据类型体系，包含 RunMetadata、TaskMetadata、FileRecord、FailedShardRecord、RunSummary、GlobalConfigSnapshot，所有类型均实现 Serialize/Deserialize。

#### Scenario: RunMetadata 包含完整运行信息
- **WHEN** 系统完成一次生成运行
- **THEN** RunMetadata 包含 software_version、dependency_versions、start_time、end_time、elapsed_ms、global_config、tasks、summary 所有字段

#### Scenario: TaskMetadata 正确汇总分片统计
- **WHEN** 一个任务有多个分片结果
- **THEN** TaskMetadata 的 total_frames 等于所有分片 frames_written 之和，total_bytes 等于所有分片 bytes_written 之和

#### Scenario: FileRecord 正确映射 OutputStats
- **WHEN** 分片成功完成且有 output_path
- **THEN** FileRecord 的 filename 为 output_path 的文件名部分，frames 和 file_size 与 OutputStats 一致

### Requirement: write_metadata 写入元数据 JSON
系统 SHALL 提供 write_metadata() 函数，接收 Manifest、ShardResult 列表、起止时间，生成格式化的 metadata.json 文件。

#### Scenario: 产出合法 JSON 且字段完整
- **WHEN** 调用 write_metadata 并传入合法的 Manifest 和 ShardResult
- **THEN** 生成 metadata.json，反序列化后 RunMetadata 各字段与输入一致

#### Scenario: 失败分片被正确记录
- **WHEN** 某个 ShardResult 的 error 字段为 Some
- **THEN** 对应 FailedShardRecord 被计入 TaskMetadata.failed_shards，且 RunSummary.failed_shard_count > 0

#### Scenario: 写入失败返回错误
- **WHEN** output_dir 无写入权限
- **THEN** write_metadata 返回 CoreError::IoError

### Requirement: ProgressTracker 实时进度追踪
系统 SHALL 提供 ProgressTracker，支持原子操作更新进度并计算百分比和 ETA。

#### Scenario: 百分比计算与 ETA 非负
- **WHEN** ProgressTracker 已报告部分完成
- **THEN** progress() 返回的 percent 在 0-100 之间，eta_secs >= 0

#### Scenario: 零样本除零安全
- **WHEN** ProgressTracker 以 total_samples=0 创建
- **THEN** progress() 返回 percent=0.0 且 eta_secs=0.0，不 panic

### Requirement: init_logger 日志系统初始化
系统 SHALL 提供 init_logger() 函数，基于 tracing-subscriber 实现控制台+可选文件双输出。

#### Scenario: 控制台输出正常
- **WHEN** 调用 init_logger("info", None)
- **THEN** 日志系统成功初始化，后续 log::info! 调用输出到控制台

#### Scenario: 文件输出正常
- **WHEN** 调用 init_logger("debug", Some(&log_file_path))
- **THEN** 日志同时输出到控制台和指定文件

#### Scenario: 重复初始化应报错
- **WHEN** init_logger 被调用第二次
- **THEN** 返回 Err，避免全局状态混乱

## REMOVED Requirements
无
