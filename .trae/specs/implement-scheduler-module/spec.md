# 第三阶段：调度协调模块 (scheduler) Spec

## Why
Phase 2 已完成三个独立功能模块（sink、pipeline、generators）。Phase 3 需要实现系统协调中枢 scheduler，将 YAML 清单转换为可执行的生成任务，管理确定性种子派生，通过 rayon 并行执行所有分片任务，协调生成器→后处理管道→输出适配器之间的完整数据流动。

## What Changes
- 新增 `src/scheduler/` 模块，包含 seed、manifest、shard、executor、mod 五个源文件
- 更新 `Cargo.toml` 添加 `rayon`、`serde_yaml` 依赖
- 更新 `src/sink/mod.rs` 重导出 `OutputStats`（scheduler 需要）
- 更新 `src/main.rs` 添加 `mod scheduler;`

## Impact
- Affected specs: core + sink + pipeline + generators（全部 Phase 1/2 模块）
- Affected code: `Cargo.toml`, `src/scheduler/`, `src/sink/mod.rs`, `src/main.rs`

## ADDED Requirements

### Requirement: 种子派生
系统 SHALL 提供确定性的种子派生函数 `derive_seed(base_seed, shard_idx)`，使用 wrapping_add 确保不同分片种子唯一。

#### Scenario: 确定性
- **WHEN** 使用相同 (base_seed, shard_idx) 调用 derive_seed 多次
- **THEN** 每次返回相同结果

#### Scenario: wrapping 行为
- **WHEN** base_seed = u64::MAX, shard_idx = 1
- **THEN** 结果应为 0（wrapping_add）

### Requirement: 清单解析与校验
系统 SHALL 支持从 YAML 反序列化 Manifest 和 TaskSpec，并执行完整性校验。

#### Scenario: 拒绝未注册的生成器
- **WHEN** TaskSpec 引用了 GeneratorRegistry 中不存在的生成器名称
- **THEN** 返回 ManifestError

#### Scenario: 拒绝零样本数
- **WHEN** TaskSpec.count == 0
- **THEN** 返回 ManifestError

#### Scenario: 拒绝空输出目录
- **WHEN** GlobalConfig.output_dir 为空字符串
- **THEN** 返回 ConfigError

#### Scenario: 合法 YAML 解析成功
- **WHEN** YAML 包含有效任务列表和全局配置
- **THEN** 成功反序列化为 Manifest，所有字段正确填充

### Requirement: 分片切分
系统 SHALL 根据分片大小将任务切分为多个 Shard，每个 Shard 携带派生种子和样本数量。

#### Scenario: 均匀分片
- **WHEN** count=1000, shard_size=250
- **THEN** 产生 4 个分片，每个 sample_count=250，总和=1000

#### Scenario: 不均匀分片
- **WHEN** count=1001, shard_size=250
- **THEN** 产生 5 个分片（250+250+250+250+1），总和=1001

#### Scenario: 自动分片大小
- **WHEN** shard_size 未指定
- **THEN** 根据 CPU 核心数自动计算（target_shards = num_cpus * 4）

### Requirement: 分片执行
系统 SHALL 实现 `execute_shard()` 串联完整数据流：实例化生成器→组装后处理管道→驱动输出适配器→逐帧写出→收集统计。

#### Scenario: 端到端数据流
- **WHEN** 使用 mock 生成器（固定产出 10 帧）、空管道、mock 输出适配器执行分片
- **THEN** ShardResult 记录正确的 frames_written=10

#### Scenario: 单分片失败不影响其他分片
- **WHEN** 注入一个会 panic 的生成器
- **THEN** 该分片的 ShardResult.error 为 Some(...)，其他分片正常完成

### Requirement: 并行调度入口
系统 SHALL 提供 `run_manifest()` 作为模块唯一公开入口，使用 rayon 并行执行全部分片并汇总结果。

#### Scenario: 确定性与幂等性
- **WHEN** 相同清单运行两次
- **THEN** 两次产生的 ShardResult 列表完全一致（相同数量、相同顺序的值）

#### Scenario: 容错汇总
- **WHEN** 部分分片执行失败
- **THEN** run_manifest 返回包含成功和失败结果的完整 Vec<ShardResult>，不中止运行
