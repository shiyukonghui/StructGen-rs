# NCA 输出格式兼容 Python 版本 Spec

## Why
Rust 版本 (`StructGen-rs`) 与 Python 版本 (`nca-pre-pretraining`) 的 NCA 数据生成格式存在关键差异，导致生成的数据无法直接用于 Python 训练流程。主要问题包括：温度参数默认值不同（Python `0.0` vs Rust `1.0`）、序列组织方式不同（Python 自动串联多网格 vs Rust 每帧独立）、以及批量生成与流式生成的差异。

## What Changes
- **修改** `default_temperature()` 函数，将默认值从 `1.0` 改为 `0.0`，匹配 Python 版本的确定性演化行为
- **新增** `SequenceStitcher` 处理器，用于将多个独立帧串联成训练序列格式 `[start][frame0_patches][end][start][frame1_patches][end]...`
- **新增** `BatchCollector` 处理器，用于将流式帧收集为批量张量格式 `(B, T, H, W, C)`
- **新增** `NpyBatchSink` 输出器，用于输出 NumPy 格式的批量数据文件，兼容 Python 训练数据加载

## Impact
- Affected specs: NCA 数据生成、训练数据格式、数据输出
- Affected code:
  - `src/generators/nca2d.rs` - 温度参数默认值修改
  - `src/pipeline/sequence_stitcher.rs` - 新增序列串联处理器
  - `src/pipeline/batch_collector.rs` - 新增批量收集处理器
  - `src/sink/npy_batch.rs` - 新增 NumPy 批量输出器
  - `src/pipeline/mod.rs` - 注册新处理器
  - `src/sink/mod.rs` - 注册新输出器

## ADDED Requirements

### Requirement: 温度参数默认值兼容
系统 SHALL 将 NCA 生成器的 `temperature` 参数默认值设置为 `0.0`，以匹配 Python 版本的确定性演化行为。

#### Scenario: 默认温度参数产生确定性演化
- **WHEN** 用户使用默认参数创建 NCA 生成器
- **THEN** 生成的演化序列在同种子下完全确定性（argmax 选择）

### Requirement: 序列串联处理器
系统 SHALL 提供 `SequenceStitcher` 处理器，将多个独立帧的 token 序列串联成训练格式。

#### Scenario: 串联多帧为训练序列
- **WHEN** 用户配置 `frames_per_sequence=5` 的 SequenceStitcher
- **THEN** 输出序列格式为 `[start][frame0_patches][end][start][frame1_patches][end]...[start][frame4_patches][end]`

#### Scenario: 序列级 start/end token
- **WHEN** 用户配置 `add_sequence_start=true` 和 `add_sequence_end=true`
- **THEN** 整个序列开头和结尾添加额外的 start/end token

### Requirement: 批量收集处理器
系统 SHALL 提供 `BatchCollector` 处理器，将流式帧收集为批量张量格式。

#### Scenario: 收集帧为批量张量
- **WHEN** 用户配置 `batch_size=10` 和 `num_frames=5` 的 BatchCollector
- **THEN** 输出为 `(10, 5, H, W, C)` 形状的批量数据

### Requirement: NumPy 批量输出器
系统 SHALL 提供 `NpyBatchSink` 输出器，输出 NumPy 格式的批量数据文件。

#### Scenario: 输出 NumPy 格式文件
- **WHEN** 用户配置 NpyBatchSink 输出到指定路径
- **THEN** 生成 `.npy` 文件，可直接用 `numpy.load()` 加载

## MODIFIED Requirements

### Requirement: NCA 生成器参数默认值
原有默认值 `temperature=1.0` 修改为 `temperature=0.0`，确保默认行为与 Python 版本一致。

```rust
// 修改前
fn default_temperature() -> f64 { 1.0 }

// 修改后
fn default_temperature() -> f64 { 0.0 }
```

## REMOVED Requirements
无移除的需求。