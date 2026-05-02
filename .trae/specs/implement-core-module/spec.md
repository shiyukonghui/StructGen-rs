# 基础设施 —— core 模块 Spec

## Why
core 模块是 StructGen-rs 系统的最底层，定义了所有模块共享的数据类型、trait 接口、错误类型和生成器注册表。没有 core 模块，sink、pipeline、generators、scheduler、metadata、CLI 等所有上层模块均无法开发。必须首先完成 core 模块并通过测试。

## What Changes
- 新建 `src/core/` 目录及全部子模块文件
- 更新 `Cargo.toml` 添加 `serde`、`serde_json`、`thiserror` 依赖（core 模块最小依赖集）
- **BREAKING**: 无（全新项目，无破坏性变更）

## Impact
- Affected specs: 无（首个 spec）
- Affected code: `src/core/`（新建目录）、`Cargo.toml`（新增依赖）

---

## ADDED Requirements

### Requirement: FrameState 标记联合体
系统 SHALL 提供 `FrameState` 枚举，统一承载整型（`Integer(i64)`）、浮点型（`Float(f64)`）和布尔型（`Bool(bool)`）三种状态值。每个变体 SHALL 实现 `as_integer()`、`as_float()`、`as_bool()` 三个安全取值方法，以及 `variant_name()` 返回变体名称。

#### Scenario: 整数值转换
- **WHEN** `FrameState::Integer(42)` 调用 `as_integer()`
- **THEN** 返回 `Some(42)`
- **WHEN** 调用 `as_float()`
- **THEN** 返回 `Some(42.0)`

#### Scenario: 浮点值不自动转换为整数
- **WHEN** `FrameState::Float(3.14)` 调用 `as_integer()`
- **THEN** 返回 `None`（防止精度静默丢失）

### Requirement: FrameData 帧状态向量
系统 SHALL 提供 `FrameData` 结构体，包含 `values: Vec<FrameState>` 字段，并提供 `new()`、`from_iter()`、`dim()`、`is_empty()` 方法。

#### Scenario: 创建帧数据
- **WHEN** 调用 `FrameData::from_iter([FrameState::Integer(1), FrameState::Bool(true)])`
- **THEN** `dim()` 返回 2，`is_empty()` 返回 false

### Requirement: SequenceFrame 时序帧
系统 SHALL 提供 `SequenceFrame` 结构体，包含 `step_index: u64`、`state: FrameData`、`label: Option<String>` 字段，并提供 `new()` 和 `with_label()` 构造方法。

#### Scenario: 创建带标签的帧
- **WHEN** 调用 `SequenceFrame::with_label(0, frame_data, "period_2")`
- **THEN** `step_index` = 0，`label` = `Some("period_2")`

### Requirement: GenParams 通用参数载体
系统 SHALL 提供 `GenParams` 结构体，包含 `seq_length: usize`、`grid_size: Option<GridSize>`、`extensions: HashMap<String, Value>` 字段，并提供 `simple()`、`get_extension::<T>()`、`set_extension::<T>()` 方法。

#### Scenario: 扩展字段往返
- **WHEN** 创建 `GenParams::simple(100)`，调用 `set_extension("rule", &110u32)` 后 `get_extension::<u32>("rule")`
- **THEN** 返回 `Ok(110)`

### Requirement: GridSize 网格尺寸
系统 SHALL 提供 `GridSize` 结构体，包含 `rows: usize` 和 `cols: usize` 字段。

### Requirement: OutputFormat 输出格式枚举
系统 SHALL 提供 `OutputFormat` 枚举，包含 `Parquet`、`Text`、`Binary` 三个变体。

### Requirement: GlobalConfig 全局配置
系统 SHALL 提供 `GlobalConfig` 结构体，包含 `num_threads`、`default_format`、`output_dir`、`log_level`、`shard_max_sequences`、`stream_write` 字段。

### Requirement: Generator trait 生成器接口
系统 SHALL 提供 `Generator` trait，要求实现者标注 `Send + Sync`，并实现 `name()`、`from_extensions()`、`generate_stream()` 方法。`generate_batch()` SHALL 提供默认实现（调用 `generate_stream` 后 collect）。

#### Scenario: 流式生成
- **WHEN** 调用 `generator.generate_stream(seed, &params)`
- **THEN** 返回 `Ok(Box<dyn Iterator<Item = SequenceFrame> + Send>)`

### Requirement: GeneratorRegistry 生成器注册表
系统 SHALL 提供 `GeneratorRegistry` 结构体，通过 `HashMap<&str, GeneratorFactory>` 管理名称到构造函数的映射，提供 `new()`、`register()`、`instantiate()`、`list_names()`、`contains()` 方法。

#### Scenario: 注册与查找
- **WHEN** 注册名称为 "ca" 的生成器后调用 `instantiate("ca", &extensions)`
- **THEN** 返回 `Ok(Box<dyn Generator>)`

#### Scenario: 查找未注册名称
- **WHEN** 调用 `instantiate("unknown", &extensions)`
- **THEN** 返回 `Err(CoreError::GeneratorNotFound(_))`

#### Scenario: 重复注册 panic
- **WHEN** 对同一名称调用两次 `register()`
- **THEN** 第二次调用 panic

### Requirement: CoreError 统一错误类型
系统 SHALL 提供 `CoreError` 枚举（派生 `thiserror::Error`），包含以下变体：`InvalidParams`、`GeneratorNotFound`、`GeneratorInitError`、`GenerationError`、`IoError`(from `std::io::Error`)、`SerializationError`、`ManifestError`、`PipelineError`、`SinkError`、`ConfigError`、`Other`。

#### Scenario: IoError 自动转换
- **WHEN** 在返回 `CoreResult<T>` 的函数中使用 `?` 传播 `std::io::Error`
- **THEN** 自动转换为 `CoreError::IoError`

### Requirement: CoreResult 类型别名
系统 SHALL 提供 `pub type CoreResult<T> = Result<T, CoreError>` 类型别名。

### Requirement: core 模块 mod.rs 重导出
`src/core/mod.rs` SHALL 重导出所有公开类型，使外部模块可通过 `use crate::core::*` 一次性导入全部公共接口。

### Requirement: 项目依赖配置
`Cargo.toml` SHALL 添加 `serde` (features: derive)、`serde_json`、`thiserror` 依赖。所有 core 类型 SHALL 派生或实现 `Debug`、`Clone`、`Serialize`、`Deserialize`（除 trait 对象和注册表外）。

### Requirement: 测试覆盖
core 模块 SHALL 通过以下测试：
- FrameState 值转换正确性
- GenParams 扩展字段序列化/反序列化往返
- GeneratorRegistry 注册、查找、未找到错误、重复注册 panic
- SequenceFrame / FrameData 构造方法
- CoreError Display 输出
