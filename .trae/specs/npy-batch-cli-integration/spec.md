# NpyBatchAdapter CLI 管道集成 Spec

## Why

`NpyBatchAdapter` 已实现完整的 NPY 批量输出功能（7 个单元测试全部通过），但未集成到 CLI 管道。当前 `create_adapter` 函数将 `OutputFormat::Npy` 映射到 `NpyAdapter`（单帧输出），而 `NpyBatchAdapter`（批量 `(B,T,H,W,C)` 输出）只能通过编程 API 调用，无法通过 YAML 配置或 CLI 参数使用。

核心障碍：`SinkAdapterFactory` 类型签名为 `fn(OutputFormat) -> CoreResult<Box<dyn SinkAdapter>>`，不接受配置参数，而 `NpyBatchAdapter` 需要 `NpyBatchConfig`（包含 `batch_size`、`num_frames`、可选 `rows/cols/channels`）。

## What Changes

1. **新增** `OutputFormat::NpyBatch` 枚举变体，在 `src/core/params.rs` 中
2. **修改** `SinkAdapterFactory` 类型签名为 `fn(OutputFormat, &Value) -> CoreResult<Box<dyn SinkAdapter>>`，在 `src/sink/factory.rs` 中
3. **修改** `create_adapter` 函数，增加 `config` 参数，处理 `NpyBatch` 变体，在 `src/main.rs` 中
4. **修改** `execute_shard_inner` 函数，从 `task.params.extensions` 提取 `sink_config` 并传给 `adapter_factory`，在 `src/scheduler/executor.rs` 中
5. **修改** `run_manifest` 及相关调用链传递 `sink_config`
6. **新增** `CliFormat::NpyBatch` 变体及对应 `From<CliFormat>` 映射，在 `src/main.rs` 中
7. **更新** 各处 `mock_adapter_factory` 签名以适配新的 `SinkAdapterFactory` 类型
8. **新增** 端到端集成测试和 YAML 配置示例

## Impact

- Affected code:
  - `src/core/params.rs` - OutputFormat 枚举新增变体
  - `src/sink/factory.rs` - SinkAdapterFactory 类型签名变更
  - `src/main.rs` - create_adapter 函数、CliFormat 枚举、mock/test
  - `src/scheduler/executor.rs` - execute_shard_inner 提取 sink_config
  - `src/scheduler/mod.rs` - run_manifest 传递适配器工厂
  - `src/sink/npy_batch.rs` - NpyBatchAdapter 无代码变更，但需确保 NpyBatchConfig 可从 Value 反序列化

## ADDED Requirements

### Requirement: OutputFormat 支持 NpyBatch 变体

系统 SHALL 在 `OutputFormat` 枚举中新增 `NpyBatch` 变体，YAML 中使用 `"NpyBatch"` 字符串表示。

#### Scenario: YAML 配置指定 NpyBatch 输出格式

```yaml
tasks:
  - name: "nca2d_batch"
    output_format: "NpyBatch"
    params:
      extensions:
        sink_config:
          batch_size: 10
          num_frames: 5
          rows: 12
          cols: 12
          channels: 10
```

- **WHEN** 用户在 YAML 清单中指定 `output_format: "NpyBatch"`
- **THEN** 系统创建 `NpyBatchAdapter` 实例，使用 `sink_config` 中的配置

### Requirement: SinkAdapterFactory 接受配置参数

系统 SHALL 将 `SinkAdapterFactory` 类型从 `fn(OutputFormat) -> CoreResult<Box<dyn SinkAdapter>>` 修改为 `fn(OutputFormat, &Value) -> CoreResult<Box<dyn SinkAdapter>>`。

#### Scenario: 不需要配置的格式传入空配置

- **WHEN** 输出格式为 `Parquet`/`Text`/`Binary`/`Npy`
- **THEN** 工厂函数忽略 `config` 参数，创建默认实例

#### Scenario: NpyBatch 传入配置

- **WHEN** 输出格式为 `NpyBatch`
- **THEN** 工厂函数从 `config` 反序列化为 `NpyBatchConfig`，创建 `NpyBatchAdapter` 实例

### Requirement: sink_config 配置传递

系统 SHALL 从 `task.params.extensions["sink_config"]` 提取输出器配置，传递给 `adapter_factory`。

#### Scenario: YAML 中配置 sink_config

- **WHEN** YAML 任务 `params.extensions` 包含 `sink_config` 字段
- **THEN** 该字段的值作为 `Value` 传递给 `adapter_factory` 的第二个参数

#### Scenario: 未配置 sink_config

- **WHEN** YAML 任务 `params.extensions` 不包含 `sink_config` 字段
- **THEN** 传递 `Value::Null` 给 `adapter_factory`

### Requirement: CLI --format 支持 npy-batch

系统 SHALL 在 `CliFormat` 枚举中新增 `NpyBatch` 变体，CLI 中使用 `--format npy-batch` 表示。

#### Scenario: CLI 指定 npy-batch 格式

- **WHEN** 用户使用 `--format npy-batch`
- **THEN** 等价于 YAML 中 `output_format: "NpyBatch"`

## MODIFIED Requirements

### Requirement: SinkAdapterFactory 类型签名

原有类型签名 `fn(OutputFormat) -> CoreResult<Box<dyn SinkAdapter>>` 修改为 `fn(OutputFormat, &Value) -> CoreResult<Box<dyn SinkAdapter>>`。

所有使用该类型的函数签名均需更新：
- `execute_shard` 函数签名
- `execute_shard_inner` 函数签名
- `run_manifest` 函数签名
- 所有 `mock_adapter_factory` 测试辅助函数

## REMOVED Requirements

无移除的需求。

## Boundary Conditions and Exception Handling

1. **NpyBatch 无 sink_config**: `create_adapter` 返回 `InvalidParams` 错误，提示用户必须提供配置
2. **sink_config 反序列化失败**: 返回 `SerializationError`，包含具体解析错误信息
3. **NpyBatchConfig 校验**: `batch_size >= 1` 和 `num_frames >= 1` 在 `create_npy_batch_adapter` 中已有校验
4. **向后兼容**: 现有 Parquet/Text/Binary/Npy 格式不受影响，`config` 参数被忽略
5. **SinkAdapterFactory 类型变更**: 所有使用该类型的函数签名（`mock_adapter_factory` 等）均需更新

## Data Flow Path

```
YAML: output_format: "NpyBatch"
  + extensions.sink_config: { batch_size: 10, ... }
      │
      ▼
TaskSpec.output_format = Some(OutputFormat::NpyBatch)
TaskSpec.params.extensions["sink_config"] = Value::Object({...})
      │
      ▼
executor.rs: execute_shard_inner
  format = OutputFormat::NpyBatch
  sink_config = extensions["sink_config"].cloned().unwrap_or(Null)
  adapter = adapter_factory(format, &sink_config)
      │
      ▼
main.rs: create_adapter(OutputFormat::NpyBatch, config)
  → serde_json::from_value::<NpyBatchConfig>(config)
  → NpyBatchAdapter::new(npy_config)
      │
      ▼
NpyBatchAdapter::open → write_frame → close
  输出 .npy 文件，shape: (B, T, H, W, C)
```

## Expected Outcomes

1. 用户可通过 YAML `output_format: "NpyBatch"` 指定批量输出格式
2. 用户可通过 CLI `--format npy-batch` 指定批量输出格式
3. `NpyBatchAdapter` 从 `extensions.sink_config` 读取配置
4. 现有格式（Parquet/Text/Binary/Npy）行为不变
5. 全部现有测试 + 新增测试通过