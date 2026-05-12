# Tasks

- [x] Task 1: 新增 OutputFormat::NpyBatch 枚举变体
  - [x] SubTask 1.1: 在 `src/core/params.rs` 中为 `OutputFormat` 枚举添加 `NpyBatch` 变体
  - [x] SubTask 1.2: 添加文档注释说明 NpyBatch 用于批量输出 (B,T,H,W,C) 形状
  - [x] SubTask 1.3: 运行单元测试验证枚举序列化/反序列化正常

- [x] Task 2: 修改 SinkAdapterFactory 类型签名
  - [x] SubTask 2.1: 在 `src/sink/factory.rs` 中修改 `SinkAdapterFactory` 类型为 `fn(OutputFormat, &Value) -> CoreResult<Box<dyn SinkAdapter>>`
  - [x] SubTask 2.2: 更新文档注释说明新增的 config 参数用途

- [x] Task 3: 修改 create_adapter 函数支持 NpyBatch
  - [x] SubTask 3.1: 在 `src/main.rs` 中修改 `create_adapter` 函数签名，增加 `config: &Value` 参数
  - [x] SubTask 3.2: 在 match 分支中添加 `OutputFormat::NpyBatch` 处理逻辑
  - [x] SubTask 3.3: 实现 NpyBatchConfig 反序列化和错误处理
  - [x] SubTask 3.4: 确保其他格式（Parquet/Text/Binary/Npy）忽略 config 参数

- [x] Task 4: 新增 CliFormat::NpyBatch 变体
  - [x] SubTask 4.1: 在 `src/main.rs` 中为 `CliFormat` 枚举添加 `NpyBatch` 变体
  - [x] SubTask 4.2: 更新 `From<CliFormat> for OutputFormat` 实现添加 NpyBatch 映射
  - [x] SubTask 4.3: 添加文档注释说明 CLI 使用 `--format npy-batch`

- [x] Task 5: 修改 executor.rs 提取并传递 sink_config
  - [x] SubTask 5.1: 在 `execute_shard` 函数签名中更新 `adapter_factory` 参数类型
  - [x] SubTask 5.2: 在 `execute_shard_inner` 中从 `task.params.extensions` 提取 `sink_config`
  - [x] SubTask 5.3: 将 `sink_config` 传递给 `adapter_factory(format, &sink_config)`
  - [x] SubTask 5.4: 更新 `mock_adapter_factory` 测试辅助函数签名

- [x] Task 6: 修改 scheduler/mod.rs 传递 sink_config
  - [x] SubTask 6.1: 在 `run_manifest` 函数签名中更新 `adapter_factory` 参数类型
  - [x] SubTask 6.2: 更新 `mock_adapter_factory` 测试辅助函数签名
  - [x] SubTask 6.3: 确保测试用例正常通过

- [x] Task 7: 运行完整测试套件验证
  - [x] SubTask 7.1: 运行 `cargo test` 确保所有现有测试通过
  - [x] SubTask 7.2: 运行 `cargo clippy` 确保无 lint 错误
  - [x] SubTask 7.3: 运行 `cargo build` 确保编译成功

# Task Dependencies

- Task 2 依赖 Task 1（需要先有 NpyBatch 变体才能在工厂中处理）
- Task 3 依赖 Task 1 和 Task 2（需要枚举变体和工厂类型签名）
- Task 4 依赖 Task 1（需要 OutputFormat::NpyBatch 存在）
- Task 5 依赖 Task 2（需要新的工厂类型签名）
- Task 6 依赖 Task 2 和 Task 5（需要工厂类型签名和 executor 更新）
- Task 7 依赖所有前置任务完成