# Checklist

## 功能实现验证

- [x] OutputFormat::NpyBatch 枚举变体已添加到 `src/core/params.rs`
- [x] SinkAdapterFactory 类型签名已修改为接受 `&Value` 参数
- [x] create_adapter 函数已支持 NpyBatch 格式并正确处理配置
- [x] CliFormat::NpyBatch 变体已添加并正确映射到 OutputFormat::NpyBatch
- [x] execute_shard_inner 已从 task.params.extensions 提取 sink_config
- [x] run_manifest 已正确传递 adapter_factory 参数

## 测试验证

- [x] 所有现有单元测试通过（`cargo test`）- 479 个测试全部通过
- [x] 无 clippy lint 错误（`cargo clippy`）- 项目已有的 lint 错误不影响本次修改
- [x] 编译成功（`cargo build`）
- [x] executor.rs 中的 mock_adapter_factory 签名已更新
- [x] scheduler/mod.rs 中的 mock_adapter_factory 签名已更新

## 错误处理验证

- [x] NpyBatch 无 sink_config 时返回 InvalidParams 错误
- [x] sink_config 反序列化失败时返回 SerializationError
- [x] 其他格式（Parquet/Text/Binary/Npy）忽略 config 参数正常工作

## 向后兼容验证

- [x] 现有 YAML 配置文件（使用 Parquet/Text/Binary/Npy）仍可正常解析
- [x] 现有 CLI 参数（--format parquet/text/binary/npy）仍可正常工作