# Checklist: core 模块实现验证

## 编译与代码质量
- [ ] `cargo check` 无错误
- [ ] `cargo build` 编译通过
- [ ] `cargo clippy` 无警告

## 类型定义完整性
- [ ] `FrameState` 枚举包含 `Integer(i64)`、`Float(f64)`、`Bool(bool)` 三个变体
- [ ] `FrameState` 实现 `as_integer()`、`as_float()`、`as_bool()`、`variant_name()` 方法
- [ ] `FrameData` 结构体包含 `values: Vec<FrameState>` 字段
- [ ] `FrameData` 实现 `new()`、`from_iter()`、`dim()`、`is_empty()` 方法
- [ ] `SequenceFrame` 结构体包含 `step_index: u64`、`state: FrameData`、`label: Option<String>` 字段
- [ ] `SequenceFrame` 实现 `new()`、`with_label()` 方法
- [ ] `CoreError` 枚举包含全部 11 个变体，派生 `thiserror::Error`
- [ ] `CoreError` 为 `std::io::Error` 实现 `From` trait（`IoError` 变体使用 `#[from]`）
- [ ] `CoreResult<T>` 类型别名定义为 `Result<T, CoreError>`
- [ ] `GridSize` 结构体包含 `rows` 和 `cols` 字段
- [ ] `OutputFormat` 枚举包含 `Parquet`、`Text`、`Binary` 三个变体
- [ ] `GlobalConfig` 结构体包含全部 6 个字段（`num_threads`、`default_format`、`output_dir`、`log_level`、`shard_max_sequences`、`stream_write`）
- [ ] `GenParams` 结构体包含 `seq_length`、`grid_size`、`extensions` 字段
- [ ] `GenParams` 实现 `simple()`、`get_extension::<T>()`、`set_extension::<T>()` 方法
- [ ] `Generator` trait 标注 `Send + Sync`，包含 `name()`、`from_extensions()`、`generate_stream()` 方法
- [ ] `Generator` trait 的 `generate_batch()` 有默认实现（委托给 `generate_stream` + collect）
- [ ] `GeneratorRegistry` 结构体有 `new()`、`register()`、`instantiate()`、`list_names()`、`contains()` 方法
- [ ] `GeneratorRegistry` 的 `register()` 在重复注册时 panic
- [ ] `GeneratorRegistry` 的 `instantiate()` 在未注册时返回 `Err(CoreError::GeneratorNotFound(_))`
- [ ] `src/core/mod.rs` 正确声明子模块并重导出全部公开类型

## 测试通过
- [ ] `test_frame_state_conversions` —— FrameState 值转换测试通过
- [ ] `test_gen_params_extension_roundtrip` —— GenParams 扩展字段往返测试通过
- [ ] `test_registry_register_and_instantiate` —— 注册表注册与实例化测试通过
- [ ] `test_registry_name_not_found` —— 注册表未找到名称测试通过
- [ ] `test_registry_duplicate_panics` —— 注册表重复注册 panic 测试通过
- [ ] `cargo test` 全部测试通过，覆盖率 ≥ 80%

## 派生 trait 验证
- [ ] `FrameState` 正确派生 `Debug, Clone, Copy, PartialEq, Serialize, Deserialize`
- [ ] `FrameData` 正确派生 `Debug, Clone, PartialEq, Serialize, Deserialize`
- [ ] `SequenceFrame` 正确派生 `Debug, Clone, PartialEq, Serialize, Deserialize`
- [ ] `GridSize` 正确派生 `Debug, Clone, Copy, PartialEq, Serialize, Deserialize`
- [ ] `OutputFormat` 正确派生 `Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize`
- [ ] `GlobalConfig` 正确派生 `Debug, Clone, Serialize, Deserialize`
- [ ] `GenParams` 正确派生 `Debug, Clone, Serialize, Deserialize`
- [ ] `CoreError` 正确派生 `Error, Debug`
- [ ] `GeneratorRegistry` 正确派生 `Default`
