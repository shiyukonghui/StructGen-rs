# Tasks: 实现 core 模块

- [ ] Task 1: 更新 Cargo.toml 添加依赖
  - 添加 `serde = { version = "1", features = ["derive"] }`
  - 添加 `serde_json = "1"`
  - 添加 `thiserror = "1"`
  - 运行 `cargo check` 确认依赖拉取成功

- [ ] Task 2: 实现 core::frame —— 帧数据类型
  - 创建 `src/core/frame.rs`
  - 定义 `FrameState` 枚举（Integer/Float/Bool），派生 Debug/Clone/Copy/PartialEq/Serialize/Deserialize
  - 实现 `as_integer()`、`as_float()`、`as_bool()`、`variant_name()` 方法
  - 定义 `FrameData` 结构体（values: Vec\<FrameState\>），派生 Debug/Clone/PartialEq/Serialize/Deserialize
  - 实现 `new()`、`from_iter()`、`dim()`、`is_empty()` 方法
  - 定义 `SequenceFrame` 结构体（step_index/u64 + state/FrameData + label/Option\<String\>），派生 Debug/Clone/PartialEq/Serialize/Deserialize
  - 实现 `new()`、`with_label()` 方法
  - 编写单元测试：FrameState 值转换、FrameData 构造、SequenceFrame 构造

- [ ] Task 3: 实现 core::error —— 错误类型
  - 创建 `src/core/error.rs`
  - 定义 `CoreError` 枚举（派生 thiserror::Error），包含全部 11 个变体
  - 为 `std::io::Error` 实现 `From` 自动转换（`IoError` 变体）
  - 定义 `CoreResult<T>` 类型别名
  - 编写单元测试：Display 输出、io::Error 自动转换

- [ ] Task 4: 实现 core::params —— 参数类型
  - 创建 `src/core/params.rs`
  - 定义 `GridSize` 结构体（rows/usize + cols/usize），派生 Debug/Clone/Copy/PartialEq/Serialize/Deserialize
  - 定义 `OutputFormat` 枚举（Parquet/Text/Binary），派生 Debug/Clone/Copy/PartialEq/Eq/Serialize/Deserialize
  - 定义 `GlobalConfig` 结构体（6 个字段），派生 Debug/Clone/Serialize/Deserialize
  - 定义 `GenParams` 结构体（seq_length + grid_size/Option\<GridSize\> + extensions/HashMap\<String,Value\>），派生 Debug/Clone/Serialize/Deserialize
  - 实现 `simple()`、`get_extension::<T>()`、`set_extension::<T>()` 方法
  - 编写单元测试：GenParams 扩展字段往返、GridSize 构造

- [ ] Task 5: 实现 core::generator —— Generator trait
  - 创建 `src/core/generator.rs`
  - 定义 `Generator` trait（标注 Send + Sync），包含 `name()`、`from_extensions()`、`generate_stream()` 三个必须方法
  - 为 `generate_batch()` 提供默认实现（调用 generate_stream + collect）
  - 注意：trait 不派生 serde，所有方法签名按设计文档定义

- [ ] Task 6: 实现 core::registry —— GeneratorRegistry
  - 创建 `src/core/registry.rs`
  - 定义 `GeneratorFactory` 类型别名
  - 定义 `GeneratorRegistry` 结构体（内部 HashMap\<&'static str, GeneratorFactory\>）
  - 实现 `new()`、`register()`（重复注册 panic）、`instantiate()`（未找到返回 Error）、`list_names()`、`contains()` 方法
  - 派生 `Default`
  - 编写单元测试：注册→查找 成功、查找不存在名称 失败、重复注册 panic、list_names 返回完整列表

- [ ] Task 7: 实现 core::mod —— 模块根
  - 创建 `src/core/mod.rs`
  - 声明子模块：`pub mod frame; pub mod error; pub mod params; pub mod generator; pub mod registry;`
  - 重导出全部公开类型（`pub use frame::*;` 等）
  - 运行 `cargo check --lib` 确保所有类型正确导出

- [ ] Task 8: 运行全量测试并验证
  - 运行 `cargo test` 确保所有单元测试通过
  - 运行 `cargo clippy` 确保无警告
  - 运行 `cargo build` 确保编译通过

# Task Dependencies
- Task 2～7 全部依赖于 Task 1（Cargo.toml 依赖）
- Task 2、3、4、5 相互独立，可并行开发
- Task 6 依赖于 Task 5（Generator trait 定义）
- Task 7 依赖于 Task 2～6 全部完成
- Task 8 依赖于 Task 7 完成
