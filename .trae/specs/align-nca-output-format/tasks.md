# Tasks

- [x] Task 1: 修改 NCA 温度参数默认值：将 `default_temperature()` 从 `1.0` 改为 `0.0`
  - [x] SubTask 1.1: 修改 `src/generators/nca2d.rs` 中的 `default_temperature()` 函数
  - [x] SubTask 1.2: 更新相关测试用例的期望值
  - [x] SubTask 1.3: 运行测试验证修改正确性

- [x] Task 2: 实现 SequenceStitcher 处理器：将多帧串联为训练序列格式
  - [x] SubTask 2.1: 创建 `src/pipeline/sequence_stitcher.rs` 文件
  - [x] SubTask 2.2: 定义 `SequenceStitcherConfig` 配置结构体
  - [x] SubTask 2.3: 实现 `SequenceStitcher` 处理器
  - [x] SubTask 2.4: 实现迭代器适配器 `SequenceStitchIter`
  - [x] SubTask 2.5: 编写单元测试验证序列串联逻辑
  - [x] SubTask 2.6: 在 `src/pipeline/mod.rs` 中注册处理器

- [x] Task 3: 实现 BatchCollector 处理器：将流式帧收集为批量张量
  - [x] SubTask 3.1: 创建 `src/pipeline/batch_collector.rs` 文件
  - [x] SubTask 3.2: 定义 `BatchCollectorConfig` 配置结构体
  - [x] SubTask 3.3: 实现 `BatchCollector` 处理器
  - [x] SubTask 3.4: 定义 `BatchData` 结构体承载批量数据
  - [x] SubTask 3.5: 编写单元测试验证批量收集逻辑
  - [x] SubTask 3.6: 在 `src/pipeline/mod.rs` 中注册处理器

- [x] Task 4: 实现 NpyBatchSink 输出器：输出 NumPy 格式批量数据
  - [x] SubTask 4.1: 创建 `src/sink/npy_batch.rs` 文件
  - [x] SubTask 4.2: 定义 `NpyBatchSinkConfig` 配置结构体
  - [x] SubTask 4.3: 实现 NumPy 文件格式写入逻辑（使用 `npy` crate）
  - [x] SubTask 4.4: 实现 `Sink` trait
  - [x] SubTask 4.5: 编写单元测试验证输出格式
  - [x] SubTask 4.6: 在 `src/sink/mod.rs` 中注册输出器

- [x] Task 5: 集成测试与验证：确保端到端流程正确
  - [x] SubTask 5.1: 编写集成测试验证完整数据生成流程
  - [x] SubTask 5.2: 验证输出数据可被 Python `numpy.load()` 正确加载
  - [x] SubTask 5.3: 更新文档说明新增的处理器和输出器

# Task Dependencies
- [Task 2] 可独立进行
- [Task 3] 可独立进行
- [Task 4] 依赖 [Task 3]（需要 BatchData 结构体）
- [Task 5] 依赖 [Task 1], [Task 2], [Task 3], [Task 4]