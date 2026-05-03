# Checklist: 第二阶段独立功能模块验证

## 编译与代码质量
- [x] `cargo check` 无错误
- [x] `cargo build` 编译通过
- [x] `cargo clippy` 无实质性警告（仅 dead_code，符合自底向上开发预期）
- [x] `cargo test` 全部通过 (210 passed, 0 failed)，覆盖率 ≥ 80%

## Sink 模块
- [x] `SinkAdapter` trait 定义完整：`format()`/`open()`/`write_frame()`/`write_batch()`/`close()`
- [x] `OutputStats` 包含 `frames_written`/`bytes_written`/`output_path`/`file_hash` 字段
- [x] `OutputConfig` 包含 `compression_level`/`max_file_bytes`/`max_frames_per_file`/`compute_hash` 字段
- [x] `format_output_filename()` 产出的文件名包含 task_name/shard_id/seed/extension
- [x] `ParquetAdapter` 实现完整生命周期（open→write→close）
- [x] `TextAdapter` 实现 Unicode 字符映射和 BufWriter 行写出
- [x] `BinaryAdapter` 实现固定头格式和帧序列化
- [x] Parquet 写入→读取往返一致性测试通过
- [x] TextAdapter 输出为合法 UTF-8 测试通过
- [x] BinaryAdapter 文件头 magic "SGEN" 校验测试通过
- [x] 原子写入测试：关闭后无 `.tmp` 残留文件
- [x] 文件名格式唯一性测试通过

## Pipeline 模块
- [x] `Processor` trait 定义完整：`name()`/`process()`
- [x] `ProcessorRegistry` 实现 `new()`/`register()`/`get()`/`list_names()`
- [x] `NullProcessor` 透传测试通过
- [x] `Normalizer` 支持 Linear/LogBucket/UniformQuantile 三种方法
- [x] `DedupFilter` 支持连续重复去重、全零帧过滤、低熵过滤
- [x] `DiffEncoder` 支持相邻帧差分（Integer相减/Float相减/Bool XOR）
- [x] `TokenMapper` 支持整数值→Unicode 码点映射，钳位到安全范围
- [x] `ClipStitcher` 支持序列截断和分隔符插入
- [x] Normalizer Linear 缩放正确性测试通过
- [x] DedupFilter 连续相同帧移除测试通过
- [x] DiffEncoder 差分计算正确性测试通过
- [x] TokenMapper 输出码点在 Unicode 安全范围测试通过
- [x] 多处理器链式组合端到端测试通过
- [x] ProcessorRegistry 未知名称拒绝测试通过

## Generators 模块
- [x] `register_all()` 注册了全部 9 种生成器
- [x] `CellularAutomaton`：确定性测试通过、规则 30/110 验证通过、参数校验通过
- [x] `LorenzSystem`：确定性测试通过、吸引子形态基本正确、参数校验通过
- [x] `LogisticMap`：确定性测试通过、r 参数校验（[0,4]范围）通过
- [x] `NBodySim`：确定性测试通过、能量监控正常
- [x] `LSystem`：确定性测试通过
- [x] `AlgorithmVM`：确定性测试通过
- [x] `BooleanNetwork`：确定性测试通过
- [x] `IFS`：确定性测试通过
- [x] `FormalGrammar`：确定性测试通过
