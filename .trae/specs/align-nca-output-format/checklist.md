# Checklist

## Task 1: 温度参数默认值修改
- [x] `default_temperature()` 函数返回值改为 `0.0`
- [x] `test_default_temperature_is_one` 测试更新为验证 `0.0`
- [x] 所有 NCA 相关测试通过

## Task 2: SequenceStitcher 处理器
- [x] `SequenceStitcherConfig` 结构体包含 `frames_per_sequence`、`add_sequence_start`、`add_sequence_end` 字段
- [x] `SequenceStitcher` 实现 `Processor` trait
- [x] 序列串联逻辑正确：输出格式为 `[start][frame0_patches][end][start][frame1_patches][end]...`
- [x] 处理器可通过 `ProcessorRegistry` 实例化
- [x] 单元测试覆盖边界情况（空输入、单帧、多帧）

## Task 3: BatchCollector 处理器
- [x] `BatchCollectorConfig` 结构体包含 `batch_size`、`num_frames` 字段
- [x] `BatchCollector` 实现 `Processor` trait
- [x] `BatchData` 结构体包含 `data: Vec<Vec<SequenceFrame>>` 或张量形式
- [x] 批量收集逻辑正确：输出形状为 `(batch_size, num_frames, H, W, C)`
- [x] 处理器可通过 `ProcessorRegistry` 实例化
- [x] 单元测试覆盖边界情况

## Task 4: NpyBatchSink 输出器
- [x] `NpyBatchSinkConfig` 结构体包含 `output_path` 字段
- [x] `NpyBatchSink` 实现 `Sink` trait（或相应接口）
- [x] NumPy 文件格式正确：可被 `numpy.load()` 加载
- [x] 输出数据类型为 `int32` 或 `int64`（匹配 Python JAX array）
- [x] 单元测试验证输出文件格式

## Task 5: 集成测试与验证
- [x] 端到端测试：NCA 生成 → PatchTokenizer → SequenceStitcher → BatchCollector → NpyBatchSink
- [x] Python 脚本验证：生成的 `.npy` 文件可正确加载
- [x] 数据形状验证：`(B, T, H, W, C)` 格式正确
- [x] 文档更新：新增处理器和输出器的使用说明