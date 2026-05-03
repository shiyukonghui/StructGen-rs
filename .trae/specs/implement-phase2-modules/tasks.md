# Tasks: 实现第二阶段独立功能模块

## 依赖准备

- [x] Task 0: 更新 Cargo.toml 添加 Phase 2 依赖
  - [x] 添加 `arrow = "53"` (sink 模块依赖，升级以兼容 chrono 0.4.44)
  - [x] 添加 `parquet = "53"` (sink 模块依赖，features=["arrow"])
  - [x] 添加 `tempfile = "3"` (dev-dependencies，测试用)
  - [x] 运行 `cargo check` 确认新依赖拉取成功

## 模块 A: sink（输出适配层）

- [x] Task A1: 实现 sink 模块基础类型 (adapter.rs + factory.rs)
  - [x] 创建 `src/sink/adapter.rs`：定义 `SinkAdapter` trait、`OutputStats`、`OutputConfig`、`format_output_filename()`
  - [x] 创建 `src/sink/factory.rs`：定义 `SinkAdapterFactory` 类型别名
  - [x] 编写单元测试：`format_output_filename` 唯一性
  - [x] 运行 `cargo test sink` 确认测试通过

- [x] Task A2: 实现 ParquetAdapter (parquet.rs)
  - [x] 创建 `src/sink/parquet.rs`：实现 `ParquetAdapter` 结构体
  - [x] 构建 Arrow Schema（step_index/state_dim/state_values/label）
  - [x] 实现 `open()`/`write_frame()`/`close()` 生命周期
  - [x] 支持 Snappy 压缩写出
  - [x] 编写 Parquet 写入→读取往返测试
  - [x] 运行 `cargo test sink` 确认测试通过

- [x] Task A3: 实现 TextAdapter (text.rs)
  - [x] 创建 `src/sink/text.rs`：实现 `TextAdapter` 结构体
  - [x] 实现 Unicode 字符映射逻辑（Integer→char、Float→0-255映射为char、Bool→'0'/'1'）
  - [x] 使用 BufWriter 行写出，每帧一行
  - [x] 编写 Text 输出为合法 UTF-8 测试
  - [x] 运行 `cargo test sink` 确认测试通过

- [x] Task A4: 实现 BinaryAdapter (binary.rs)
  - [x] 创建 `src/sink/binary.rs`：实现 `BinaryAdapter` 结构体
  - [x] 实现固定头格式（magic "SGEN" + version + frame_count + state_dim）
  - [x] 实现帧序列化（step_index + type_tag + data + label_len + label）
  - [x] 实现 close() 时回填 frame_count
  - [x] 编写 Binary 文件头 magic 校验测试
  - [x] 运行 `cargo test sink` 确认测试通过

- [x] Task A5: 实现 sink 模块根和完整性验证
  - [x] 创建 `src/sink/mod.rs`：声明子模块，重导出公开类型
  - [x] 编写原子写入测试（关闭后无 .tmp 残留）
  - [x] 运行全量 `cargo test sink` 确认全部 sink 测试通过

## 模块 B: pipeline（后处理管道层）

- [x] Task B1: 实现 Processor trait 和 ProcessorRegistry
  - [x] 创建 `src/pipeline/processor.rs`：定义 `Processor` trait、`ProcessorFactory` 类型别名
  - [x] 创建 `src/pipeline/registry.rs`：实现 `ProcessorRegistry`
  - [x] 编写测试：ProcessorRegistry 未知名称拒绝
  - [x] 运行 `cargo test pipeline` 确认测试通过

- [x] Task B2: 实现 NullProcessor (null_proc.rs)
  - [x] 创建 `src/pipeline/null_proc.rs`：实现 `NullProcessor`（透传处理器）
  - [x] 编写测试验证透传行为
  - [x] 运行 `cargo test pipeline` 确认测试通过

- [x] Task B3: 实现 Normalizer (normalizer.rs)
  - [x] 创建 `src/pipeline/normalizer.rs`：实现 `Normalizer` 结构体
  - [x] 实现 Linear/LogBucket/UniformQuantile 三种标准化方法
  - [x] 支持两遍扫描（自动计算 min/max）或配置指定边界
  - [x] 编写 Linear 缩放正确性测试
  - [x] 运行 `cargo test pipeline` 确认测试通过

- [x] Task B4: 实现 DedupFilter (dedup.rs)
  - [x] 创建 `src/pipeline/dedup.rs`：实现 `DedupFilter` 结构体
  - [x] 实现连续重复去重、全零帧过滤、低熵过滤（香农熵估计）
  - [x] 编写连续相同帧移除测试
  - [x] 运行 `cargo test pipeline` 确认测试通过

- [x] Task B5: 实现 DiffEncoder (diff_encoder.rs)
  - [x] 创建 `src/pipeline/diff_encoder.rs`：实现 `DiffEncoder` 结构体
  - [x] 实现相邻帧差分计算（Integer 相减、Float 相减、Bool XOR）
  - [x] 支持首帧零参考选项
  - [x] 编写差分计算正确性测试
  - [x] 运行 `cargo test pipeline` 确认测试通过

- [x] Task B6: 实现 TokenMapper (token_mapper.rs)
  - [x] 创建 `src/pipeline/token_mapper.rs`：实现 `TokenMapper` 结构体
  - [x] 实现整数值→Unicode 码点映射，钳位到安全范围
  - [x] 支持可选换行符分隔
  - [x] 编写输出码点在 Unicode 安全范围测试
  - [x] 运行 `cargo test pipeline` 确认测试通过

- [x] Task B7: 实现 ClipStitcher (clip_stitcher.rs)
  - [x] 创建 `src/pipeline/clip_stitcher.rs`：实现 `ClipStitcher` 结构体
  - [x] 实现序列截断（按 max_len 切分）和分隔符插入
  - [x] 编写序列截断测试
  - [x] 运行 `cargo test pipeline` 确认测试通过

- [x] Task B8: 实现 pipeline 模块根和链式组合测试
  - [x] 创建 `src/pipeline/mod.rs`：声明子模块，实现 `register_all()`，重导出
  - [x] 编写多处理器链式组合端到端测试（normalize→dedup→diff→token_map）
  - [x] 运行全量 `cargo test pipeline` 确认全部 pipeline 测试通过

## 模块 C: generators（生成器仓储）

- [x] Task C1: 实现 generators 模块根和注册入口
  - [x] 创建 `src/generators/mod.rs`：声明子模块，实现 `register_all()` 注册所有生成器

- [x] Task C2: 实现 CellularAutomaton 生成器 (ca.rs)
  - [x] 创建 `src/generators/ca.rs`：1D CA 实现
  - [x] 支持 Wolfram 规则（0-255），规则编译为 LUT
  - [x] 支持周期/固定/反射三种边界条件
  - [x] 双缓冲无分配迭代
  - [x] 编写确定性测试（固定种子→固定输出）和规则 30/110 经典模式验证
  - [x] 编写参数校验测试（非法规则号）
  - [x] 运行 `cargo test generators` 确认测试通过

- [x] Task C3: 实现 LorenzSystem 生成器 (lorenz.rs)
  - [x] 创建 `src/generators/lorenz.rs`：洛伦兹系统实现
  - [x] RK4 积分，参数化 σ/ρ/β，默认值 σ=10, ρ=28, β=8/3
  - [x] 编写确定性测试和吸引子形态验证
  - [x] 编写参数校验测试
  - [x] 运行 `cargo test generators` 确认测试通过

- [x] Task C4: 实现 LogisticMap 生成器 (logistic.rs)
  - [x] 创建 `src/generators/logistic.rs`：逻辑斯蒂映射
  - [x] 参数 r 可配置（默认 3.9），x₀ 由种子 PRNG 生成
  - [x] 编写确定性测试和周期窗口验证
  - [x] 编写参数校验测试（r 超出 [0,4] 报错）
  - [x] 运行 `cargo test generators` 确认测试通过

- [x] Task C5: 实现 NBodySim 生成器 (nbody.rs)
  - [x] 创建 `src/generators/nbody.rs`：多体引力模拟
  - [x] 支持显式欧拉和 RK4 两种积分器
  - [x] 软化因子避免引力奇点
  - [x] 编写确定性测试和能量监控
  - [x] 运行 `cargo test generators` 确认测试通过

- [x] Task C6: 实现 LSystem 生成器 (lsystem.rs)
  - [x] 创建 `src/generators/lsystem.rs`：L-System 实现
  - [x] 支持上下文无关重写规则
  - [x] 可选龟图解释器
  - [x] 编写确定性测试
  - [x] 运行 `cargo test generators` 确认测试通过

- [x] Task C7: 实现 AlgorithmVM 生成器 (vm.rs)
  - [x] 创建 `src/generators/vm.rs`：精简栈式虚拟机
  - [x] 随机程序生成，最大步数保护防无限循环
  - [x] 编写确定性测试
  - [x] 运行 `cargo test generators` 确认测试通过

- [x] Task C8: 实现 BooleanNetwork 生成器 (boolean_network.rs)
  - [x] 创建 `src/generators/boolean_network.rs`：随机布尔网络
  - [x] 编写确定性测试
  - [x] 运行 `cargo test generators` 确认测试通过

- [x] Task C9: 实现 IFS 生成器 (ifs.rs)
  - [x] 创建 `src/generators/ifs.rs`：迭代函数系统
  - [x] 编写确定性测试
  - [x] 运行 `cargo test generators` 确认测试通过

- [x] Task C10: 实现 FormalGrammar 生成器 (formal_grammar.rs)
  - [x] 创建 `src/generators/formal_grammar.rs`：上下文无关文法
  - [x] 编写确定性测试
  - [x] 运行 `cargo test generators` 确认测试通过

- [x] Task C11: generators 模块全量验证
  - [x] 运行全量 `cargo test generators` 确认全部生成器测试通过
  - [x] 验证 `register_all()` 注册了所有 9 种生成器

## 最终集成验证

- [x] Task Z1: 运行全量测试
  - [x] `cargo test` 全部通过 (210 passed, 0 failed)
  - [x] `cargo clippy` 无实质性警告（仅 dead_code，符合自底向上开发预期）
  - [x] `cargo build` 编译通过

# Task Dependencies

- Task 0 (Cargo.toml) 必须先完成
- 模块 A (sink)、模块 B (pipeline)、模块 C (generators) 相互独立，可并行开发
- Task A2/A3/A4 依赖于 Task A1（trait 定义）
- Task B2~B7 依赖于 Task B1（trait 和 registry）
- Task B8 依赖于 Task B1~B7 全部完成
- Task C2~C10 依赖于 Task C1（mod.rs）
- Task Z1 依赖于所有模块完成
