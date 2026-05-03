# 第二阶段：独立功能模块 Spec

## Why
Phase 1 (core 模块) 已完成，定义了系统所需的全部公共类型与接口契约。Phase 2 需要基于这些契约，实现三个完全独立、相互无依赖的功能模块：输出适配层 (sink)、后处理管道层 (pipeline) 和生成器仓储 (generators)。这三个模块可以并行开发。

## What Changes
- 新增 `src/sink/` 模块，实现 `SinkAdapter` trait 及 Parquet/Text/Binary 三种适配器
- 新增 `src/pipeline/` 模块，实现 `Processor` trait 及 Normalizer/DedupFilter/DiffEncoder/TokenMapper/ClipStitcher 五种处理器
- 新增 `src/generators/` 模块，实现 9 种生成器（CA/Lorenz/LogisticMap/NBody/LSystem/AlgorithmVM/BooleanNetwork/IFS/FormalGrammar）
- 更新 `Cargo.toml` 添加 arrow、parquet、tempfile 等新依赖

## Impact
- Affected specs: 基础 core 模块
- Affected code: `Cargo.toml`, `src/sink/`, `src/pipeline/`, `src/generators/`

## ADDED Requirements

### Requirement: Sink 输出适配层
系统 SHALL 提供统一的输出适配器接口 (`SinkAdapter` trait)，支持 Parquet、Text、Binary 三种输出格式，以及原子写入和分片文件命名。

#### Scenario: Parquet 写入→读取往返一致
- **WHEN** 使用 ParquetAdapter 写入一组 SequenceFrame 后读取
- **THEN** 读取的帧数据与原始帧数据完全一致

#### Scenario: TextAdapter 输出为合法 UTF-8
- **WHEN** 使用 TextAdapter 写入帧数据
- **THEN** 输出文件可被 `std::fs::read_to_string` 正确读取，无非法字节序列

#### Scenario: BinaryAdapter 文件头 magic 校验
- **WHEN** 使用 BinaryAdapter 写入帧数据
- **THEN** 输出文件的前 4 字节为 `b"SGEN"`

#### Scenario: 原子写入
- **WHEN** 适配器正常关闭
- **THEN** 输出目录中不存在 `.tmp` 残留文件，仅有最终文件

#### Scenario: 文件名格式唯一性
- **WHEN** 使用不同 shard_id 生成输出文件名
- **THEN** 产生的文件名互不相同

### Requirement: Pipeline 后处理管道层
系统 SHALL 提供可组合的后处理器接口 (`Processor` trait)，支持标准化、去重、差分编码、令牌映射、序列截断等变换。

#### Scenario: Normalizer Linear 缩放正确性
- **WHEN** 对含 Float 值的帧应用 Linear 标准化
- **THEN** 输出值全部在 [0, max_val] 范围内

#### Scenario: DedupFilter 连续相同帧移除
- **WHEN** 输入包含连续相同帧
- **THEN** 输出中不包含连续重复帧

#### Scenario: DiffEncoder 差分计算正确
- **WHEN** 输入相邻两帧值 [10.0, 12.0]
- **THEN** 差分后第二帧值为 Float(2.0)

#### Scenario: TokenMapper 输出码点在 Unicode 安全范围
- **WHEN** 对整数值帧应用令牌映射
- **THEN** 所有输出值对应的 Unicode 码点 ≤ 0x10FFFF

#### Scenario: 多处理器链式组合
- **WHEN** 将多个处理器链式串联处理帧流
- **THEN** 最终产出正确变换后的帧序列

#### Scenario: ProcessorRegistry 未知名称拒绝
- **WHEN** 查询未注册的处理器名称
- **THEN** 返回 Error

### Requirement: Generators 生成器仓储
系统 SHALL 提供统一的生成器注册机制和 9 种内建生成器，所有生成器满足确定性要求（固定种子→固定输出）。

#### Scenario: 确定性验证
- **WHEN** 使用固定种子调用同一生成器两次
- **THEN** 两次产出的前 N 帧二进制完全一致

#### Scenario: CA 规则 30/110 经典模式
- **WHEN** 创建规则 30 的 CA 生成器
- **THEN** 产出符合 Wolfram Rule 30 的经典模式（至少验证前 10 步特定元胞值）

#### Scenario: 洛伦兹系统吸引子形态
- **WHEN** 使用默认参数 (σ=10, ρ=28, β=8/3) 运行洛伦兹系统
- **THEN** 轨迹在有限范围内演化，不出奇异值

#### Scenario: 参数校验
- **WHEN** 传入非法规则号（如 256 以上的一维 CA 规则号）
- **THEN** 生成器初始化返回 Error
