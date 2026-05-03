# StructGen-rs 核心抽象层 (core) 详细设计规格

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| 1.0 | 2026-05-01 | - | 初始版本，定义所有公共数据类型与核心接口 |
| 1.1 | 2026-05-02 | - | 同步代码审查修复：移除 from_extensions、增加 Float 有限性约束、as_float 精度校验、LogLevel 枚举、validate 方法、register 返回 Result |

## 1. 模块概述

核心抽象层（core）是 StructGen-rs 系统的最底层模块，定义了整个系统中所有可共享的数据结构、核心 trait 接口和公共类型别名。它是其余所有模块（scheduler、generators、pipeline、sink、metadata、CLI）的共同依赖基础。

该模块的职责包括：
- 定义统一的帧数据容器（`FrameState`、`FrameData`、`SequenceFrame`），承载生成器产出的结构化状态数据。
- 定义通用参数载体（`GenParams`），解耦生成器特有配置与公共调度信息。
- 定义生成器抽象接口（`Generator` trait），作为所有生成器的行为契约。
- 定义贯穿系统的错误类型体系（`CoreError`、`CoreResult`），统一错误传播语义。
- 定义公共配置与格式枚举，供上层模块引用。

**核心原则**：core 模块不包含任何业务逻辑实现，仅定义类型与契约。它也不应依赖任何其他业务模块，形成纯粹的自底向上的单向依赖图。

## 2. 设计目标与原则

- **强类型抽象**：使用 Rust 的 enum 标记联合体统一表示整型、浮点型和布尔型状态值，避免 `Vec<u8>` 等弱类型承载方式带来的语义歧义。
- **零依赖接口**：core 模块仅依赖 Rust 标准库和极少数必需的基础 crate（如 `serde` 用于序列化），不引入任何领域特定库。
- **Send + Sync 保证**：所有 trait 对象必须标注 `Send + Sync`，确保可在线程池（rayon）中安全跨线程传递。
- **迭代器优先**：生成器主推流式 `Iterator` 接口，以惰性求值控制内存峰值；批量接口作为同步语法糖提供。
- **开放-封闭**：数据容器预留扩展字段（`extensions`），支持生成器特有参数而不修改核心接口。
- **错误收敛**：整个系统的错误类型收敛到统一的 `CoreError` 枚举，各上层模块可向其中追加变体。

## 3. 模块内部结构组织

core 模块的源码文件划分如下：

```
src/core/
├── mod.rs          # 模块根，重导出所有公开类型
├── frame.rs        # FrameState、FrameData、SequenceFrame 定义
├── params.rs       # GenParams、GlobalConfig 定义
├── generator.rs    # Generator trait 定义
├── error.rs        # CoreError、CoreResult 定义
└── registry.rs     # GeneratorRegistry 生成器注册表定义
```

各文件的职责：

| 文件 | 职责 |
|------|------|
| `frame.rs` | 定义帧数据的三种状态变体、帧数据结构、带时间步的完整帧结构 |
| `params.rs` | 定义通用参数载体、全局配置结构和输出格式枚举 |
| `generator.rs` | 定义 `Generator` trait 接口（流式/批量两种模式） |
| `error.rs` | 定义 `CoreError` 枚举（覆盖参数校验、生成、I/O 等错误类别） |
| `registry.rs` | 定义 `GeneratorRegistry` 结构体，管理名称→构造函数的映射 |
| `mod.rs` | 统一重导出所有公开类型，作为外部模块的导入入口 |

## 4. 公开接口定义

### 4.1 FrameState —— 帧状态值

```rust
use serde::{Deserialize, Serialize};

/// 精确可表示为 f64 的最大 i64 绝对值 (2^53 = 9007199254740992)
const EXACT_F64_MAX_I64: i64 = 1i64 << 53;

/// 单个状态值的标记联合体，统一承载整型、浮点型和布尔型数据。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FrameState {
    /// 有符号 64 位整数（可表示离散状态、符号索引等）
    Integer(i64),
    /// 64 位浮点数（连续系统的状态变量）
    ///
    /// # 不变量
    /// 该变体必须持有有限值（非 NaN、非 ±Infinity）。
    /// 请使用 [`FrameState::float`] 构造以确保该约束。
    Float(f64),
    /// 布尔值（二值网格元胞等）
    Bool(bool),
}

impl FrameState {
    /// 创建有限浮点数值，拒绝 NaN 和 ±Infinity。
    ///
    /// # Errors
    /// 当值为非有限数（NaN 或 ±Infinity）时返回 `CoreError::InvalidParams`。
    pub fn float(value: f64) -> Result<Self, CoreError> { /* ... */ }
    /// 尝试将值解释为 i64，失败返回 None。
    pub fn as_integer(&self) -> Option<i64> { /* ... */ }
    /// 尝试将值解释为 f64，失败返回 None。
    /// 对于 Integer 变体，仅当值在 [-2^53, 2^53] 范围内时返回 Some，
    /// 超出此范围的整数无法精确表示为 f64，返回 None。
    pub fn as_float(&self) -> Option<f64> { /* ... */ }
    /// 尝试将值解释为 bool，失败返回 None。
    pub fn as_bool(&self) -> Option<bool> { /* ... */ }
    /// 返回变体的判别名："Integer" / "Float" / "Bool"。
    pub fn variant_name(&self) -> &'static str { /* ... */ }
}
```

**设计说明**：
- 使用 `i64` 而非更小整数类型，是因为后处理管道中的标准化器会将浮点数映射到 0–255 或更大范围，`i64` 可直接承载且与 `f64` 同宽。
- `Float` 变体持有**有限性不变量**：NaN 和 ±Infinity 会导致 `serde_json` 将其序列化为 `null`，破坏 JSON 往返一致性。生成器应通过 `FrameState::float()` 受检构造函数创建浮点值，而非直接使用 `FrameState::Float(v)`。
- `as_float()` 对 `Integer` 变体执行**精确范围检查**：仅当 `|v| ≤ 2^53` 时返回 `Some(v as f64)`，超出此范围的整数无法无损转换为 f64，返回 `None` 以防止静默精度丢失。
- 提供 `as_` 方法族供下游模块（如 sink 的文本输出）安全取出原始值，避免模式匹配散落各处。

### 4.2 FrameData —— 帧状态向量

```rust
/// 一帧中所有状态值的集合，即一个样本在单个时间步上的完整状态快照。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameData {
    /// 状态值序列，顺序与生成器定义的状态维度一致。
    pub values: Vec<FrameState>,
}

impl FrameData {
    /// 创建空的帧数据。
    pub fn new() -> Self { /* ... */ }
    /// 状态维度（values 长度）。
    pub fn dim(&self) -> usize { self.values.len() }
    /// 判断是否为空帧。
    pub fn is_empty(&self) -> bool { self.values.is_empty() }
}

impl FromIterator<FrameState> for FrameData {
    fn from_iter<I: IntoIterator<Item = FrameState>>(iter: I) -> Self { /* ... */ }
}
```

**设计说明**：
- `FrameData` 实现标准 `FromIterator` trait，支持 `FrameData::from_iter([...])` 调用语法以及 `.collect::<FrameData>()` 惯用写法。

### 4.3 SequenceFrame —— 时序帧

```rust
/// 一个时间步的完整帧，包含步索引、状态数据和可选的语义标签。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceFrame {
    /// 时间步索引，从 0 开始递增。
    pub step_index: u64,
    /// 该时间步的完整状态快照。
    pub state: FrameData,
    /// 可选的语义标签，如 "period_2_detected"、"chaos_onset"、"sort_completed" 等，
    /// 用于后续文本融合训练或数据筛选。
    pub label: Option<String>,
}

impl SequenceFrame {
    /// 创建带标签的帧。
    pub fn with_label(step_index: u64, state: FrameData, label: impl Into<String>) -> Self { /* ... */ }
    /// 创建无标签的帧。
    pub fn new(step_index: u64, state: FrameData) -> Self { /* ... */ }
}
```

### 4.4 GenParams —— 通用参数

```rust
use std::collections::HashMap;
use serde_json::Value;

/// 生成器的通用参数载体，包含所有生成器共享的字段以及动态扩展。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenParams {
    /// 目标序列长度（要生成的帧数），0 表示无限制（由外部迭代决定）。
    pub seq_length: usize,
    /// 网格尺寸（对 CA、布尔网络等有空间维度概念的生成器），None 表示不适用。
    pub grid_size: Option<GridSize>,
    /// 动态扩展字段，承载生成器特有参数（以 JSON Value 形式存储），
    /// 生成器实例化时自行从此字段中反序列化自己的配置。
    pub extensions: HashMap<String, Value>,
}

/// 二维网格尺寸。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GridSize {
    pub rows: usize,
    pub cols: usize,
}

impl GenParams {
    /// 创建最简参数（仅指定序列长度）。
    pub fn simple(seq_length: usize) -> Self { /* ... */ }
    /// 从扩展字段中提取并反序列化生成器特有参数。
    pub fn get_extension<T: DeserializeOwned>(&self, key: &str) -> Result<T, CoreError> { /* ... */ }
    /// 向扩展字段中插入生成器特有参数。
    pub fn set_extension<T: Serialize>(&mut self, key: &str, value: &T) -> Result<(), CoreError> { /* ... */ }
}
```

### 4.5 输出格式、日志级别与全局配置

```rust
/// 输出文件格式枚举。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Apache Parquet 列式存储。
    #[default]
    Parquet,
    /// 纯文本（令牌映射后的 Unicode 序列）。
    Text,
    /// 内存映射二进制原始转储。
    Binary,
}

/// 日志级别枚举。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

/// 全局配置，适用于整个运行而非单个任务。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfig {
    /// 并行线程数，None 表示自动检测（= CPU 逻辑核心数）。
    pub num_threads: Option<usize>,
    /// 默认输出格式，可被任务级配置覆盖。
    pub default_format: OutputFormat,
    /// 输出根目录。
    pub output_dir: String,
    /// 日志级别。
    pub log_level: LogLevel,
    /// 每个输出分片文件的最大序列数，超出后自动切分。
    pub shard_max_sequences: usize,
    /// 流式写出模式（true） vs 阻塞收集模式（false）。
    pub stream_write: bool,
}

impl GlobalConfig {
    /// 校验配置的合法性。
    ///
    /// # Errors
    /// 当配置项不合法时返回 `CoreError::ConfigError`，包括：
    /// - `num_threads` 为 `Some(0)`
    /// - `output_dir` 为空字符串
    /// - `shard_max_sequences` 为 0
    pub fn validate(&self) -> Result<(), CoreError> { /* ... */ }
}

/// 二维网格尺寸。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridSize {
    pub rows: usize,
    pub cols: usize,
}

impl GridSize {
    /// 校验网格尺寸的合法性。
    ///
    /// # Errors
    /// 当 rows 或 cols 为 0 时返回 `CoreError::InvalidParams`。
    pub fn validate(&self) -> Result<(), CoreError> { /* ... */ }
}
```

**设计说明**：
- `LogLevel` 采用枚举而非自由字符串，在类型层面排除无效日志级别（如 `"banana"`），`#[serde(rename_all = "lowercase")]` 使 JSON 中为小写形式（`"trace"`、`"debug"` 等），与 YAML 配置文件中的习惯一致。
- `GlobalConfig::validate()` 和 `GridSize::validate()` 提供显式校验入口，应在 CLI 参数解析和调度器构造时调用，将配置错误尽早暴露。

### 4.6 Generator trait —— 生成器接口

```rust
/// 生成器抽象接口。所有具体生成器必须实现此 trait。
///
/// 该 trait 要求实现者为 `Send + Sync`，确保实例可在 rayon 线程间安全共享。
///
/// 生成器的构造不通过 trait 方法完成，而是通过注册在 `GeneratorRegistry`
/// 中的工厂函数进行。每个生成器模块应提供一个符合 `GeneratorFactory`
/// 签名的构造函数，并在注册表中注册。
pub trait Generator: Send + Sync {
    /// 返回该生成器的唯一标识名称，如 "cellular_automaton"、"lorenz_system"。
    fn name(&self) -> &'static str;

    /// 流式生成（推荐模式）。返回一个惰性迭代器，按时间步产出 `SequenceFrame`。
    ///
    /// 迭代器在 `Some(seq_length)` 时最多产出 `seq_length` 个帧；
    /// 当 `seq_length == 0` 时无限产出，由调用方控制消费数量。
    ///
    /// # 生命周期约束
    /// 返回的迭代器具有 `'static` 生命周期，即不能借用 `&self` 或 `params`。
    /// 生成器实现需将必要状态克隆或移入迭代器闭包中。该约束使得迭代器
    /// 可安全地跨 rayon 线程传递，无需关心借用的生命周期。
    ///
    /// # Arguments
    /// * `seed` - 确定性随机种子，决定初始状态和随机性注入。
    /// * `params` - 通用参数，从中提取 `seq_length` 等共用信息。
    fn generate_stream(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> Result<Box<dyn Iterator<Item = SequenceFrame> + Send>, CoreError>;

    /// 批量生成（同步糖）。内部调用 `generate_stream` 并收集为 `Vec`。
    /// 仅适用于中小规模数据；大规模请使用流式接口。
    /// 要求 `params.seq_length > 0`，否则返回错误（因为无限流无法收集为 Vec）。
    ///
    /// # Arguments
    /// * `seed` - 确定性随机种子。
    /// * `params` - 通用参数。
    ///
    /// # Errors
    /// 当 `params.seq_length == 0` 时返回 `CoreError::InvalidParams`。
    fn generate_batch(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> Result<Vec<SequenceFrame>, CoreError> {
        if params.seq_length == 0 {
            return Err(CoreError::InvalidParams(
                "generate_batch requires seq_length > 0; \
                 use generate_stream for unbounded generation".into(),
            ));
        }
        self.generate_stream(seed, params)
            .map(|iter| iter.collect())
    }
}
```

**接口说明**：
- `generate_stream` 是核心方法，`generate_batch` 有默认实现作为语法糖。
- 迭代器的 `Item` 是 `SequenceFrame`（非 `Result<SequenceFrame, _>`），因为生成过程中的错误应在迭代器内部处理为提前终止（返回 `None`）或通过 `Result` 在外层捕获。
- 迭代器标注 `+ Send`，确保可跨线程传输。返回的迭代器具有 `'static` 生命周期，实现者需将所需状态克隆到闭包中，而非借用 `&self`。
- `generate_batch` 在 `seq_length == 0` 时返回 `InvalidParams` 错误，防止对无限流调用 `.collect()` 导致 OOM。
- 生成器的构造**不通过 trait 方法**完成（已移除 `from_extensions`），而是统一通过 `GeneratorRegistry` 中的工厂函数进行。这消除了 trait 静态方法（`where Self: Sized`）与 trait 对象（`dyn Generator`）之间的割裂。

### 4.7 GeneratorRegistry —— 生成器注册表

```rust
/// 生成器构造函数类型：接受 extensions 映射，返回 Generator trait 对象。
pub type GeneratorFactory = fn(&HashMap<String, Value>) -> CoreResult<Box<dyn Generator>>;

/// 生成器的全局注册表。采用名称→构造函数的静态映射。
///
/// 所有生成器在程序启动时通过 `register` 方法注册自身。
/// 调度器通过 `instantiate` 方法按名称查找构造函数并实例化生成器。
#[derive(Default)]
pub struct GeneratorRegistry {
    factories: HashMap<&'static str, GeneratorFactory>,
}

impl GeneratorRegistry {
    /// 创建空的注册表。
    pub fn new() -> Self { /* ... */ }

    /// 注册一个生成器构造函数。
    ///
    /// # Errors
    /// 当名称已存在时返回 `CoreError::InvalidParams`，防止重复注册导致静默覆盖。
    pub fn register(&mut self, name: &'static str, factory: GeneratorFactory) -> CoreResult<()> { /* ... */ }

    /// 按名称实例化一个生成器。
    ///
    /// # Errors
    /// 当名称未注册时返回 `CoreError::GeneratorNotFound`。
    pub fn instantiate(
        &self,
        name: &str,
        extensions: &HashMap<String, Value>,
    ) -> CoreResult<Box<dyn Generator>> { /* ... */ }

    /// 列出所有已注册的生成器名称。
    pub fn list_names(&self) -> Vec<&'static str> { /* ... */ }

    /// 判断某个名称是否已注册。
    pub fn contains(&self, name: &str) -> bool { /* ... */ }
}
```

**设计说明**：
- `register()` 返回 `CoreResult<()>` 而非 panic，使调用方可以在运行时决定如何处理重复注册（如记录警告、跳过或终止），而非直接崩溃。对于插件化场景尤为关键。
- `GeneratorFactory` 和 `instantiate()` 统一使用 `CoreResult<T>` 类型别名，与全系统错误类型一致。

### 4.8 错误类型定义

```rust
use thiserror::Error;

/// 核心错误类型，覆盖系统所有可预见的失败场景。
#[derive(Error, Debug)]
pub enum CoreError {
    /// 参数不合法（缺失、越界、类型不匹配等）。
    #[error("invalid parameter: {0}")]
    InvalidParams(String),

    /// 未找到指定名称的生成器。
    #[error("generator not found: {0}")]
    GeneratorNotFound(String),

    /// 生成器实例化过程中发生的错误。
    #[error("generator initialization failed: {0}")]
    GeneratorInitError(String),

    /// 生成过程中发生的错误（如数值发散、内部状态损坏）。
    #[error("generation error: {0}")]
    GenerationError(String),

    /// I/O 错误（文件写入、读取、创建目录等）。
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON 序列化/反序列化错误。
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// YAML 清单解析错误。
    #[error("manifest parse error: {0}")]
    ManifestError(String),

    /// 后处理管道中的错误。
    #[error("pipeline error: {0}")]
    PipelineError(String),

    /// 输出适配器中的错误。
    #[error("sink error: {0}")]
    SinkError(String),

    /// 配置错误（如线程数设为 0）。
    #[error("configuration error: {0}")]
    ConfigError(String),

    /// 其他未分类错误。
    #[error("{0}")]
    Other(String),
}

/// 核心结果类型别名，全系统统一使用。
pub type CoreResult<T> = Result<T, CoreError>;
```

## 5. 核心逻辑详解

### 5.1 FrameState 的值转换逻辑

```
FrameState::Integer(v) → as_integer() → Some(v)
                       → as_float()   → Some(v as f64)   仅当 |v| ≤ 2^53
                                        None              当 |v| > 2^53（精度丢失风险）
                       → as_bool()    → Some(v != 0)

FrameState::Float(v)   → as_integer() → None              // 有损，不自动转换
                       → as_float()   → Some(v)
                       → as_bool()    → None

FrameState::Bool(v)    → as_integer() → Some(v as i64)    // 0 或 1
                       → as_float()   → Some(v as u8 as f64)
                       → as_bool()    → Some(v)
```

设计原则：
- 整数→浮点：条件安全，仅当值在 f64 精确表示范围（`[-2^53, 2^53]`）内时转换；超出则返回 `None`，防止静默精度丢失。
- 浮点→整数：不自动转换，防止精度静默丢失（需要标准化器显式处理）。
- 布尔→数值：安全，0/1 映射。
- Float 有限性：`FrameState::float()` 构造函数拒绝 NaN 和 ±Infinity，确保 `serde_json` 序列化的 JSON 往返一致性。直接使用 `FrameState::Float(v)` 绕过检查时，非有限值将被 `serde_json` 序列化为 `null`，反序列化时会失败。

### 5.2 GeneratorRegistry 的注册与查找流程

```
程序启动
  ↓
各生成器模块调用 registry.register("ca", ca_factory)?
  ↓ 重复名称时返回 Err(InvalidParams)，调用方可选择终止或跳过
[HashMap<&str, GeneratorFactory> 建立完毕]
  ↓
Scheduler 调用 registry.instantiate("ca", extensions)
  ↓
查找 factories["ca"] → 存在 → 调用 factory(extensions) → Ok(Box<dyn Generator>)
                      → 不存在 → Err(GeneratorNotFound("ca"))
```

**注意**：`register()` 返回 `CoreResult<()>`，不再 panic。调用方应在注册阶段处理可能的重复名称错误。

### 5.3 GenParams 扩展字段的序列化协议

扩展字段使用 JSON 作为中间表示，生成器通过 `get_extension::<T>()` 和 `set_extension::<T>()` 完成序列化/反序列化：

```
YAML 清单中:
  params:
    extensions:
      ca:
        rule: 110
        boundary: periodic
        initial_state: random

GenParams.extensions["ca"] = Value::Object({...})
  ↓
registry.instantiate("ca", extensions) → 调用 ca_factory(extensions)
  ↓
ca_factory 内部：
  let ca_config: CAConfig = serde_json::from_value(extensions["ca"].clone())?;
  Ok(Box::new(CAGenerator::new(ca_config)))
```

## 6. 与其他模块的交互

### 6.1 依赖关系图

```
core (本模块)
 ↑
 ├── scheduler   ─ 使用 Generator trait、GenParams、GeneratorRegistry、CoreError
 ├── generators  ─ 实现 Generator trait、使用 FrameState/SequenceFrame/GenParams
 ├── pipeline    ─ 使用 SequenceFrame、CoreError（待定义 Processor trait 时补充）
 ├── sink        ─ 使用 SequenceFrame、OutputFormat、CoreError
 ├── metadata    ─ 使用 GenParams、CoreError
 └── CLI         ─ 使用 GlobalConfig、OutputFormat、CoreError
```

### 6.2 core 不依赖任何其他业务模块

core 模块仅依赖于：
- `std`（Rust 标准库）
- `serde` + `serde_json`（序列化）
- `thiserror`（错误派生宏）

这些依赖不引入领域逻辑，保证了 core 模块的纯粹性。

### 6.3 各模块对 core 的具体使用

| 使用方模块 | 使用的 core 类型 | 使用方式 |
|------------|-----------------|----------|
| scheduler | `GeneratorRegistry`、`Generator` trait、`GenParams` | 查表实例化生成器、构造任务参数 |
| generators（CA等） | `Generator` trait、`FrameState`、`SequenceFrame`、`GenParams` | 实现 trait，产出帧序列 |
| pipeline | `SequenceFrame`、`CoreError` | 消费/产出帧迭代器 |
| sink | `SequenceFrame`、`OutputFormat`、`CoreError` | 按格式写出帧数据 |
| metadata | `GenParams`、`CoreError` | 记录参数快照 |
| CLI | `GlobalConfig`、`OutputFormat`、`GeneratorRegistry` | 解析参数、装配流水线 |

## 7. 错误处理策略

### 7.1 错误分类

| 错误类别 | 对应变体 | 触发场景 | 处理方式 |
|----------|----------|----------|----------|
| 用户输入错误 | `InvalidParams`、`ConfigError`、`ManifestError` | 参数不合法、配置错误 | 立即报错退出，给出明确提示 |
| 资源错误 | `GeneratorNotFound` | 清单引用了未注册的生成器 | 立即报错，列出可用生成器 |
| 运行时错误 | `GenerationError`、`PipelineError`、`SinkError` | 生成或写出异常 | 记录日志，可选丢弃当前样本 |
| 系统错误 | `IoError`、`SerializationError` | 磁盘满、文件权限等 | 清理资源后报错退出 |
| 初始化错误 | `GeneratorInitError` | 生成器构造失败 | 立即报错，给出参数校验信息 |

### 7.2 错误传播链

```
具体生成器内部错误
  → CoreError::GenerationError(msg)
    → generate_stream() 返回 Err
      → Scheduler 捕获，记录到元数据
        → 决定是跳过该分片还是终止运行
```

## 8. 性能考量

- **FrameState 内存布局**：`enum FrameState` 的大小为 16 字节（8 字节判别码 + 8 字节负载），与 `i64` 对齐。对于百万帧级序列，内存占用可控。
- **零拷贝传递**：`Generator::generate_stream` 返回的迭代器按值产出 `SequenceFrame`（move 语义），下游模块可直接消费而不产生额外克隆。
- **注册表查找**：`HashMap<&str, GeneratorFactory>` 使用静态字符串键，查找为 O(1)。
- **扩展字段解析**：采用惰性解析策略——生成器实例化时才从 JSON 反序列化配置，而非预先解析整个扩展字段。避免无效解析开销。

## 9. 可测试性设计

### 9.1 单元测试

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_state_conversions() {
        let int_val = FrameState::Integer(42);
        assert_eq!(int_val.as_integer(), Some(42));
        assert_eq!(int_val.as_float(), Some(42.0));
        assert_ne!(int_val.as_bool(), None);

        let float_val = FrameState::Float(3.14);
        assert_eq!(float_val.as_integer(), None); // 不自动转换
        assert_eq!(float_val.as_float(), Some(3.14));
    }

    #[test]
    fn test_frame_state_float_checked_constructor() {
        assert!(FrameState::float(3.14).is_ok());
        assert!(FrameState::float(f64::NAN).is_err());
        assert!(FrameState::float(f64::INFINITY).is_err());
    }

    #[test]
    fn test_as_float_large_integer() {
        // 在 2^53 范围内
        assert!(FrameState::Integer(1i64 << 53).as_float().is_some());
        // 超出 2^53 范围
        assert!(FrameState::Integer((1i64 << 53) + 1).as_float().is_none());
    }

    #[test]
    fn test_gen_params_extension_roundtrip() {
        let mut params = GenParams::simple(100);
        params.set_extension("rule", &110u32).unwrap();
        let rule: u32 = params.get_extension("rule").unwrap();
        assert_eq!(rule, 110);
    }

    #[test]
    fn test_registry_register_and_instantiate() {
        let mut reg = GeneratorRegistry::new();
        reg.register("mock", mock_factory).unwrap();
        assert!(reg.instantiate("mock", &HashMap::new()).is_ok());
    }

    #[test]
    fn test_registry_name_not_found() {
        let reg = GeneratorRegistry::new();
        let result = reg.instantiate("nonexistent", &HashMap::new());
        assert!(matches!(result, Err(CoreError::GeneratorNotFound(_))));
    }

    #[test]
    fn test_registry_duplicate_returns_error() {
        let mut reg = GeneratorRegistry::new();
        reg.register("mock", mock_factory).unwrap();
        let result = reg.register("mock", mock_factory);
        assert!(matches!(result, Err(CoreError::InvalidParams(_))));
    }

    #[test]
    fn test_generator_batch_zero_length_rejected() {
        let gen = MockGenerator;
        let params = GenParams::simple(0);
        assert!(matches!(
            gen.generate_batch(0, &params),
            Err(CoreError::InvalidParams(_))
        ));
    }

    #[test]
    fn test_global_config_validate() {
        let config = GlobalConfig {
            output_dir: "./out".into(),
            ..Default::default()
        };
        assert!(config.validate().is_ok());
        // 无效：num_threads = Some(0)
        let bad = GlobalConfig { num_threads: Some(0), ..config.clone() };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_log_level_deserialization() {
        let level: LogLevel = serde_json::from_value(json!("debug")).unwrap();
        assert_eq!(level, LogLevel::Debug);
        // 无效字符串
        assert!(serde_json::from_value::<LogLevel>(json!("banana")).is_err());
    }
}
```

### 9.2 集成测试验证点

- 使用 core 类型构建一个最小化的 mock 生成器，验证 trait 接口可被正常实现。
- 验证 `GeneratorRegistry` 的并发安全性（多线程同时读取）。
- 验证 JSON 扩展字段对各类参数类型的序列化/反序列化正确性。
- 验证 `FrameState::Float(NaN)` 经 serde_json 序列化为 `null` 后，反序列化失败（确认有限性约束的必要性）。
- 验证 `GlobalConfig::validate()` 对各非法配置项的拦截。

## 10. 配置与参数

core 模块自身不读取配置文件中的参数。它定义的数据类型被其他模块引用如下：

| 类型 | 被引用位置 | 配置来源 |
|------|-----------|----------|
| `GlobalConfig` | CLI 模块（解析命令行参数） | 命令行参数 + 默认值 |
| `OutputFormat` | CLI 模块、Scheduler 模块 | 命令行 `--format` 参数 |
| `LogLevel` | CLI 模块、日志初始化 | 命令行 `--log-level` 参数 |
| `GenParams` | Scheduler 模块（构造任务参数） | YAML 清单中的任务定义 |
| `GridSize` | 各生成器模块 | GenParams.grid_size 或 extensions |

---

通过以上设计，核心抽象层为 StructGen-rs 提供了稳固的类型基础。所有上层模块仅需依赖 core 模块定义的接口，无需了解任何具体实现细节，实现了清晰的架构分层。
