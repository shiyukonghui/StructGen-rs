# StructGen-rs 后处理管道层 (pipeline) 详细设计规格

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| 1.0 | 2026-05-01 | - | 初始版本，定义处理器接口与内置处理器逻辑 |

## 1. 模块概述

后处理管道层（pipeline）是 StructGen-rs 系统中的数据清洗与变换层。它接收生成器产出的原始帧序列，通过可级联的处理器链对数据进行标准化、去重、令牌映射等变换，最终产出适配下游语言模型训练的规范格式。

该模块的职责包括：
- 定义统一的处理器接口（`Processor` trait），使所有后处理步骤可自由组合。
- 维护处理器注册表（`ProcessorRegistry`），支持按名称查找和实例化处理器。
- 实现核心的内置处理器：标准化器、去重过滤器、差分编码器、令牌映射器、序列截断与拼接器。
- 支持惰性迭代器链式调用，全程流式处理不物化中间结果。

**核心原则**：管道层不改变数据的内容语义，只改变表示形式（数值范围、编码方式、冗余消除）。它是单向变换的序列。

## 2. 设计目标与原则

- **可组合性**：处理器以迭代器适配器的形式实现，接受 `Iterator<Item = SequenceFrame>` 并返回同类型迭代器。任意多个处理器可链式串联。
- **惰性求值**：每个处理器是一个惰性适配器。直到数据被消费端拉取时才计算，不产生中间缓冲区。
- **参数化**：每个处理器通过自己的配置结构体（反序列化自 JSON）控制行为，不同任务可使用同一处理器的不同配置。
- **无状态 or 确定性状态**：处理器的内部状态（如标准化器的 min/max 边界）在首次遇到数据时从数据流中确定性地计算，或者从配置中显式指定，保证可复现性。
- **Send + Sync**：所有处理器实现为 `Send + Sync`，可在 rayon 线程池中使用。

## 3. 模块内部结构组织

```
src/pipeline/
├── mod.rs              # 模块根，ProcessorRegistry
├── processor.rs        # Processor trait 定义
├── registry.rs         # ProcessorRegistry 实现
├── normalizer.rs       # 标准化器（浮点→整数映射）
├── dedup.rs            # 去重与过滤（移除重复帧、低熵帧）
├── diff_encoder.rs     # 差分编码器（存储相邻帧差分）
├── token_mapper.rs     # 令牌映射器（状态值→Unicode字符）
├── clip_stitcher.rs    # 序列截断与拼接器
└── null_proc.rs        # NullProcessor（透传，用于测试）
```

| 文件 | 职责 |
|------|------|
| `processor.rs` | 定义 `Processor` trait、`ProcessorConfig` |
| `registry.rs` | 定义 `ProcessorRegistry`，管理名称→构造函数的映射 |
| `normalizer.rs` | 将浮点 `FrameState::Float` 映射到有限整数范围 |
| `dedup.rs` | 按帧内容去除连续重复帧，可选按熵过滤低复杂度序列 |
| `diff_encoder.rs` | 存储当前帧与上一帧的差分而非绝对值 |
| `token_mapper.rs` | 将 0–N 范围的整数值映射到 Unicode 字符码点 |
| `clip_stitcher.rs` | 将过长的序列切分为固定长度片段，或为序列添加分隔符 |
| `mod.rs` | 重导出所有类型，暴露 `register_all()` 函数注册全部内置处理器 |

## 4. 公开接口定义

### 4.1 Processor trait

```rust
use crate::core::{CoreResult, SequenceFrame};

/// 后处理器的抽象接口。
///
/// 每个处理器是一个迭代器适配器：接受帧流，返回变换后的帧流。
/// 实现者必须为 `Send + Sync`，保证在多线程环境中安全使用。
pub trait Processor: Send + Sync {
    /// 返回该处理器的唯一标识名称。
    fn name(&self) -> &'static str;

    /// 对输入帧流应用变换，返回变换后的帧流。
    ///
    /// # Arguments
    /// * `input` - 原始帧迭代器。消费此迭代器产生的所有帧。
    ///
    /// # Returns
    /// 变换后的惰性帧迭代器。注意：消费返回的迭代器才会驱动处理。
    fn process(
        &self,
        input: Box<dyn Iterator<Item = SequenceFrame> + Send>,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>>;
}

/// 处理器构造函数类型：接受配置 JSON，返回 Processor trait 对象。
pub type ProcessorFactory = fn(&Value) -> CoreResult<Box<dyn Processor>>;
```

### 4.2 ProcessorRegistry

```rust
use std::collections::HashMap;
use serde_json::Value;
use crate::core::CoreResult;

/// 处理器注册表。维护名称→构造函数的映射。
#[derive(Default)]
pub struct ProcessorRegistry {
    factories: HashMap<&'static str, ProcessorFactory>,
}

impl ProcessorRegistry {
    /// 创建空注册表。
    pub fn new() -> Self { /* ... */ }

    /// 注册一个处理器构造函数。
    pub fn register(&mut self, name: &'static str, factory: ProcessorFactory) { /* ... */ }

    /// 按名称实例化一个处理器。
    ///
    /// # Arguments
    /// * `name` - 处理器名称（如 "normalizer"）。
    /// * `config` - 处理器配置的 JSON 值，可为 `Value::Null` 使用默认配置。
    ///
    /// # Errors
    /// 当名称未注册或配置不合法时返回 CoreError。
    pub fn get(&self, name: &str, config: &Value) -> CoreResult<Box<dyn Processor>> { /* ... */ }

    /// 列出所有已注册的处理器名称。
    pub fn list_names(&self) -> Vec<&'static str> { /* ... */ }
}
```

### 4.3 内置处理器配置类型

```rust
/// 标准化器配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizerConfig {
    /// 目标整数范围上限（含），如 255 表示映射到 [0, 255]。
    #[serde(default = "default_max_val")]
    pub max_val: u32,
    /// 标准化方法。
    #[serde(default)]
    pub method: NormalizeMethod,
}

fn default_max_val() -> u32 { 255 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NormalizeMethod {
    /// 线性缩放：value = (x - min) / (max - min) * max_val
    Linear,
    /// 对数分桶：value = log2(1 + |x|) 缩放到目标范围
    LogBucket,
    /// 均匀量化：将值域均匀切分为 max_val+1 个桶
    UniformQuantile,
}

/// 去重过滤器配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupConfig {
    /// 是否移除连续重复帧（前后两帧完全相同）。
    #[serde(default = "default_true")]
    pub remove_consecutive_duplicates: bool,
    /// 是否移除全零帧。
    #[serde(default = "default_true")]
    pub remove_all_zeros: bool,
    /// 最小熵阈值：移除熵值低于此阈值的帧序列（0.0 ~ 1.0），0.0 表示不过滤。
    #[serde(default)]
    pub min_entropy: f64,
}

/// 差分编码器配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffEncoderConfig {
    /// 是否在序列首帧前插入一个全零的参考帧（使得首帧编码为首帧本身）。
    #[serde(default)]
    pub prepend_zero_frame: bool,
}

/// 令牌映射器配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenMapperConfig {
    /// 映射起始 Unicode 码点，如 0x4E00（CJK 统一汉字起始）。
    #[serde(default = "default_start_codepoint")]
    pub start_codepoint: u32,
    /// 是否插入换行符作为帧分隔。
    #[serde(default)]
    pub insert_newline: bool,
}

fn default_start_codepoint() -> u32 { 0x4E00 }

/// 序列截断/拼接器配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipStitcherConfig {
    /// 最大序列长度（帧数），超出部分切分为新序列。
    #[serde(default)]
    pub max_len: Option<usize>,
    /// 是否在序列之间插入分隔标记帧。
    #[serde(default)]
    pub insert_separator: bool,
}
```

## 5. 核心逻辑详解

### 5.1 标准化器 (Normalizer)

**功能**：将连续浮点值映射到有限整数范围，消除浮点精度噪声，形成固定大小词汇表。

**算法**（以 Linear 方法为例）：

```
输入: frames: Iterator<SequenceFrame>, config: NormalizerConfig

步骤:
1. 第一遍扫描：遍历所有帧，收集所有 FrameState::Float 的值，计算全局 min 和 max。
   若 min == max，则所有值映射为 0。

2. 第二遍变换：对每帧：
   for each FrameState in frame.state.values:
       match state:
           Float(v) → 计算 scaled = (v - min) / (max - min) * config.max_val
                      将 scaled 钳位到 [0, config.max_val] 后取整为 i64
                      替换为 FrameState::Integer(clamped)
           Integer(v) → 保持不变（已经离散）
           Bool(v)    → 替换为 FrameState::Integer(v as i64)

3. 产出变换后的帧。

特殊情况:
- LogBucket: 对绝对值取 log2(1+|x|)，然后线性缩放。
- UniformQuantile: 第一遍扫描后排序所有值，按分位数边界映射。
```

**性能优化**：第一遍扫描需要遍历全部帧。对于流式产生的大数据量，可在配置中显式指定 `min` 和 `max` 边界（而非从数据学习），从而避免两次遍历。

### 5.2 去重过滤器 (DedupFilter)

**功能**：移除冗余帧，保留"有信息量"的状态变化。

**算法**：

```
过程 process(frames):
    let mut prev_frame: Option<SequenceFrame> = None;

    for frame in frames:
        // 检查全零帧
        if config.remove_all_zeros && is_all_zero(&frame.state):
            continue;  // 跳过

        // 检查连续重复帧
        if config.remove_consecutive_duplicates:
            if let Some(ref prev) = prev_frame:
                if prev.state == frame.state:
                    continue;  // 跳过重复帧

        // 可选：低熵过滤
        if config.min_entropy > 0.0:
            let entropy = estimate_entropy(&frame.state);
            if entropy < config.min_entropy:
                continue;  // 跳过低熵帧

        prev_frame = Some(frame.clone());
        yield frame;
```

**熵估计**：对帧的 `FrameState::Integer` 值构建 256-bin 直方图，计算香农熵 `H = -Σ p_i * log2(p_i)`，归一化到 [0, 1]。

### 5.3 差分编码器 (DiffEncoder)

**功能**：对相邻帧计算差分，减少缓慢变化序列的冗余。

**算法**：

```
过程 process(frames):
    let mut prev: Option<FrameData> = None;

    for frame in frames:
        let diff_frame = match prev {
            None => {
                if config.prepend_zero_frame {
                    // 首帧编码为自身（与零帧的差分即自身）
                    frame.clone()
                } else {
                    // 首帧直接产出（不做差分）
                    frame.clone()
                }
            }
            Some(ref previous) => {
                let diff_values: Vec<FrameState> = frame.state.values.iter()
                    .zip(previous.values.iter())
                    .map(|(cur, prv)| compute_diff(cur, prv))
                    .collect();
                SequenceFrame {
                    step_index: frame.step_index,
                    state: FrameData { values: diff_values },
                    label: frame.label,
                }
            }
        };
        prev = Some(frame.state.clone());
        yield diff_frame;

fn compute_diff(cur: &FrameState, prev: &FrameState) -> FrameState:
    match (cur, prev):
        (Integer(a), Integer(b)) → Integer(a - b)
        (Float(a), Float(b))     → Float(a - b)
        (Bool(a), Bool(b))       → Bool(a ^ b)     // XOR
        _                        → cur.clone()      // 类型不匹配，产出原值
```

### 5.4 令牌映射器 (TokenMapper)

**功能**：将已离散化的整数值（0–N）映射为 Unicode 字符，使数据可直接作为语言模型的文本输入。

**算法**：

```
过程 process(frames):
    for frame in frames:
        let mut chars = Vec::with_capacity(frame.state.values.len() + 1);
        for state in &frame.state.values:
            let code_point = match state:
                Integer(v) → config.start_codepoint + (*v as u32)
                    .clamp(0, 0x10_FFFF)  // Unicode 安全上界
                Float(v)   → config.start_codepoint
                    + ((*v * 256.0) as u32).clamp(0, 255)
                Bool(v)    → config.start_codepoint + (*v as u32)
            chars.push(char::from_u32(code_point).unwrap_or('�'));
        if config.insert_newline:
            chars.push('\n');
        // 产出：将帧的状态替换为表示文本令牌的整数值
        // （下游 text sink 将整数值映射为对应的 Unicode 字符串写出）
```

### 5.5 序列截断/拼接器 (ClipStitcher)

**功能**：将过长的单序列切分为多个固定长度子序列，或在序列间插入分隔标记。

**算法**：

```
过程 process(frames):
    let mut buffer = Vec::new();
    let mut subseq_id = 0;

    for frame in frames:
        buffer.push(frame);
        if let Some(max_len) = config.max_len:
            if buffer.len() >= max_len:
                // 输出当前缓冲作为一个完整子序列
                for f in buffer.drain(..):
                    yield f;
                if config.insert_separator:
                    yield create_separator_frame(subseq_id);
                subseq_id += 1;

    // 输出剩余帧
    for f in buffer:
        yield f;
```

## 6. 与其他模块的交互

### 6.1 依赖关系

```
pipeline
  ├── 依赖 core::{SequenceFrame, FrameState, FrameData, CoreError, CoreResult}
  └── 被 scheduler 调用（通过 ProcessorRegistry::get 按名称实例化）
```

### 6.2 调度器中的调用方式

```rust
// scheduler/executor.rs 中的伪代码
let mut stream: Box<dyn Iterator<Item = SequenceFrame> + Send> = raw_generator_stream;
for proc_name in &task.pipeline {
    let config = get_processor_config(&task.params.extensions, proc_name);
    let processor = processor_registry.get(proc_name, &config)?;
    stream = processor.process(stream)?;
}
// stream 现在是经过完整管道变换后的帧迭代器
```

### 6.3 与其他模块的数据约定

| 约定 | 说明 |
|------|------|
| 管道的输入 | `SequenceFrame` 迭代器，来自 Generator::generate_stream |
| 管道的输出 | `SequenceFrame` 迭代器，传给 SinkAdapter::write_frame |
| 帧的虚实 | 管道不新增帧、不改变 `step_index` 的语义（差分编码仅改变 `state` 内容） |
| 标签保留 | `.label` 字段在管道中透传，不被修改 |

## 7. 错误处理策略

| 错误情景 | 处理 |
|----------|------|
| Processor::process 返回 Err | 向上传播到 scheduler，中止当前 Shard |
| 标准化器无法计算 min/max（空输入） | 返回空迭代器，不报错 |
| 令牌映射器遇到无法映射的值（超出 Unicode 范围） | 钳位到安全范围，不报错 |
| 注册表中未找到处理器名称 | 在清单校验阶段报错（`ManifestError`），不等待运行时 |
| 处理器配置 JSON 反序列化失败 | 在 `ProcessorRegistry::get` 中立即报错 |

## 8. 性能考量

- **迭代器零成本抽象**：每个处理器实现为自定义的 `impl Iterator` 结构体，编译器可将链式调用的多层迭代器展开为紧凑的内联循环。
- **避免中间收集**：所有处理器直接包装输入迭代器，不调用 `.collect()`。数据仅在最终消费端（sink）一次性遍历。
- **标准化器双遍历优化**：若配置中显式给定了 min/max，标准化器可跳过第一遍扫描，单遍历完成。
- **差分编码器的内存**：仅缓存上一帧的 `FrameData`，内存开销为 O(state_dim)，与序列长度无关。
- **去重过滤器的回溯**：仅回溯一帧（检查连续重复），不需要窗口缓存。

## 9. 可测试性设计

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_frames() -> Vec<SequenceFrame> { /* 构造含 Float、Integer、Bool 的帧 */ }

    #[test]
    fn test_normalizer_linear_scales_correctly() {
        let frames = make_test_frames();
        let config = NormalizerConfig { max_val: 10, method: NormalizeMethod::Linear };
        let normalizer = Normalizer::new(&config);
        let output: Vec<_> = normalizer.process(stream_from(frames)).unwrap().collect();
        // 验证所有值在 [0, 10] 范围内
    }

    #[test]
    fn test_dedup_removes_identical_consecutive_frames() {
        let frames = vec![
            make_frame(0, vec![1.0]),
            make_frame(1, vec![1.0]),  // 与上一帧相同
            make_frame(2, vec![2.0]),
        ];
        let output: Vec<_> = DedupFilter::new(&default_config())
            .process(stream_from(frames)).unwrap().collect();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_diff_encoder_produces_correct_diff() {
        let frames = vec![
            make_frame(0, vec![10.0]),
            make_frame(1, vec![12.0]),
        ];
        let output: Vec<_> = DiffEncoder::new(&default_config())
            .process(stream_from(frames)).unwrap().collect();
        assert_eq!(output[1].state.values[0], FrameState::Float(2.0)); // 12-10=2
    }

    #[test]
    fn test_token_mapper_output_in_unicode_range() {
        // ...
    }

    #[test]
    fn test_pipeline_chain_multiple_processors() {
        // normalize → dedup → diff → token_map
        let pipeline = vec!["normalizer", "dedup", "diff_encoder"];
        let mut stream = gen_stream();
        for name in pipeline {
            let proc = registry.get(name, &json!(null)).unwrap();
            stream = proc.process(stream).unwrap();
        }
        let result: Vec<_> = stream.collect();
        // 验证链式处理的最终结果
    }

    #[test]
    fn test_processor_registry_rejects_unknown_name() {
        let reg = ProcessorRegistry::new();
        assert!(reg.get("nonexistent", &json!(null)).is_err());
    }
}
```

## 10. 配置与参数

### 10.1 清单中引用处理器

在 YAML 清单中，任务的 `pipeline` 字段指定要使用的处理器列表：

```yaml
tasks:
  - name: "ca_with_processing"
    generator: "cellular_automaton"
    params:
      seq_length: 1000
      extensions:
        ca:
          rule: 110
        pipeline_config:
          normalizer: { max_val: 255, method: "linear" }
          dedup: { remove_consecutive_duplicates: true, remove_all_zeros: true }
          token_mapper: { start_codepoint: 0x4E00 }
    pipeline: ["normalizer", "dedup", "token_mapper"]
```

每个处理器的配置从 `params.extensions.pipeline_config.<processor_name>` 中反序列化。若未提供，则使用默认配置。

---

通过以上设计，后处理管道层以可组合、惰性求值的方式提供了完整的数据清洗与变换能力，确保从生成器产出的原始数据在进入输出适配器前已经过充分规范化，直接适配下游消费需求。
