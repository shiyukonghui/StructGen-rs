# StructGen-rs 任务调度层 (scheduler) 详细设计规格

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| 1.0 | 2026-05-01 | - | 初始版本，定义任务调度、种子管理、并行执行逻辑 |

## 1. 模块概述

任务调度层（scheduler）是 StructGen-rs 系统的协调中枢。它负责将用户的 YAML 清单转换为可执行的生成任务，管理确定性的种子派生，并通过线程池（rayon）并行执行所有分片任务，协调生成器、后处理管道和输出适配器之间的数据流动。

该模块的职责包括：
- 解析 YAML 清单文件，生成内部任务描述结构（`Manifest` → `Vec<TaskSpec>`）。
- 将每个任务的样本数量按分片策略切分为多个 `Shard`，并派生唯一种子。
- 在线程池中并行执行所有 `Shard`，每个分片串联"生成→后处理→写出"流程。
- 支持两种执行模式：阻塞收集模式（中小规模）和流式写出模式（大规模）。
- 收集每个分片的执行统计，汇总后传递给元数据层。

**核心原则**：调度器不关心具体生成什么样的数据——它只是任务的编排者和资源的分配者。所有的领域知识（如何生成 CA 序列、如何模拟物理系统）封装在生成器仓储中。

## 2. 设计目标与原则

- **完全确定性**：相同的清单 + 相同的二进制 → 逐比特一致的输出。所有随机性仅来自分片种子，分片种子的派生算法为确定性函数。
- **弹性并行度**：分片数量可根据 CPU 核心数和样本数量自动计算，也可由用户在清单中指定。
- **容错与隔离**：单个分片的生成失败不应影响其他分片。失败的 Shard 被记录到元数据，但不中止整个运行。
- **两种模式自由切换**：用户可通过配置选择"收集到内存后写出"或"生成即写出"，前者适合快速验证，后者适合 TB 级数据生产。
- **资源上限可控**：通过限制同时执行的分片数（并发度 = min(线程数, 总分片数)），避免 CPU 过度竞争。

## 3. 模块内部结构组织

```
src/scheduler/
├── mod.rs          # 模块根，调度入口 run_manifest()
├── manifest.rs     # Manifest、TaskSpec 定义与 YAML 解析
├── shard.rs        # Shard、ShardResult 定义与分片算法
├── executor.rs     # execute_shard() 分片执行逻辑
└── seed.rs         # 种子派生算法
```

各文件职责：

| 文件 | 职责 |
|------|------|
| `manifest.rs` | 定义 `Manifest` 和 `TaskSpec` 结构体，实现从 YAML 字符串/文件反序列化 |
| `shard.rs` | 定义 `Shard`（分片描述）和 `ShardResult`（分片执行结果），实现分片切分算法 |
| `executor.rs` | 实现 `execute_shard()` 函数：实例化生成器 → 组装管道 → 驱动输出适配器 → 收集统计 |
| `seed.rs` | 提供 `derive_seed(base, shard_index)` 确定性的种子派生函数 |
| `mod.rs` | 暴露 `run_manifest()` 作为模块唯一公开入口 |

## 4. 公开接口定义

### 4.1 清单数据结构

```rust
use serde::{Deserialize, Serialize};
use crate::core::{GenParams, GlobalConfig, OutputFormat};

/// 完整清单，包含任务列表和全局配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// 任务列表。
    pub tasks: Vec<TaskSpec>,
    /// 全局配置（输出目录、线程数等），可被命令行参数覆盖。
    #[serde(default)]
    pub global: GlobalConfig,
}

/// 单个任务的描述。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpec {
    /// 任务名称（用于元数据和日志标识）。
    pub name: String,
    /// 生成器名称，必须已在 GeneratorRegistry 中注册。
    pub generator: String,
    /// 通用参数（序列长度、网格、扩展字段）。
    pub params: GenParams,
    /// 要生成的样本数量。
    pub count: usize,
    /// 基础随机种子。
    pub seed: u64,
    /// 后处理管道中的处理器名称列表，按顺序应用。
    /// 空列表表示不进行任何后处理。
    #[serde(default)]
    pub pipeline: Vec<String>,
    /// 输出格式，若未指定则使用全局默认值。
    #[serde(default)]
    pub output_format: Option<OutputFormat>,
    /// 每个分片包含的最大样本数，若未指定则自动计算。
    #[serde(default)]
    pub shard_size: Option<usize>,
}
```

### 4.2 分片数据结构

```rust
/// 分片描述，表示一个可独立并行执行的子任务。
#[derive(Debug, Clone)]
pub struct Shard {
    /// 所属任务在 Manifest.tasks 中的索引。
    pub task_idx: usize,
    /// 分片在本任务内的编号，从 0 开始。
    pub shard_idx: usize,
    /// 派生后的唯一种子。
    pub seed: u64,
    /// 本分片需生成的样本数量。
    pub sample_count: usize,
}

/// 分片执行结果，包含输出统计信息和文件元信息。
///
/// `OutputStats` 来自 `sink` 模块，scheduler 通过 `SinkAdapter::close()` 获取。
#[derive(Debug, Clone)]
pub struct ShardResult {
    /// 关联的任务名称。
    pub task_name: String,
    /// 分片编号。
    pub shard_idx: usize,
    /// 派生种子（= derive_seed(base_seed, shard_idx)）。
    pub seed: u64,
    /// 本分片的样本数量。
    pub sample_count: usize,
    /// 输出格式。
    pub format: OutputFormat,
    /// 分片的输出统计（由 sink::SinkAdapter::close() 产生）。
    pub stats: OutputStats,
    /// 发生的错误信息（None 表示成功）。
    pub error: Option<String>,
}
```

### 4.3 公开函数

```rust
use crate::core::{CoreResult, GeneratorRegistry};

/// 运行一个完整的清单。
///
/// 这是 scheduler 模块对外暴露的唯一入口函数。
///
/// # Arguments
/// * `manifest` - 已解析的清单。
/// * `registry` - 生成器注册表（通常由 CLI 模块初始化时全局填充）。
///
/// # Returns
/// * `Vec<ShardResult>` - 所有分片的执行结果。
///
/// # Errors
/// * `CoreError::ManifestError` - 清单校验失败。
/// * `CoreError::GeneratorNotFound` - 任务引用了不存在的生成器。
pub fn run_manifest(
    manifest: &Manifest,
    registry: &GeneratorRegistry,
) -> CoreResult<Vec<ShardResult>> { /* ... */ }
```

### 4.4 种子派生函数

```rust
/// 根据基础种子和分片序号派生唯一种子。
///
/// 派生规则：`seed = base_seed.wrapping_add(shard_idx as u64)`。
/// 简单的包装加法保证不同分片、不同任务之间的种子无冲突。
///
/// 对于更高级的需求（如确保不同任务名产生完全不相关的种子空间），
/// 可以使用 SipHash 对 (base_seed, task_name, shard_idx) 三元组哈希。
///
/// # Arguments
/// * `base_seed` - 用户在清单中为每个任务指定的基础种子。
/// * `shard_idx` - 分片编号，从 0 开始。
///
/// # Returns
/// 派生后的唯一种子。
pub fn derive_seed(base_seed: u64, shard_idx: usize) -> u64 {
    base_seed.wrapping_add(shard_idx as u64)
}
```

## 5. 核心逻辑详解

### 5.1 清单解析与校验

```
read_manifest(path: &Path) → CoreResult<Manifest>
  ↓
1. 读取 YAML 文件内容
2. serde_yaml::from_str 反序列化为 Manifest
3. 执行校验：
   a. global.output_dir 非空且可写入
   b. 每个 TaskSpec.name 唯一
   c. 每个 TaskSpec.count > 0
   d. 每个 TaskSpec.pipeline 中的处理器名称在注册表中存在
   e. 每个 TaskSpec.generator 名称在 GeneratorRegistry 中存在
   f. TaskSpec.shard_size 如果指定则 > 0
4. 校验通过 → 返回 Manifest
   校验失败 → 返回 CoreError::ManifestError 附带详细错误信息
```

### 5.2 分片算法

```
shard_tasks(manifest: &Manifest) → Vec<Shard>

对每个 TaskSpec t:
    effective_shard_size = t.shard_size
        .unwrap_or_else(|| compute_auto_shard_size(t.count, num_cpus))

    shard_count = ceil_div(t.count, effective_shard_size)

    对每个 shard_idx in 0..shard_count:
        begin_sample = shard_idx * effective_shard_size
        end_sample   = min(begin_sample + effective_shard_size, t.count)
        sample_count = end_sample - begin_sample

        构造 Shard {
            task_idx:    t 在 manifest.tasks 中的索引,
            shard_idx:   shard_idx,
            seed:        derive_seed(t.seed, shard_idx),
            sample_count: sample_count,
        }

所有 Shard 收集到一个 Vec 中，任务间顺序不重要（并行执行）。

自动分片大小计算：
    target_shards = num_cpus * 4  // 目标是 CPU 核心数的 4 倍以实现负载均衡
    auto_size = max(1, ceil_div(t.count, target_shards))
```

### 5.3 分片执行流程

```
execute_shard(
    shard: &Shard,
    task: &TaskSpec,
    registry: &GeneratorRegistry,
    processor_registry: &ProcessorRegistry,    // pipeline 模块提供
    adapter_factory: &dyn SinkAdapterFactory,  // sink 模块提供
) → ShardResult

  ↓
1. 从 registry.instantiate(task.generator, &task.params.extensions) 获取生成器实例
  ↓
2. 调用 generator.generate_stream(shard.seed, &task.params) 获取帧迭代器
  ↓
3. 构造后处理链：
   let mut stream: Box<dyn Iterator<Item = SequenceFrame> + Send> = raw_stream;
   for proc_name in &task.pipeline {
       let config = task.params.extensions
           .get("pipeline_config")
           .and_then(|c| c.get(proc_name))
           .cloned()
           .unwrap_or(Value::Null);
       let processor = processor_registry.get(proc_name, &config)?;
       stream = processor.process(stream)?;
   }
  ↓
4. 初始化输出适配器：
   let format = task.output_format.unwrap_or(manifest.global.default_format);
   let mut adapter: Box<dyn SinkAdapter> = match format { ... };
   adapter.open(&output_dir, &task.name, shard.shard_idx, shard.seed, config)?;
  ↓
5. 逐帧写出（迭代驱动）：
   for frame in stream {
       adapter.write_frame(&frame)?;
   }
  ↓
6. 关闭适配器，获取最终统计：
   let stats: OutputStats = adapter.close()?;
  ↓
7. 返回 ShardResult {
       task_name: task.name.clone(),
       shard_idx: shard.shard_idx,
       seed: shard.seed,
       sample_count: shard.sample_count,
       format,
       stats,              // 包含 frames_written、bytes_written、output_path、file_hash
       error: None,
   }
```

### 5.4 两种执行模式

```
模式一：阻塞收集模式（默认，适用于中小规模数据）

run_manifest_blocking(manifest, registry) → Vec<ShardResult>
  ↓
let all_shards = shard_tasks(manifest);
let results = all_shards
    .par_iter()                            // rayon 并行
    .map(|shard| execute_shard(shard, ...))
    .collect::<Vec<_>>();                  // 阻塞等待所有分片完成
return results


模式二：流式写出模式（适用于 TB 级数据）

run_manifest_streaming(manifest, registry) → Vec<ShardResult>
  ↓
// 每个分片拥有私有的输出适配器，写入不同的分片文件，无锁竞争。
// 与阻塞模式的区别仅在于：rayon 任务内部使用 while-let 驱动迭代器，
// 不积累中间结果。逻辑上与阻塞模式的 execute_shard 完全一致。
// 实际上 execute_shard 总是流式处理——区别在于是否在更上层收集。
```

### 5.5 容错处理

```
单个 Shard 执行出错时：

execute_shard() 内部捕获所有 CoreError:
  match result {
      Ok(stats) → ShardResult { error: None, ...stats }
      Err(e)    → ShardResult { error: Some(e.to_string()), ...defaults }
  }

上层 run_manifest 收集所有 ShardResult:
  let errors: Vec<_> = results.iter().filter(|r| r.error.is_some()).collect();
  if !errors.is_empty() {
      log::warn!("{} shards failed: {:?}", errors.len(), errors);
      // 不中止运行，输出部分结果和元数据
  }
```

## 6. 与其他模块的交互

### 6.1 依赖关系

```
scheduler
  ├── 依赖 core::{Generator, GeneratorRegistry, GenParams, CoreError, SequenceFrame, OutputFormat}
  ├── 依赖 generators（通过 registry 间接调用，不直接引用具体生成器）
  ├── 依赖 pipeline::{ProcessorRegistry, Processor trait}
  ├── 依赖 sink::{SinkAdapter, SinkAdapterFactory, OutputStats}
  └── 被 CLI 调用
```

### 6.2 调用时序图

```
CLI
  │
  ├→ scheduler::run_manifest(manifest, &gen_registry)
  │     │
  │     ├→ shard_tasks(manifest) → Vec<Shard>
  │     │
  │     ├→ [ rayon 并行区域 ]
  │     │     │
  │     │     ├→ gen_registry.instantiate("ca", extensions)
  │     │     │     └→ CA_Generator::from_extensions(extensions)  // generators 模块
  │     │     │
  │     │     ├→ generator.generate_stream(seed, params)
  │     │     │     └→ Box<dyn Iterator<Item = SequenceFrame> + Send>  // generators 模块
  │     │     │
  │     │     ├→ [ 后处理链 ]
  │     │     │     ├→ proc_registry.get("normalizer")
  │     │     │     │     └→ Normalizer::new(config)  // pipeline 模块
  │     │     │     ├→ proc_registry.get("dedup")
  │     │     │     │     └→ DedupFilter::new(config) // pipeline 模块
  │     │     │     └→ ...更多处理器
  │     │     │
  │     │     ├→ adapter_factory.create(OutputFormat::Parquet)
  │     │     │     └→ ParquetAdapter::new()  // sink 模块
  │     │     │
  │     │     ├→ adapter.open(path, shard_idx, config)
  │     │     ├→ [ for frame in processed_stream ] { adapter.write_frame(&frame) }
  │     │     └→ adapter.close() → OutputStats
  │     │
  │     └→ 汇总所有 ShardResult → Vec<ShardResult>
  │
  └→ CLI 将 ShardResult 传递给 metadata 模块
```

### 6.3 调度器不感知的细节

调度器**不需要知道**：
- 具体有哪些生成器类型（只在 registry 中按名称查找）
- 后处理管道的具体处理逻辑（只通过 Processor trait 接口调用）
- 输出文件的物理格式（只通过 SinkAdapter trait 接口写入）

这种设计使得添加新生成器、新处理器或新输出格式时，scheduler 无需任何修改。

## 7. 错误处理策略

### 7.1 错误分类与处理

| 阶段 | 可能错误 | 处理方式 |
|------|---------|----------|
| 清单解析 | `ManifestError`（YAML 格式、字段缺失、值非法） | 立即返回 Err，终止运行 |
| 分片前校验 | `GeneratorNotFound`（未注册的生成器名） | 立即返回 Err，提示可用生成器列表 |
| 分片前校验 | `InvalidParams`（参数越界） | 立即返回 Err，详细指出哪个参数不合法 |
| 单个分片执行 | `GenerationError` | 捕获并记录在 ShardResult.error 中，继续执行其他分片 |
| 单个分片执行 | `IoError`（磁盘满） | 同上，记录错误 |
| 单个分片执行 | `PipelineError` | 同上，记录错误 |

### 7.2 分片级容错伪代码

```rust
fn execute_shard_fallible(...) -> ShardResult {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        execute_shard_inner(...)
    }));
    match result {
        Ok(Ok(stats)) => ShardResult { error: None, ..stats },
        Ok(Err(e))    => ShardResult { error: Some(e.to_string()), ..defaults() },
        Err(panic)    => ShardResult {
            error: Some(format!("panic: {:?}", panic)),
            ..defaults()
        },
    }
}
```

## 8. 性能考量

- **Rayon 工作窃取调度**：将分片数量设为 CPU 核心数的 2–4 倍，使工作窃取算法能很好地平衡负载（不同生成器的执行时间差异可能很大）。
- **避免分片间的同步点**：每个分片拥有私有输出适配器，写入独立文件，完全不涉及时序协调。
- **零堆分配的热路径**：`execute_shard` 的迭代循环避免在帧级别做堆分配——迭代器传递的是栈上的 `SequenceFrame`。
- **分片粒度控制**：过小的分片（如每分片 1 个样本）导致过多的线程调度开销；默认分片大小不应小于 100 个样本。

## 9. 可测试性设计

### 9.1 单元测试

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_seed_deterministic() {
        assert_eq!(derive_seed(42, 0), 42);
        assert_eq!(derive_seed(42, 1), 43);
        assert_eq!(derive_seed(0xFFFF_FFFF_FFFF_FFFF, 1), 0); // wrapping
    }

    #[test]
    fn test_shard_task_count_distribution() {
        let task = TaskSpec { count: 1000, shard_size: Some(250), .. };
        let shards = shard_single_task(&task);
        assert_eq!(shards.len(), 4);
        assert_eq!(shards.iter().map(|s| s.sample_count).sum::<usize>(), 1000);
    }

    #[test]
    fn test_shard_task_uneven_distribution() {
        let task = TaskSpec { count: 1001, shard_size: Some(250), .. };
        let shards = shard_single_task(&task);
        assert_eq!(shards.len(), 5); // 250+250+250+250+1
    }

    #[test]
    fn test_manifest_validation_rejects_unknown_generator() {
        // registry 中没有 "unknown_gen"
        // → run_manifest 应返回 Err
    }

    #[test]
    fn test_manifest_validation_rejects_zero_count() {
        // TaskSpec.count == 0 → ManifestError
    }
}
```

### 9.2 集成测试验证点

- 使用 mock 生成器（固定产出 10 帧），验证 2 分片×5 样本的总帧数为 50。
- 注入一个会 panic 的 mock 生成器，验证其他分片正常完成，失败的 ShardResult 记录错误。
- 测试流式写出模式：分片间输出文件不冲突（不同分片写入不同文件名）。
- 测试确定性：相同清单运行两次，产生的 ShardResult 完全一致。

## 10. 配置与参数

### 10.1 清单 YAML 格式示例

```yaml
global:
  output_dir: "./output"
  default_format: "parquet"
  log_level: "info"
  num_threads: 8
  shard_max_sequences: 10000
  stream_write: true

tasks:
  - name: "rule30_ca"
    generator: "cellular_automaton"
    params:
      seq_length: 1000
      grid_size: { rows: 1, cols: 256 }
      extensions:
        ca:
          rule: 30
          boundary: "periodic"
    count: 50000
    seed: 12345
    pipeline: ["normalizer", "dedup"]
    output_format: "text"
    shard_size: 5000

  - name: "lorenz_chaos"
    generator: "lorenz_system"
    params:
      seq_length: 2000
      extensions:
        lorenz:
          sigma: 10.0
          rho: 28.0
          beta: 2.667
    count: 100000
    seed: 67890
    pipeline: ["normalizer", "token_mapper"]
    output_format: "parquet"
```

### 10.2 调度器参数说明

| 参数 | 来源 | 默认值 | 说明 |
|------|------|--------|------|
| `num_threads` | GlobalConfig | CPU 逻辑核心数 | rayon 线程池大小 |
| `stream_write` | GlobalConfig | true | 流式写出 vs 阻塞收集 |
| `shard_max_sequences` | GlobalConfig | 10000 | 每分片文件最大序列数 |
| `count` | TaskSpec | - | 任务需要生成的样本总数 |
| `shard_size` | TaskSpec | 自动计算 | 每分片样本数，覆盖全局 `shard_max_sequences` |

---

通过以上设计，任务调度层作为系统的协调中枢，以完全确定性的种子派生、灵活的并行分片策略和鲁棒的容错机制，确保从清单到输出的全流程高效可控。
