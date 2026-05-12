# NpyBatchAdapter CLI 管道使用指南

## 概述

`NpyBatchAdapter` 是 StructGen-rs 的批量数据输出适配器，专门用于生成机器学习训练数据。它将流式生成的帧数据收集为批量格式，输出为 NumPy `.npy` 文件，支持 `(B, T, H, W, C)` 或 `(B, T, state_dim)` 形状的张量数据。

### 核心特性

- **批量输出**：将多个样本的帧数据合并为单个 `.npy` 文件
- **Python 友好**：输出格式直接兼容 NumPy/JAX/PyTorch
- **确定性生成**：相同种子产生相同数据
- **灵活配置**：支持多种生成器和后处理管道

## 数据流架构

```
生成器 → 后处理管道 → BatchCollector → NpyBatchAdapter → .npy 文件
   │           │              │                │
   │           │              │                └── 输出 (B, T, H, W, C)
   │           │              └── 收集 batch_size × num_frames 帧
   │           └── 可选：去重、归一化等
   └── ca2d, ca3d, nca2d 等
```

## 配置结构

### NpyBatchConfig 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `batch_size` | usize | ✅ | 每个批次的样本数量 |
| `num_frames` | usize | ✅ | 每个样本的帧数 |
| `rows` | usize | ❌ | 网格高度（用于设置 shape） |
| `cols` | usize | ❌ | 网格宽度（用于设置 shape） |
| `channels` | usize | ❌ | 通道数（用于设置 shape） |

### BatchCollectorConfig 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `batch_size` | usize | ✅ | 每个批次的样本数量 |
| `num_frames` | usize | ✅ | 每个样本的帧数 |

> **重要**：`BatchCollectorConfig` 和 `NpyBatchConfig` 的 `batch_size` 和 `num_frames` 必须保持一致。

---

## 生成器配置详解

### 1. NCA2D（神经元胞自动机 2D）

最适合生成训练数据的生成器，输出离散状态网格。

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `d_state` | u8 | 10 | 离散状态数（通道数） |
| `n_groups` | u8 | 1 | 通道组数 |
| `rows` | usize | 12 | 网格高度 |
| `cols` | usize | 12 | 网格宽度 |
| `temperature` | f64 | 0.0 | 采样温度（0=确定性） |
| `hidden_dim` | usize | 16 | 隐藏层维度 |
| `conv_features` | usize | 4 | 卷积特征数 |
| `identity_bias` | f64 | 0.0 | 恒等偏置 |

#### YAML 配置示例

```yaml
global:
  output_dir: "output/nca2d_batch"
  default_format: "NpyBatch"
  num_threads: 4

tasks:
  - name: "nca2d_train"
    generator: "nca2d"
    output_format: "NpyBatch"
    params:
      seq_length: 10
      extensions:
        d_state: 10
        n_groups: 1
        rows: 12
        cols: 12
        temperature: 0.0
        hidden_dim: 16
        conv_features: 4
        sink_config:
          batch_size: 32
          num_frames: 10
          rows: 12
          cols: 12
          channels: 10
    count: 1000
    seed: 42
    pipeline:
      - "batch_collector"
```

#### 输出形状

`(B, T, H, W, C)` = `(batch_size × num_batches, num_frames, rows, cols, d_state)`

---

### 2. CA2D（元胞自动机 2D）

经典元胞自动机，支持多种规则类型。

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rule_type` | String | "lifelike" | 规则类型 |
| `birth` | Vec\<u8\> | [3] | 诞生规则 |
| `survival` | Vec\<u8\> | [2, 3] | 生存规则 |
| `d_state` | u8 | 2 | 状态数 |
| `rows` | usize | 64 | 网格高度 |
| `cols` | usize | 64 | 网格宽度 |
| `boundary` | String | "periodic" | 边界条件 |
| `neighborhood` | String | "moore" | 邻域类型 |
| `init_mode` | String | "random" | 初始化模式 |

#### 支持的规则类型

| rule_type | 说明 | 示例 |
|-----------|------|------|
| `lifelike` | 生命类规则 | Game of Life: B3/S23 |
| `totalistic` | 总量规则 | WireWorld |
| `cyclic` | 循环 CA | Cyclic Space |
| `hensel` | Hensel 记法 | B2-a/S12 |
| `lookuptable` | 查找表 | 自定义规则 |

#### YAML 配置示例

```yaml
tasks:
  - name: "game_of_life_batch"
    generator: "ca2d"
    output_format: "NpyBatch"
    params:
      seq_length: 50
      extensions:
        rule_type: "lifelike"
        birth: [3]
        survival: [2, 3]
        rows: 32
        cols: 32
        d_state: 2
        boundary: "periodic"
        sink_config:
          batch_size: 16
          num_frames: 50
          rows: 32
          cols: 32
          channels: 2
    count: 500
    seed: 123
    pipeline:
      - "batch_collector"
```

---

### 3. CA3D（元胞自动机 3D）

三维元胞自动机，适用于体数据生成。

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rule_type` | String | "lifelike" | 规则类型 |
| `birth` | Vec\<u8\> | [5,6,7] | 诞生规则 |
| `survival` | Vec\<u8\> | [5,6] | 生存规则 |
| `d_state` | u8 | 2 | 状态数 |
| `depth` | usize | 16 | 深度 |
| `rows` | usize | 16 | 高度 |
| `cols` | usize | 16 | 宽度 |

#### YAML 配置示例

```yaml
tasks:
  - name: "ca3d_batch"
    generator: "ca3d"
    output_format: "NpyBatch"
    params:
      seq_length: 20
      extensions:
        rule_type: "lifelike"
        birth: [5, 6, 7]
        survival: [5, 6]
        depth: 8
        rows: 8
        cols: 8
        d_state: 2
        sink_config:
          batch_size: 8
          num_frames: 20
          rows: 8
          cols: 8
          channels: 2
    count: 100
    seed: 456
    pipeline:
      - "batch_collector"
```

---

### 4. CA（一维元胞自动机）

经典一维元胞自动机，如 Rule 30、Rule 110。

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rule` | u32 | 30 | 规则号 (0-255) |
| `width` | usize | 100 | 网格宽度 |

#### YAML 配置示例

```yaml
tasks:
  - name: "rule30_batch"
    generator: "ca"
    output_format: "NpyBatch"
    params:
      seq_length: 100
      extensions:
        rule: 30
        width: 100
        sink_config:
          batch_size: 64
          num_frames: 100
          rows: 1
          cols: 100
          channels: 2
    count: 200
    seed: 789
    pipeline:
      - "batch_collector"
```

---

### 5. Lorenz（洛伦兹系统）

连续动力系统，生成混沌时间序列。

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sigma` | f64 | 10.0 | σ 参数 |
| `rho` | f64 | 28.0 | ρ 参数 |
| `beta` | f64 | 8.0/3.0 | β 参数 |
| `dt` | f64 | 0.01 | 时间步长 |

#### YAML 配置示例

```yaml
tasks:
  - name: "lorenz_batch"
    generator: "lorenz"
    output_format: "NpyBatch"
    params:
      seq_length: 1000
      extensions:
        sigma: 10.0
        rho: 28.0
        beta: 2.6667
        dt: 0.01
        sink_config:
          batch_size: 32
          num_frames: 1000
    count: 100
    seed: 111
    pipeline:
      - "batch_collector"
```

---

### 6. Logistic（逻辑映射）

离散混沌映射。

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `r` | f64 | 4.0 | 控制参数 (0-4) |
| `x0` | f64 | 0.5 | 初始值 (0-1) |

#### YAML 配置示例

```yaml
tasks:
  - name: "logistic_batch"
    generator: "logistic"
    output_format: "NpyBatch"
    params:
      seq_length: 500
      extensions:
        r: 3.9
        sink_config:
          batch_size: 64
          num_frames: 500
    count: 200
    seed: 222
    pipeline:
      - "batch_collector"
```

---

## 完整配置示例

### 示例 1：NCA2D 训练数据生成

```yaml
# nca2d_training.yaml
global:
  output_dir: "output/nca2d_train"
  default_format: "NpyBatch"
  num_threads: 8
  log_level: "info"

tasks:
  - name: "nca2d_deterministic"
    generator: "nca2d"
    output_format: "NpyBatch"
    params:
      seq_length: 10
      extensions:
        d_state: 10
        n_groups: 1
        rows: 12
        cols: 12
        temperature: 0.0
        hidden_dim: 16
        conv_features: 4
        sink_config:
          batch_size: 32
          num_frames: 10
          rows: 12
          cols: 12
          channels: 10
    count: 1000
    seed: 42
    pipeline:
      - "batch_collector"

  - name: "nca2d_stochastic"
    generator: "nca2d"
    output_format: "NpyBatch"
    params:
      seq_length: 10
      extensions:
        d_state: 10
        n_groups: 1
        rows: 12
        cols: 12
        temperature: 0.5
        hidden_dim: 16
        conv_features: 4
        sink_config:
          batch_size: 32
          num_frames: 10
          rows: 12
          cols: 12
          channels: 10
    count: 1000
    seed: 43
    pipeline:
      - "batch_collector"
```

### 示例 2：多生成器混合批次

```yaml
# multi_generator.yaml
global:
  output_dir: "output/mixed_batch"
  default_format: "NpyBatch"
  num_threads: 4

tasks:
  - name: "ca2d_life"
    generator: "ca2d"
    output_format: "NpyBatch"
    params:
      seq_length: 50
      extensions:
        rule_type: "lifelike"
        birth: [3]
        survival: [2, 3]
        rows: 32
        cols: 32
        sink_config:
          batch_size: 16
          num_frames: 50
          rows: 32
          cols: 32
          channels: 2
    count: 500
    seed: 100
    pipeline:
      - "batch_collector"

  - name: "ca2d_maze"
    generator: "ca2d"
    output_format: "NpyBatch"
    params:
      seq_length: 50
      extensions:
        rule_type: "lifelike"
        birth: [3]
        survival: [1, 2, 3, 4, 5]
        rows: 32
        cols: 32
        sink_config:
          batch_size: 16
          num_frames: 50
          rows: 32
          cols: 32
          channels: 2
    count: 500
    seed: 200
    pipeline:
      - "batch_collector"
```

---

## CLI 使用

### 基本命令

```bash
# 运行 YAML 配置
structgen-rs run -m config.yaml

# 指定输出目录
structgen-rs run -m config.yaml -o ./output

# 指定格式（覆盖 YAML 配置）
structgen-rs run -m config.yaml --format npy-batch

# 指定线程数
structgen-rs run -m config.yaml -t 8
```

### 输出文件命名

输出文件格式：`{task_name}_{shard_id:04d}_{seed:08x}.npy`

示例：
- `nca2d_train_0000_0000002a.npy`
- `nca2d_train_0001_0000002b.npy`

---

## Python 加载示例

```python
import numpy as np

# 加载 NPY 文件
data = np.load('nca2d_train_0000_0000002a.npy')

# 数据形状: (B, T, H, W, C)
print(f"Shape: {data.shape}")
# 示例输出: Shape: (32, 10, 12, 12, 10)

# 用于训练
import jax.numpy as jnp

# 转换为 JAX 数组
batch = jnp.array(data)

# 分离输入和目标
X = batch[:, :-1]  # 前 T-1 帧
Y = batch[:, 1:]   # 后 T-1 帧（预测目标）
```

---

## 常见问题

### Q1: batch_size 和 count 的关系？

`count` 是总样本数，`batch_size` 是每个 `.npy` 文件包含的样本数。
- 如果 `count = 1000`，`batch_size = 32`，则生成 `ceil(1000/32) = 32` 个 `.npy` 文件
- 最后一个文件可能不足 `batch_size` 个样本

### Q2: 如何确保数据确定性？

1. 固定 `seed` 参数
2. 设置 `temperature: 0.0`（对于 NCA2D）
3. 使用相同的配置文件

### Q3: sink_config 中的 rows/cols/channels 是否必须？

不必须。如果省略，NpyBatchAdapter 会从实际数据推断 shape。但建议显式设置以确保输出形状正确。

### Q4: 如何处理不同大小的生成器输出？

确保 `sink_config` 中的 `rows`、`cols`、`channels` 与生成器的输出维度匹配：
- NCA2D: `rows`、`cols`、`channels = d_state`
- CA2D: `rows`、`cols`、`channels = d_state`
- Lorenz/Logistic: 不需要 `rows`/`cols`/`channels`

---

## 性能优化建议

1. **并行度**：设置 `num_threads` 为 CPU 核心数
2. **批次大小**：根据 GPU 内存调整 `batch_size`（推荐 32-128）
3. **分片大小**：使用 `shard_size` 控制每个分片的样本数
4. **流式写入**：默认启用，减少内存占用

```yaml
global:
  num_threads: 8
  shard_max_sequences: 10000
  stream_write: true
```
