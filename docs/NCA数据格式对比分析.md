# Rust 版本 vs Python 版本 NCA 数据格式对比分析

本文档详细对比 Rust 版本 (`StructGen-rs`) 和 Python 版本 (`nca-pre-pretraining`) 的 NCA (神经细胞自动机) 数据生成和处理格式的差异。

---

## 一、数据生成层面对比

| 特性 | Python 版本 | Rust 版本 |
|------|-------------|-----------|
| **框架** | JAX/Flax (神经网络) | 纯 Rust f64 实现 |
| **网络架构** | Conv3x3 → Conv1x1 → ReLU → Conv1x1 | Conv3x3 → Conv1x1 → ReLU → Conv1x1 (相同) |
| **权重初始化** | Flax 默认初始化 | LeCun Normal 初始化 |
| **偏置初始化** | Flax 默认 | 全零初始化 |
| **温度参数默认值** | `temperature=0.0` | `temperature=1.0` |
| **identity_bias 默认值** | `identity_bias=0.0` | `identity_bias=0.0` (相同) |

---

## 二、输出数据格式对比

### Python 版本输出格式

```python
# generate_nca_dataset 输出
sims.shape = (B, T, H, W, C)
# B: batch size (模拟数量)
# T: 时间步数 (num_examples)
# H, W: 网格高度和宽度 (grid_size)
# C: 通道数 (1，单通道)

# 数据类型: JAX array (int32/int64)
```

### Rust 版本输出格式

```rust
// SequenceFrame 结构
pub struct SequenceFrame {
    pub step_index: u64,        // 时间步索引
    pub state: FrameData,       // 状态数据
    pub label: Option<String>,  // 可选标签
    pub sample_id: Option<u64>, // 样本ID
}

pub struct FrameData {
    pub values: Vec<FrameState>, // 状态值序列
}

pub enum FrameState {
    Integer(i64),  // 整数状态
    Float(f64),    // 浮点状态
    Bool(bool),    // 布尔状态
}
```

---

## 三、关键差异详解

### 1. 数据组织结构差异

```
Python 版本:
┌─────────────────────────────────────────────────┐
│  sims: (B, T, H, W, C)                         │
│  - 批量生成，一次性返回所有时间步               │
│  - 张量形式，可直接用于训练                     │
│  - 包含多个样本的完整演化序列                   │
└─────────────────────────────────────────────────┘

Rust 版本:
┌─────────────────────────────────────────────────┐
│  Iterator<Item = SequenceFrame>                │
│  - 流式生成，逐帧产出                           │
│  - 每帧独立，包含 step_index                    │
│  - 需要外部收集和组织                           │
└─────────────────────────────────────────────────┘
```

### 2. 网格数据布局差异

**Python 版本**:
```python
# 网格数据: (H, W) 或 (H, W, C)
# 索引: grid[r, w] 或 grid[r, w, c]
# 多组数据: (H, W, G, D) → reshape → (H, W, G*D)
```

**Rust 版本**:
```rust
// 网格数据: Vec<u8> 长度 = rows * cols * n_groups
// 索引: (r * cols + c) * n_groups + g
// 扁平化存储，交错排列各组数据
```

### 3. Patch Tokenization 差异

**Python 版本** (`utils/tokenizers.py`):
```python
def encode_task(self, grid: jnp.ndarray) -> torch.Tensor:
    """
    输入: (B, N, H, W, C)
    B: batch, N: num_examples, H,W: grid, C: channel
    """
    # 1. reshape 和 transpose
    grid = grid.reshape(B, N, N_H, self.patch, N_W, self.patch)
    grid = grid.transpose(0, 1, 2, 4, 3, 5)
    grid = grid.reshape(B, N, N_H*N_W, self.patch * self.patch)
    
    # 2. base-N 编码
    powers = (self.num_colors ** jnp.arange(self.patch * self.patch))
    tokens = jnp.einsum('bnlp,p->bnl', grid, powers)
    
    # 3. 添加 start/end token
    tokens = jnp.concat([start_tokens, tokens, end_tokens], axis=-1)
    
    # 4. flatten 为序列
    tokens = tokens.reshape(B, -1)  # (B, N * (N_H*N_W + 2))
```

**Rust 版本** (`src/pipeline/patch_tokenizer.rs`):
```rust
fn next(&mut self) -> Option<Self::Item> {
    // 1. 每个 group 独立 tokenization
    for g in 0..self.n_groups {
        for ph in 0..n_h {
            for pw in 0..n_w {
                // 2. base-N 编码
                let mut token: i64 = 0;
                for di in 0..self.patch {
                    for dj in 0..self.patch {
                        let idx = (r * self.cols + c) * self.n_groups + g;
                        let val = frame.state.values[idx];
                        token += val * self.powers[local_idx];
                    }
                }
                tokens.push(token);
            }
        }
    }
    
    // 3. 添加 start/end token (每帧)
    tokens.insert(0, self.start_token);
    tokens.push(self.end_token);
    
    // 输出: 单帧的 token 序列
}
```

---

## 四、序列结构差异

### Python 版本的训练序列

```
一个样本的完整序列:
[start][grid_0_tokens][end][start][grid_1_tokens][end]...[start][grid_N_tokens][end]

特点:
- 多个网格示例串联成一个长序列
- 每个网格都有独立的 start/end token
- 用于 In-Context Learning: 从多个示例学习规则
```

### Rust 版本的输出序列

```
每帧独立输出:
Frame 0: [start][patch_tokens][end]
Frame 1: [start][patch_tokens][end]
Frame 2: [start][patch_tokens][end]
...

特点:
- 每帧独立，需要外部串联
- 流式输出，适合大规模数据生成
- 需要额外处理才能形成训练序列
```

---

## 五、具体差异表格

| 维度 | Python 版本 | Rust 版本 | 影响 |
|------|-------------|-----------|------|
| **批量 vs 流式** | 批量生成 `(B, T, ...)` | 流式迭代器逐帧产出 | Rust 需外部收集 |
| **多示例串联** | 自动串联多个网格 | 每帧独立，不串联 | Rust 需外部串联 |
| **start/end token** | 每个网格独立添加 | 每帧独立添加 | 相同 |
| **n_groups 处理** | flatten 为 `(H, W, G*D)` | 交错存储 `(r,c,g)` | 紧凑性不同 |
| **数据类型** | JAX array (int32) | `FrameState::Integer(i64)` | Rust 更通用 |
| **温度默认值** | `temperature=0.0` | `temperature=1.0` | **重要差异！** |
| **权重初始化** | Flax 默认 | LeCun Normal | 可能产生不同规则 |
| **偏置初始化** | Flax 默认 | 全零 | 可能产生不同规则 |

---

## 六、最关键的差异

### 1. 温度参数默认值不同

```python
# Python: temperature=0.0 → argmax (确定性)
next_state = jax.random.categorical(rng, logits/temperature, axis=-1)
# temperature=0 时使用 argmax
```

```rust
// Rust: temperature=1.0 → 随机采样
let next_state = if temperature == 0.0 {
    // argmax
} else {
    // temperature > 0: 缩放后分类采样
    categorical_sample(&scaled, rng)
};
```

**影响**: 默认情况下，Python 版本产生确定性演化，Rust 版本产生随机演化。

### 2. 序列组织方式不同

Python 版本直接生成适合训练的序列格式：
```python
# 输入模型: [S][G0][E][S][G1][E]...[S][GN][E]
# 模型学习: 从 G0→G1, G1→G2, ... 推断规则
```

Rust 版本需要外部处理：
```rust
// 需要将多个帧串联:
// Frame0 → Frame1 → Frame2 → ...
// 然后添加 start/end token
```

---

## 七、数据流向对比图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Python 版本数据流                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  generate_nca_dataset()                                             │
│       │                                                             │
│       ↓                                                             │
│  sims: (B, T, H, W, C)  ← 批量张量                                  │
│       │                                                             │
│       ↓                                                             │
│  NCA_Tokenizer.encode_task()                                        │
│       │                                                             │
│       ↓                                                             │
│  tokens: (B, N*(grid_len))  ← 串联序列                              │
│       │                                                             │
│       ↓                                                             │
│  NCADataset  ← 直接用于 DataLoader                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Rust 版本数据流                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  NeuralCA2D.generate_stream()                                       │
│       │                                                             │
│       ↓                                                             │
│  Iterator<Item=SequenceFrame>  ← 流式逐帧                           │
│       │                                                             │
│       ↓                                                             │
│  PatchTokenizer.process()                                           │
│       │                                                             │
│       ↓                                                             │
│  Iterator<Item=SequenceFrame>  ← 每帧独立 tokenized                 │
│       │                                                             │
│       ↓                                                             │
│  需要外部收集和串联 ← 才能形成训练序列                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 八、训练数据格式详解

### Python 版本训练数据结构

```python
# NCADataset.__getitem__ 返回
def __getitem__(self, idx):
    seq = self.seq[idx]      # 输入序列
    targets = self.targets[idx]  # 目标序列
    
    # 1. 屏蔽负值 token
    target = torch.where(seq < 0, torch.tensor(-100), seq)
    
    # 2. 屏蔽最小示例数 (前 min_grid 个网格不计算损失)
    target[:(self.min_grid * self.grid_len)] = -100
    
    # 3. 自回归位移
    seq = seq[:-1]        # 输入: 去掉最后一个 token
    targets = target[1:]  # 目标: 去掉第一个 token
    
    return seq, targets
```

### 训练目标示意

```
输入序列:  [start][grid1][end][start][grid2][end]...[start][gridN
目标序列:  [grid1][end][start][grid2][end]...[start][gridN][end]

损失计算:
- 前 min_grid 个网格的目标设为 -100 (不计算损失)
- 只对后续网格预测计算损失
```

---

## 九、适配建议

如果要让 Rust 版本生成与 Python 版本兼容的训练数据，需要：

### 1. 调整温度参数

设置 `temperature=0.0` 以匹配 Python 默认行为：

```json
{
  "temperature": 0.0
}
```

### 2. 串联多帧

将多个 `SequenceFrame` 串联成一个训练序列。

### 3. 调整序列格式

```rust
// 目标格式:
// [start][frame0_patches][end][start][frame1_patches][end]...
// 而不是每帧独立的 [start][patches][end]
```

### 4. 添加序列级处理

创建一个新的 Processor 来串联多帧：

```rust
// 建议新增: SequenceStitcher 处理器
pub struct SequenceStitcherConfig {
    /// 每个序列包含的帧数
    pub frames_per_sequence: usize,
    /// 是否在每个序列开头添加 start token
    pub add_sequence_start: bool,
    /// 是否在每个序列结尾添加 end token
    pub add_sequence_end: bool,
}
```

---

## 十、参数对照表

### 默认参数对比

| 参数 | Python 默认值 | Rust 默认值 | 说明 |
|------|--------------|-------------|------|
| `grid` / `rows, cols` | 12 | 12 | 网格大小 |
| `d_state` / `num_colors` | 10 | 10 | 状态数/颜色数 |
| `patch` | 3 | 3 | Patch 大小 |
| `dT` | 1 | - | 时间步间隔 |
| `min_grid` | 1 | - | 最小示例数 |
| `temperature` | 0.0 | 1.0 | **不同！** |
| `identity_bias` | 0.0 | 0.0 | 恒等偏置 |
| `hidden_dim` | 16 | 16 | 隐藏层维度 |
| `conv_features` | 4 | 4 | 卷积特征数 |

### Token 计算公式

```
grid_len = (grid / patch)^2 + 2  # 每个 grid 的 token 数
start_token = num_colors^(patch^2)
end_token = start_token + 1
vocab_size = num_colors^(patch^2) + 2
```

---

## 十一、总结

Rust 版本与 Python 版本的主要差异在于：

1. **生成模式**: Python 批量生成，Rust 流式生成
2. **序列组织**: Python 自动串联多网格，Rust 每帧独立
3. **温度默认值**: Python `temperature=0.0` (确定性)，Rust `temperature=1.0` (随机)
4. **权重初始化**: 可能因初始化方法不同产生不同的规则

要让 Rust 版本兼容 Python 训练格式，需要：
- 添加一个序列串联处理器
- 调整默认温度参数为 0.0
- 确保权重初始化方法一致

---

## 参考文件

### Python 版本
- `Code/nca-pre-pretraining-main/utils/nca.py` - NCA 模拟器
- `Code/nca-pre-pretraining-main/utils/tokenizers.py` - Tokenizer
- `Code/nca-pre-pretraining-main/src/nca_ppt.py` - 训练脚本
- `Code/nca-pre-pretraining-main/utils/dataset_utils.py` - 数据集工具

### Rust 版本
- `src/generators/nca2d.rs` - NCA 生成器
- `src/pipeline/patch_tokenizer.rs` - Patch Tokenizer
- `src/core/frame.rs` - 数据结构定义
- `src/core/generator.rs` - 生成器接口
