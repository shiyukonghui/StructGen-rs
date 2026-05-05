# NCA 生成器功能对比文档

> 对比对象：**StructGen-rs** 项目中的 CA（元胞自动机）生成器 vs **nca-pre-pretraining-main** 中的 NCA（神经元胞自动机）系统

---

## 1. 项目定位与目标

| 维度 | StructGen-rs CA 生成器 | nca-pre-pretraining NCA 系统 |
|------|------------------------|-------------------------------|
| **核心目标** | 生成1D元胞自动机时间序列数据，作为结构化训练数据 | 利用NCA生成合成数据，对语言模型进行"预预训练"(Pre-Pretraining) |
| **应用场景** | 通用结构化数据生成框架中的内置生成器 | LLM训练管线中的合成数据前处理阶段 |
| **理论背景** | Wolfram 1D元胞自动机（规则0-255） | 神经元胞自动机（Game of Life的神经网络推广） |
| **研究论文** | 无（工程实现） | arXiv: 2603.10055 |
| **语言生态** | Rust（纯实现） | Python（JAX仿真 + PyTorch训练） |

---

## 2. 元胞自动机模型对比

### 2.1 维度与拓扑

| 特性 | StructGen-rs CA | nca-pre-pretraining NCA |
|------|-----------------|--------------------------|
| **维度** | 1D（一维线性格子） | 2D（二维网格） |
| **默认网格大小** | width=128（单行） | grid_size=12（12×12 方阵） |
| **状态空间** | 二值（Bool: true/false） | 多值离散（d_state=10，10种颜色） |
| **邻域类型** | 3邻域（左、中、右） | 3×3卷积邻域（9个输入格） |

### 2.2 演化规则

| 特性 | StructGen-rs CA | nca-pre-pretraining NCA |
|------|-----------------|--------------------------|
| **规则定义方式** | 查找表（LUT）：8位规则号 → 8个布尔输出 | 神经网络（可学习参数）：3×3 Conv → 1×1 Conv → ReLU → 1×1 Conv |
| **规则数量** | 256种固定规则（Wolfram编号0-255） | 理论上无限（随机初始化的神经网络参数定义不同规则） |
| **确定性** | 完全确定性（相同输入 → 相同输出） | 可配置：temperature=0时确定性，>0时随机采样 |
| **状态持久性** | 无（完全由邻域决定） | identity_bias参数控制当前状态的偏向 |
| **状态转移** | 布尔映射：`lut[neighborhood_index]` | 分类采样：`categorical((logits + state_oh*identity_bias)/temperature)` |

**关键差异**：StructGen-rs 使用固定的查找表规则，规则空间有限但确定性极强；NCA 使用随机初始化的神经网络作为转移函数，规则空间无限且可调节随机性。

### 2.3 边界条件

| 边界类型 | StructGen-rs CA | nca-pre-pretraining NCA |
|----------|-----------------|--------------------------|
| **周期性 (Periodic/Wrap)** | 支持 | 支持（`jnp.pad(x, pad_width=1, mode='wrap')`） |
| **固定 (Fixed)** | 支持（越界视为0） | 不支持 |
| **反射 (Reflective)** | 支持 | 不支持 |

### 2.4 NCANetwork 架构细节

NCA 的核心是一个 Flax 神经网络模块，结构如下：

```
输入: one_hot(state) → shape (H, W, d_state * n_groups)
  ↓
Wrap Padding → shape (H+2, W+2, d_state * n_groups)
  ↓
Conv2D(3×3, features=4, VALID) → shape (H, W, 4)
  ↓
Conv2D(1×1, features=16) → shape (H, W, 16)
  ↓
ReLU
  ↓
Conv2D(1×1, features=d_state*n_groups) → shape (H, W, d_state*n_groups)
  ↓
输出: logits for categorical distribution
```

对比 StructGen-rs CA 的规则计算：

```
输入: grid[center-1], grid[center], grid[center+1]（3个布尔值）
  ↓
3-bit索引计算: left*4 + center*2 + right → 索引 0-7
  ↓
查找表: lut[index] → 布尔输出
```

---

## 3. 数据生成流程对比

### 3.1 StructGen-rs CA 数据生成

```
种子 → SeedRng(PRNG)
  ↓
随机初始化1D网格（width个布尔值）
  ↓
循环演化：
  ├─ 计算每个格子的3邻域索引
  ├─ 查表得到下一状态
  ├─ 双缓冲交换
  └─ 输出 SequenceFrame { step_index, FrameData { Vec<FrameState> } }
  ↓
可选：限制序列长度 / 无限流
```

**特点**：
- 纯函数式迭代器（`std::iter::from_fn`）
- 惰性求值，内存占用 O(width)
- 单规则单轨迹生成
- 每帧输出 width 个 `FrameState::Bool`

### 3.2 nca-pre-pretraining NCA 数据生成

```
随机种子 → JAX PRNGKey
  ↓
采样NCA规则参数（NCANetwork.init()）
  ↓
初始化2D网格状态（categorical分布采样）
  ↓
批量仿真（jax.vmap）：
  ├─ 每条规则生成多条轨迹
  ├─ rollout_simulation: lax.scan扫描演化步
  ├─ 支持时间子采样（dT步间隔）
  └─ 输出 shape (num_sims, num_examples, H, W, 1)
  ↓
规则过滤（可选）：
  ├─ 计算 gzip 压缩率作为复杂性代理指标
  ├─ 过滤阈值 [threshold, upper_bound]
  └─ 保留中等复杂性的规则
  ↓
Token化：NCA_Tokenizer.encode_task()
  ├─ 网格分割为 patch×patch 块
  ├─ 每块编码为单token（基于num_colors进制的混合编码）
  ├─ 添加 START/END 特殊token
  └─ 输出 (seq, targets) 用于自回归训练
```

**特点**：
- JAX vmap 批量并行仿真
- 多规则多轨迹同时生成
- 内置复杂性过滤机制
- 自动 tokenize 为序列数据
- 支持多组（n_groups）独立演化

### 3.3 数据规模对比

| 参数 | StructGen-rs CA | nca-pre-pretraining NCA |
|------|-----------------|--------------------------|
| 单帧数据量 | width=128 个布尔值 = 16字节 | 12×12=144 个整数 ≈ 576字节 |
| 典型序列长度 | 用户自定义（可无限） | seq_len=1024 tokens |
| 规则数量 | 1种规则/次 | 16000种训练规则 + 2000种验证规则 |
| 轨迹数量 | 1条/次 | 500条/规则（训练），100条/规则（验证） |
| 总数据规模 | 千万帧级别 | 164M NCA tokens |

---

## 4. 配置参数对比

### 4.1 StructGen-rs CA 参数

```yaml
# YAML manifest 配置
extensions:
  rule: 30              # Wolfram 规则号 (0-255)
  width: 128            # 网格宽度
  boundary: "periodic"  # 边界条件: periodic | fixed | reflective
```

参数简洁，3个配置项即可定义完整的CA行为。

### 4.2 nca-pre-pretraining NCA 参数

```python
# NCA 模型参数
grid_size: 12           # 网格尺寸 (H=W)
d_state: 10             # 状态空间大小（颜色数）
n_groups: 1             # 独立演化组数
identity_bias: 0.0      # 状态持久性偏向
temperature: 0.0        # 采样温度（0=确定性）

# 数据生成参数
train_num_rules: 16000  # 训练规则数
train_num_sim: 500      # 每规则仿真数
val_num_rules: 2000     # 验证规则数
val_num_sim: 100        # 验证仿真数
seq_len: 1024           # 序列长度
patch: 2                # patch 大小
dT: 1                   # 时间采样步长
init_rollout_steps: 10  # 初始burn-in步数

# 规则过滤参数
filter_rules: False     # 是否启用过滤
filter_rules_threshold: 0.4    # 最低压缩率
filter_rules_upper_bound: 1.0  # 最高压缩率
filter_rules_mode: 'gzip'      # 复杂性度量方式

# 模型参数
n_layer: 24             # Transformer层数
n_head: 32              # 注意力头数
n_embd: 2048            # 嵌入维度
vocab_size: 64000       # 输出词表大小
input_vocab_size: 10002 # 输入词表大小
```

参数丰富，涵盖NCA模型、数据生成、规则过滤、训练模型等多个层面。

---

## 5. 输出格式对比

### 5.1 StructGen-rs CA 输出

| 输出格式 | 描述 |
|----------|------|
| **Parquet** | Apache Parquet 列式存储，Arrow schema |
| **Text** | Unicode文本，带分隔符和表头 |
| **Binary** | 紧凑二进制编码 |

数据结构：
```rust
SequenceFrame {
    step_index: u64,           // 时间步编号
    state: FrameData {
        values: Vec<FrameState::Bool>  // width个布尔值
    },
    label: Option<String>,     // 可选语义标签
}
```

### 5.2 nca-pre-pretraining NCA 输出

| 输出格式 | 描述 |
|----------|------|
| **JAX Array** | 原始仿真数组 (num_sims, num_examples, H, W, 1) |
| **Token序列** | 经NCA_Tokenizer编码的 (seq, targets) 张量对 |
| **RGB图像** | 经render_state可视化的RGB图像 |

数据结构：
```python
# 仿真原始数据
sims: jnp.ndarray  # shape (B, N, H, W, 1)

# Token化后
seq: torch.Tensor    # shape (B, L) 输入序列
targets: torch.Tensor # shape (B, L) 目标序列（next-token prediction）
```

---

## 6. 核心功能差异分析

### 6.1 StructGen-rs CA 独有功能

| 功能 | 说明 |
|------|------|
| **多边界条件** | 支持 Periodic/Fixed/Reflective 三种边界，NCA 仅支持 Periodic |
| **流式生成** | 惰性迭代器，支持无限流，内存占用极小 |
| **后处理管线** | 内置6种后处理器：normalizer, dedup, diff_encoder, token_mapper, clip_stitcher, null_proc |
| **多种输出格式** | Parquet/Text/Binary 三种内置输出适配器 |
| **并行调度** | 基于 Rayon 的多线程并行生成，支持分片（shard） |
| **YAML声明式配置** | 通过 manifest 文件声明任务，批量执行 |
| **确定性种子派生** | SHA-256 基于种子的确定性分片派生 |
| **元数据记录** | 自动生成 metadata.json 和结构化日志 |

### 6.2 nca-pre-pretraining NCA 独有功能

| 功能 | 说明 |
|------|------|
| **神经网络转移函数** | 用可学习参数的CNN替代固定规则，规则空间无限 |
| **2D网格演化** | 12×12二维网格，3×3卷积邻域 |
| **多状态空间** | 支持10种（或任意d_state种）离散状态，远超二值 |
| **随机性控制** | temperature参数控制采样随机性，identity_bias控制状态惯性 |
| **规则复杂性过滤** | gzip压缩率作为Kolmogorov复杂性代理，筛选"有趣"规则 |
| **NCA Token化** | patch-based编码：2×2/3×3块 → 单token，支持START/END标记 |
| **批量仿真** | JAX vmap并行生成多规则多轨迹 |
| **下游训练管线** | 完整的预预训练→预训练→微调→评估链路 |
| **多组演化** | n_groups参数支持多组独立的NCA同时演化 |
| **可视化渲染** | 内置render_state函数，支持固定/可学习颜色映射 |

### 6.3 功能交集

| 共同功能 | StructGen-rs CA | nca-pre-pretraining NCA |
|----------|-----------------|--------------------------|
| **元胞自动机演化** | 1D Wolfram规则 | 2D 神经网络规则 |
| **周期性边界** | 支持 | 支持 |
| **确定性种子** | SeedRng (Xorshift64) | JAX PRNGKey |
| **时间序列输出** | SequenceFrame 迭代器 | JAX scan 滚动输出 |
| **网格状态快照** | 每帧完整状态 | 每步完整状态 |

---

## 7. 技术栈对比

| 维度 | StructGen-rs CA | nca-pre-pretraining NCA |
|------|-----------------|--------------------------|
| **核心语言** | Rust | Python |
| **数值计算** | 原生Rust实现 | JAX (GPU加速) |
| **并行策略** | Rayon (CPU多线程) | JAX vmap (SIMD/GPU并行) |
| **序列化** | serde_json + serde_yaml | PyTorch tensors + JAX arrays |
| **输出存储** | Parquet (Arrow) / Text / Binary | PyTorch DataLoader / JAX arrays |
| **测试** | Rust内置 #[test] (13个测试) | Python脚本式测试 |
| **配置管理** | YAML manifest + clap CLI | Python dataclass + argparse |

---

## 8. 性能特征对比

| 维度 | StructGen-rs CA | nca-pre-pretraining NCA |
|------|-----------------|--------------------------|
| **单步计算复杂度** | O(width) — 纯查表 | O(H×W×k²) — 卷积神经网络 |
| **内存占用** | O(width) — 双缓冲 | O(H×W×d_state×n_groups) — 全网格 |
| **吞吐量** | 极高（纯CPU整数运算） | 中等（需要GPU加速的卷积运算） |
| **初始化开销** | 近零（查表预计算8项） | 高（神经网络参数采样和JIT编译） |
| **扩展性** | 线性（增加线程/分片） | 批量并行（增加GPU批次） |

---

## 9. 设计哲学差异

### StructGen-rs CA

- **工程导向**：作为通用数据生成框架的一个组件，追求高性能、低开销、可组合
- **简洁性**：3个参数即可完整定义CA行为，规则空间有限但足够
- **流式架构**：惰性迭代器避免全量物化，适合大规模数据生成
- **可组合性**：与后处理器、输出适配器无缝集成，形成完整管线
- **确定性保证**：同种子必同输出，适合可复现实验

### nca-pre-pretraining NCA

- **研究导向**：服务于LLM预预训练的学术研究，追求效果验证
- **丰富性**：无限规则空间，通过随机神经网络参数产生多样动力学
- **批量架构**：vmap并行仿真，适合一次性生成大量训练数据
- **端到端**：从NCA仿真到token化到模型训练的完整链路
- **复杂性调控**：gzip过滤机制主动选择"有趣"的规则

---

## 10. 互补性与整合方向

### 10.1 当前互补关系

```
                    ┌─────────────────────────────────────┐
                    │      nca-pre-pretraining-main        │
                    │                                     │
                    │  NCA仿真 ──→ Token化 ──→ LM训练     │
                    │  (JAX)      (Python)    (PyTorch)    │
                    └─────────────┬───────────────────────┘
                                  │ 合成数据
                                  ↓
                    ┌─────────────────────────────────────┐
                    │         StructGen-rs                  │
                    │                                     │
                    │  CA生成 ──→ 后处理 ──→ 格式输出      │
                    │  (Rust)    (Pipeline)   (Sink)       │
                    └─────────────────────────────────────┘
```

### 10.2 潜在整合方向

1. **Rust化NCA引擎**：将NCA的JAX仿真逻辑移植为Rust实现，集成到StructGen-rs的生成器框架中
   - 优势：消除Python/JAX依赖，获得原生性能
   - 挑战：需要实现2D卷积、分类采样、vmap等价逻辑

2. **CA→NCA升级路径**：将现有1D CA生成器扩展为2D NCA生成器
   - 1D CA (Wolfram规则) → 2D CA (如Game of Life) → 2D NCA (神经网络规则)
   - 可渐进式实现，复用现有Generator trait和管线架构

3. **复杂性过滤功能**：将gzip复杂性过滤机制引入StructGen-rs管线
   - 作为后处理器实现：评估生成序列的复杂性，过滤低质量输出
   - 与现有的 `dedup`、`token_mapper` 等处理器协同工作

4. **Token化适配**：将NCA的patch-based token化策略作为StructGen-rs的后处理器
   - 扩展 `token_mapper` 支持2D patch编码
   - 支持 START/END 特殊token
   - 输出格式兼容 PyTorch DataLoader

---

## 11. 总结

| 维度 | StructGen-rs CA | nca-pre-pretraining NCA | 差异程度 |
|------|-----------------|--------------------------|----------|
| **自动机维度** | 1D | 2D | 高 |
| **规则表达力** | 256种固定规则 | 无限种神经规则 | 高 |
| **状态空间** | 二值 | 多值离散 | 高 |
| **随机性** | 无 | 可配置 | 中 |
| **数据规模** | 中小（单规则流式） | 大规模（批量生成） | 中 |
| **工程成熟度** | 高（测试/文档/管线） | 中（研究原型） | 中 |
| **训练集成** | 无（纯数据生成） | 完整（预预训练管线） | 高 |
| **性能** | 极高（纯CPU查表） | 中（GPU卷积） | 中 |
| **可扩展性** | 高（trait + 工厂） | 中（硬编码管线） | 低 |

**核心结论**：两个系统服务于不同层面的需求——StructGen-rs CA 是一个高性能的1D CA数据生成器，注重工程质量和管线可组合性；nca-pre-pretraining NCA 是一个完整的2D NCA研究平台，注重规则多样性、训练效果验证和端到端流程。两者在CA维度、规则表达力和训练集成度上差异显著，但在周期性边界、确定性演化、时间序列输出等基础功能上存在交集，具备整合的技术基础。
