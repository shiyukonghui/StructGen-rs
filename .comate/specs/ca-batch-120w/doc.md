# CA 批量数据生成 — 19 种规则各 120 万条

## 需求场景

使用 StructGen-rs 项目中所有可用的元胞自动机（CA）规则，**每种规则分别生成 120 万条**数据样本，输出为 **Npy 格式**，便于直接对接 PyTorch/TensorFlow 等框架的模型训练管线（`numpy.load()` 即可加载，无需额外依赖）。

## 可用规则清单

### 1D 元胞自动机（5 种）
| 序号 | 规则名称 | 规则号 | 特征 |
|------|----------|--------|------|
| 1 | rule30 | 30 | 混沌/伪随机 |
| 2 | rule54 | 54 | 复杂行为 |
| 3 | rule90 | 90 | 谢尔宾斯基三角 |
| 4 | rule110 | 110 | 图灵完备 |
| 5 | rule184 | 184 | 交通流模型 |

### 2D 元胞自动机（10 种，排除 xrule 因需要外部查找表）
| 序号 | 规则名称 | 规则类型 | 核心参数 |
|------|----------|----------|----------|
| 1 | game_of_life | LifeLike | B3/S23 |
| 2 | highlife | LifeLike | B36/S23 |
| 3 | day_night | LifeLike | B3678/S34678 |
| 4 | seeds | LifeLike | B2/S |
| 5 | diamoeba | LifeLike | B35678/S5678 |
| 6 | replicator_2d | LifeLike | B1357/S1357 |
| 7 | maze | LifeLike | B3/S12345 |
| 8 | wireworld | WireWorld | 4 状态 |
| 9 | cyclic | Cyclic | n_states=14, threshold=1 |
| 10 | ameyalli | Hensel | B2ci3ar4krtz5cq6c7ce/S01e2ek3qj4kt5ceayq6cki7c8 |

### 3D 元胞自动机（3 种）
| 序号 | 规则名称 | 规则类型 | 核心参数 |
|------|----------|----------|----------|
| 1 | life_3d | LifeLike | B567/S56 |
| 2 | cyclic_3d | Cyclic | n_states=14, threshold=1 |
| 3 | fredkin_3d | Fredkin | 奇偶自复制 |

### NCA2D 神经元胞自动机（1 种）
| 序号 | 生成器 | 特征 |
|------|--------|------|
| 1 | nca2d | 随机权重初始化，每种子生成独特规则 |

**总计：19 种规则，每种 120 万条 = 2,280 万条数据**

## 技术方案

### 实现方式
创建一个完整的 YAML 清单文件 `ca_all_rules_120w.yaml`，包含 19 个 task，每个 task 配置一种 CA 规则，count=1,200,000。

通过 `structgen-rs run --manifest ca_all_rules_120w.yaml` 命令执行批量生成。

### 参数设计

#### 1D CA 参数（5 个 task）
- `seq_length: 128`（演化步数）
- `width: 128`（网格宽度）
- `boundary: "periodic"`

#### 2D CA 参数（10 个 task）
- `seq_length: 64`（演化步数）
- `rows: 32, cols: 32`（网格尺寸）
- `boundary: "periodic"`
- `neighborhood: "moore"`
- `init_mode: "random"`

#### 3D CA 参数（3 个 task）
- `seq_length: 32`（演化步数）
- `depth: 8, rows: 8, cols: 8`（网格尺寸）
- `boundary: "periodic"`
- `neighborhood: "moore"`
- `init_mode: "random"`

#### NCA2D 参数（1 个 task）
- `seq_length: 64`
- `rows: 12, cols: 12`
- `d_state: 10, n_groups: 1`
- `temperature: 1.0, identity_bias: 0.0`
- `hidden_dim: 16, conv_features: 4`

### 全局配置
- `output_dir`: `"F:/RustProjects/StructGen-rs/output_ca_120w"`
- `default_format`: `"Npy"`
- `num_threads`: 8
- `stream_write`: true
- `shard_max_sequences`: 10000

### 种子分配
每种规则使用不同的基础种子，确保数据多样性：
- 1D 规则：seed 从 1001 递增（rule30=1001, rule54=1002, ...）
- 2D 规则：seed 从 2001 递增
- 3D 规则：seed 从 3001 递增
- NCA2D：seed=4001

## 影响文件

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `ca_all_rules_120w.yaml`（新建） | 新增 | 19 个 task 的完整批量生成清单 |

无需修改任何源代码，仅创建 YAML 清单文件。

## 边界条件与异常处理

1. **xrule 规则**：项目预设中 xrule 需要外部 512 位查找表，无法通过 preset 直接使用，故排除
2. **3D CA 性能**：3D 元胞自动机计算量较大，网格尺寸控制在 8x8x8 以平衡速度与数据质量
3. **NCA2D 性能**：神经网络推理较慢，网格尺寸使用默认 12x12
4. **磁盘空间**：Npy 为原始数值存储，无压缩，预估总量约 100-200 GB（相比 Parquet 略大，但加载速度更快、无需解压）
5. **分片大小**：使用 shard_max_sequences=10000，确保内存可控且并行粒度合理

## 数据流路径

```
YAML 清单 → Manifest 解析 → Task 分片 → Rayon 并行执行
  → 每个 shard: Generator 生成 → Pipeline 处理(空) → Sink 写出 Npy
  → 原子写入（.tmp → rename）→ metadata.json 汇总
```

## 预期产出

- 19 个子目录，每个包含约 120 个 Npy 分片文件（120万/10000=120 shards）
- 1 个 metadata.json 汇总文件
- 总数据量约 2,280 万条样本
- 训练时可直接 `np.load()` 加载，配合 `np.memmap` 支持大规模数据按需读取
