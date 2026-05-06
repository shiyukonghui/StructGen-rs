# StructGen-rs

> 高性能程序化结构化数据生成器 — 为机器学习与人工智能研究生成高质量时序数据集

---

## 项目概述

StructGen-rs 是一个基于 Rust 的**可扩展程序化数据生成框架**，专注于生成用于 ML/AI 训练与评估的**时序结构数据集**。它通过确定性随机种子和插件化生成器架构，在保证**可复现性**的同时支持大规模并行生成。

### 核心特性

- **12 种内置生成器**：元胞自动机 (1D/2D/3D)、神经元胞自动机 (NCA)、混沌系统 (Lorenz/Logistic)、N 体引力模拟、形式语言 (LSystem/形式文法)、虚拟执行机 (AlgorithmVM)、布尔网络、IFS 分形
- **3 种输出格式**：Parquet (列式)、Text (Unicode 文本)、Binary (紧凑二进制)
- **6 种后处理器**：标准化、去重、差分编码、令牌映射、截断拼接、透传
- **确定性复现**：基于 SHA2 的种子派生，同一配置 + 种子 → 完全一致的输出
- **并行调度**：Rayon 线程池 + 任务分片，充分利用多核 CPU
- **声明式清单**：YAML 描述生成任务，CLI 一键执行
- **流式处理**：惰性迭代器链，避免内存中积累大量中间数据
- **完整元数据**：每次运行产出 `metadata.json`，包含配置和运行时统计

---

## 环境要求与安装

### 前置条件

- **Rust** 工具链 1.70+ （通过 [rustup](https://rustup.rs/) 安装）
- Windows / Linux / macOS

### 编译

```bash
git clone <repository-url>
cd StructGen-rs

# Debug 构建
cargo build

# Release 构建（推荐用于生产）
cargo build --release
```

编译产物位于 `target/release/structgen-rs`（Linux/macOS）或 `target/release/structgen-rs.exe`（Windows）。

---

## 快速入门

### 1. 编写任务清单

创建一个 `tasks.yaml` 文件：

```yaml
global:
  output_dir: "./output"
  default_format: "Parquet"
  num_threads: 4

tasks:
  - name: "rule30_ca"
    generator: "ca"
    params:
      seq_length: 1000
      extensions:
        rule: 30
        width: 128
    count: 100
    seed: 12345
```

> 详细语法参考下方 [YAML 清单格式](#yaml-清单格式)

### 2. 运行生成

```bash
structgen-rs --manifest tasks.yaml
```

### 3. 查看输出

```
output/
├── rule30_ca_00000_0000000000003039.parquet
├── rule30_ca_00001_...
├── ...
└── metadata.json          # 运行时元数据
```

---

## 命令行参数

```
structgen-rs --manifest <FILE> [OPTIONS]
```

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--manifest <FILE>` | `-m` | **必需** 任务清单 YAML 文件路径 | — |
| `--output-dir <DIR>` | `-o` | 覆盖清单中的 `output_dir` | 清单的值 |
| `--format <FORMAT>` | `-f` | 覆盖全局默认输出格式：`parquet` / `text` / `binary` | 清单的值 |
| `--num-threads <N>` | `-t` | 并行线程数 | CPU 核心数 |
| `--log-level <LEVEL>` | `-l` | 日志级别：`trace` / `debug` / `info` / `warn` / `error` | `info` |
| `--stream-write` | — | 强制流式写出模式（可选 `false` 关闭） | `true` |
| `--no-progress` | — | 禁用进度条（适用于 CI 管道） | `false` |
| `--help` | `-h` | 显示帮助信息 | — |
| `--version` | `-V` | 显示版本号 | — |

### 退出码

| 退出码 | 含义 |
|--------|------|
| `0` | 全部任务成功完成 |
| `1` | 部分任务失败（至少有一个成功） |
| `2` | 全部任务失败或启动阶段致命错误 |

---

## YAML 清单格式

清单（Manifest）是描述整个生成任务的 YAML 文件，包含全局配置和任务列表两部分。

### 完整结构

```yaml
global:
  output_dir: "./output"         # 输出根目录 (必需)
  default_format: "Parquet"      # 全局默认格式: Parquet / Text / Binary
  num_threads: 4                 # 线程数 (省略 = 自动检测)
  log_level: "info"              # 日志级别: trace / debug / info / warn / error
  shard_max_sequences: 10000     # 单文件最大序列数
  stream_write: true             # 流式写出模式

tasks:
  - name: "task1"                # 任务名称 (必需，必须唯一)
    generator: "ca"              # 生成器名称 (必需)
    params:                      # 生成参数
      seq_length: 1000           # 目标序列长度 (帧数)
      grid_size:                 # 网格尺寸 (可选)
        rows: 32
        cols: 32
      extensions: {}             # 生成器专用参数 (见各生成器文档)
    count: 100                   # 样本数量 (必需，>0)
    seed: 12345                  # 基础随机种子 (必需)
    pipeline: []                 # 处理器链 (可选，按顺序应用)
    output_format: "Text"        # 任务级格式覆盖 (可选)
    shard_size: 50               # 分片大小 (可选，默认自动)
```

### `extensions` 字段说明

`extensions` 是一个扁平键值映射，直接传参给生成器工厂函数。每个生成器支持的 key 不同，详见下方[生成器参考](#生成器参考)。

```yaml
extensions:
  rule: 30          # CA 的 Wolfram 规则号
  sigma: 10.0       # Lorenz 的 sigma 参数
  r: 3.9            # Logistic 的生长率
  num_bodies: 5     # NBody 的天体数
```

---

## 生成器参考

StructGen-rs 内置 **12 种生成器**，每种可同时通过短名和长名注册。

### 生成器注册表

| 生成器 | 注册名 (短名 / 长名) | 输出内容 | 输出类型 |
|--------|---------------------|----------|----------|
| 元胞自动机 | `ca` / `cellular_automaton` | 规则演化网格 | Bool |
| 洛伦兹系统 | `lorenz` / `lorenz_system` | 三维混沌轨迹 | Float |
| 逻辑斯蒂映射 | `logistic` / `logistic_map` | 一维混沌序列 | Float |
| N 体引力模拟 | `nbody` / `nbody_sim` | 天体位置与速度 | Float |
| L 系统 | `lsystem` | 字符串重写 | Integer |
| 算法虚拟机 | `vm` / `algorithm_vm` | 程序执行状态 | Integer |
| 布尔网络 | `boolean_network` | 网络状态向量 | Bool |
| 迭代函数系统 | `ifs` | 分形点集 | Float |
| 形式文法 | `formal_grammar` | 推导序列 | Integer |
| 2D 元胞自动机 | `ca2d` / `cellular_automaton_2d` | 2D 网格演化 | Integer |
| 3D 元胞自动机 | `ca3d` / `cellular_automaton_3d` | 3D 网格演化 | Integer |
| 2D 神经元胞自动机 | `nca2d` / `neural_cellular_automaton_2d` | NCA 网格演化 | Integer |

---

### 支持的著名规则

**1D 规则（ca 生成器）**：
- Rule 30：混沌、伪随机
- Rule 90：谢尔宾斯基三角形
- Rule 110：图灵完备
- Rule 184：交通流模型

**2D 规则（ca2d 生成器）**：
- Conway's Game of Life (B3/S23)：经典生命游戏
- HighLife (B36/S23)：支持复制体
- WireWorld：4 状态电子线路模拟
- Day & Night (B3678/S34678)：对称规则
- 循环 CA：螺旋波形态

**3D 规则（ca3d 生成器）**：
- 3D Life (B567/S56)：3D 滑翔机
- 3D 循环 CA：嵌套壳状结构
- Fredkin 规则：自我复制

---

### `ca` — 一维元胞自动机

使用 Wolfram 规则进行状态演化，输出每一时间步的完整网格状态。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rule` | `u8` | `30` | Wolfram 规则号（0–255） |
| `width` | `usize` | `128` | 网格宽度（每帧的元胞数） |
| `boundary` | `string` | `"periodic"` | 边界条件：`periodic`（环形）、`fixed`（固定 0）、`reflective`（镜面反射） |

**示例清单：**

```yaml
- name: "rule110"
  generator: "ca"
  params:
    seq_length: 500
    extensions:
      rule: 110
      width: 256
      boundary: "periodic"
  count: 50
  seed: 42
```

---

### `lorenz` — 洛伦兹混沌吸引子

经典三维连续混沌系统 dx/dt = σ(y−x), dy/dt = x(ρ−z)−y, dz/dt = xy−βz，使用四阶 Runge-Kutta (RK4) 积分。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sigma` | `f64` | `10.0` | 普朗特数 |
| `rho` | `f64` | `28.0` | 瑞利数 |
| `beta` | `f64` | `2.666…` | 阻尼系数 |
| `dt` | `f64` | `0.01` | 积分步长（必须 > 0） |

**示例：**

```yaml
- name: "lorenz_chaos"
  generator: "lorenz_system"
  params:
    seq_length: 2000
    extensions:
      sigma: 10.0
      rho: 28.0
      beta: 2.667
      dt: 0.01
  count: 100
  seed: 67890
```

---

### `logistic` — 逻辑斯蒂映射

一维离散混沌映射：x_{n+1} = r·x_n·(1−x_n)。每帧输出当前 x 值。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `r` | `f64` | `3.9` | 生长率参数（必须在 [0, 4] 内） |

**示例：**

```yaml
- name: "logistic_chaos"
  generator: "logistic"
  params:
    seq_length: 500
    extensions:
      r: 3.99
  count: 100
  seed: 123
```

> **高级用法**：可通过 `params.extensions` 中的 `logistic` 嵌套键设置初始值 x0（详见源码 `src/generators/logistic.rs`）。

---

### `nbody` — N 体引力模拟

牛顿引力模拟，用软化因子防止近距离奇点。支持欧拉法和四阶龙格-库塔法两种积分器。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_bodies` | `usize` | `5` | 天体数量（> 0） |
| `dt` | `f64` | `0.01` | 积分步长（> 0） |
| `softening` | `f64` | `0.1` | 软化因子 ε |
| `integrator` | `string` | `"rk4"` | 积分器：`"euler"` 或 `"rk4"` |

**每帧输出**：(px, py, vx, vy) × num_bodies 个 `Float` 值。

**示例：**

```yaml
- name: "3body_sim"
  generator: "nbody"
  params:
    seq_length: 1000
    extensions:
      num_bodies: 3
      dt: 0.005
      softening: 0.05
      integrator: "rk4"
  count: 50
  seed: 777
```

---

### `lsystem` — L 系统 (Lindenmayer 系统)

上下文无关字符串重写系统。从公理开始，每次迭代将符号串中匹配产生式规则的字符替换为对应字符串。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `axiom` | `string` | `"F"` | 公理（起始符号串，不能为空） |
| `rules` | `map<char, string>` | 空 | 产生式规则 |
| `iterations` | `usize` | `5` | 最大迭代次数 |

**示例（Koch 曲线）：**

```yaml
- name: "koch_curve"
  generator: "lsystem"
  params:
    seq_length: 6              # 迭代 6 步
  count: 10
  seed: 0                      # L 系统完全确定性（种子仅影响其他操作）
```

> **注意**：L 系统是确定性的（不使用种子）；当 `rules` 为空时使用默认规则 `F -> "F+F-F-F+F"`。

---

### `vm` — 算法虚拟机

精简指令集的栈式虚拟机。随机生成程序，逐指令执行并输出内部状态。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `program_size` | `usize` | `32` | 随机生成的指令条数 |
| `max_steps` | `usize` | `1000` | 最大执行步数（防无限循环） |

**指令集：**

| 指令 | 操作 | 说明 |
|------|------|------|
| `Push(i64)` | 压栈 | 将立即数压入栈顶 |
| `Pop` | 出栈 | 弹出栈顶值 |
| `Add` | 加法 | 弹出两个值，压入和 |
| `Sub` | 减法 | 弹出两个值，压入差 |
| `Mul` | 乘法 | 弹出两个值，压入积 |
| `Div` | 除法 | 弹出两个值，压入商（除零安全） |
| `Dup` | 复制 | 复制栈顶值 |
| `Swap` | 交换 | 交换栈顶两个值 |
| `Jmp(i64)` | 无条件跳转 | PC += offset |
| `Jz(i64)` | 条件跳转 | 若栈顶 == 0 则跳转 |
| `Halt` | 停机 | 终止执行 |

**每帧输出**：PC + 8 个通用寄存器 + 栈顶值（共 10 个 `Integer`）。

**示例：**

```yaml
- name: "vm_execution"
  generator: "algorithm_vm"
  params:
    seq_length: 200
  count: 100
  seed: 555
```

---

### `boolean_network` — 随机布尔网络

N 个节点的布尔网络。每个节点有唯一的 3 输入布尔函数和随机的 3 个上游节点。同步更新后输出全部节点状态。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_nodes` | `usize` | `32` | 网络节点数（> 0） |

**示例：**

```yaml
- name: "random_grn"
  generator: "boolean_network"
  params:
    seq_length: 300
    extensions:
      num_nodes: 64
  count: 20
  seed: 111
```

---

### `ifs` — 迭代函数系统

使用随机迭代函数系统生成分形点集。每次从多个仿射变换中按概率选择一个，应用到当前点。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_transforms` | `usize` | `4` | 仿射变换数量（> 0） |

**每帧输出**：`Float(x), Float(y)` 二维坐标。

**示例：**

```yaml
- name: "barnsley_fern"
  generator: "ifs"
  params:
    seq_length: 5000
    extensions:
      num_transforms: 4
  count: 10
  seed: 42
```

---

### `formal_grammar` — 形式文法

上下文无关文法的推导过程。从起始符号开始，每次在最左侧非终结符上随机选择一个产生式展开。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `productions` | `map<char, string[]>` | 空 | 产生式规则 |
| `start_symbol` | `char` | `'S'` | 起始非终结符 |
| `max_derivations` | `usize` | `100` | 最大推导次数 |

**示例：**

```yaml
- name: "context_free"
  generator: "formal_grammar"
  params:
    seq_length: 100
    extensions:
      start_symbol: "S"
      max_derivations: 200
  count: 20
  seed: 333
```

---

### `ca2d` — 2D 元胞自动机

支持 Life-like、Totalistic、WireWorld、Cyclic、Hensel、LookupTable 六种规则类型的 2D 元胞自动机。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rule_type` | `String` | `"lifelike"` | 规则类型：`lifelike` / `totalistic` / `wireworld` / `cyclic` / `hensel` / `lookuptable` |
| `birth` | `[u8]` | `[3]` | Life-like 出生条件（默认 Conway's Game of Life） |
| `survival` | `[u8]` | `[2, 3]` | Life-like 存活条件 |
| `totalistic_table` | `[u8]` | — | Totalistic 规则查表 |
| `d_state` | `u8` | `2` | 状态数 |
| `rows` | `usize` | `64` | 网格行数 |
| `cols` | `usize` | `64` | 网格列数 |
| `boundary` | `String` | `"periodic"` | 边界条件：`periodic` / `fixed` / `reflective` |
| `neighborhood` | `String` | `"moore"` | 邻域类型：`moore` / `vonneumann` |
| `init_mode` | `String` | `"random"` | 初始化模式：`random` / `single_center` |
| `hensel_notation` | `String` | — | Hensel 记法（如 `"B36/S23"`，仅 hensel 规则类型使用） |
| `lookup_table_hex` | `String` | — | 十六进制查找表（128 字符，仅 lookuptable 规则类型使用） |
| `n_states` | `u8` | `14` | 循环 CA 状态数（仅 cyclic 规则类型使用） |
| `threshold` | `u8` | `1` | 循环 CA 阈值（仅 cyclic 规则类型使用） |

**示例：**

```yaml
# Conway's Game of Life
- name: "ca_2d_life"
  generator: "ca2d"
  params:
    seq_length: 64
    extensions:
      rule_type: "lifelike"
      birth: [3]
      survival: [2, 3]
      rows: 32
      cols: 32
  count: 400000
  seed: 2024

# WireWorld 电子模拟
- name: "ca_2d_wireworld"
  generator: "ca2d"
  params:
    seq_length: 100
    extensions:
      rule_type: "wireworld"
      rows: 32
      cols: 32
  count: 10000
  seed: 100

# 循环 CA 螺旋波
- name: "ca_2d_cyclic"
  generator: "ca2d"
  params:
    seq_length: 200
    extensions:
      rule_type: "cyclic"
      n_states: 14
      threshold: 1
      rows: 64
      cols: 64
  count: 10000
  seed: 200
```

---

### `ca3d` — 3D 元胞自动机

支持 Life-like、Totalistic、Cyclic、Fredkin 四种规则类型的 3D 元胞自动机。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rule_type` | `String` | `"lifelike"` | 规则类型：`lifelike` / `totalistic` / `cyclic` / `fredkin` |
| `birth` | `[u8]` | `[5, 6, 7]` | 3D Life-like 出生条件 |
| `survival` | `[u8]` | `[5, 6]` | 3D Life-like 存活条件 |
| `d_state` | `u8` | `2` | 状态数 |
| `depth` | `usize` | `16` | 网格深度 |
| `rows` | `usize` | `16` | 网格行数 |
| `cols` | `usize` | `16` | 网格列数 |
| `boundary` | `String` | `"periodic"` | 边界条件：`periodic` / `fixed` / `reflective` |
| `neighborhood` | `String` | `"moore"` | 邻域类型：`moore`(26邻居) / `vonneumann`(6邻居) |
| `init_mode` | `String` | `"random"` | 初始化模式：`random` / `single_center` |
| `n_states` | `u8` | `14` | 循环 CA 状态数（仅 cyclic 规则类型使用） |
| `threshold` | `u8` | `1` | 循环 CA 阈值（仅 cyclic 规则类型使用） |

**示例：**

```yaml
# 3D Life
- name: "ca_3d_life"
  generator: "ca3d"
  params:
    seq_length: 32
    extensions:
      rule_type: "lifelike"
      birth: [5, 6, 7]
      survival: [5, 6]
      depth: 8
      rows: 8
      cols: 8
  count: 400000
  seed: 2025
```

---

### `nca2d` — 2D 神经元胞自动机

纯 Rust 实现的 2D 神经元胞自动机（Neural Cellular Automaton），每个规则约 584 个参数。通过小型神经网络学习局部转换规则。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `d_state` | `u8` | `10` | 状态数/颜色数 |
| `n_groups` | `u8` | `1` | 规则组数 |
| `rows` | `usize` | `12` | 网格行数 |
| `cols` | `usize` | `12` | 网格列数 |
| `identity_bias` | `f64` | `0.0` | 身份偏置 |
| `temperature` | `f64` | `0.0` | 采样温度（0.0 为确定性模式） |
| `hidden_dim` | `usize` | `16` | 隐藏层维度 |
| `conv_features` | `usize` | `4` | 卷积特征数 |

**示例：**

```yaml
# 2D NCA 神经元胞自动机
- name: "nca_2d_gen"
  generator: "nca2d"
  params:
    seq_length: 50
    extensions:
      d_state: 10
      rows: 12
      cols: 12
      hidden_dim: 16
      conv_features: 4
      temperature: 0.0
  count: 10000
  seed: 300
```

---

## 后处理管道

管道（Pipeline）在生成器输出和文件写入之间，对帧序列进行变换处理。处理器按 `pipeline` 列表顺序串联执行。

### 可用处理器

| 名称 | 功能 |
|------|------|
| `null` | 透传处理器，不做任何变换 |
| `normalizer` | 将 Float/Bool 值归一化到整数范围 (0–255) |
| `dedup` | 去除连续重复帧和全零帧，支持最小熵过滤 |
| `diff_encoder` | 计算相邻帧的差分，减小数据冗余 |
| `token_mapper` | 将整数映射到 Unicode 码点，生成可读文本 |
| `clip_stitcher` | 将序列截断为固定长度并拼接 |

### 使用示例

```yaml
pipeline: []                                                  # 空管道（直接输出原始数据）

pipeline: ["normalizer"]                                       # 单处理器

pipeline: ["normalizer", "token_mapper"]                       # 标准化→文本映射

pipeline: ["normalizer", "dedup", "diff_encoder", "token_mapper"]  # 全链处理
```

---

## 输出格式

| 格式 | 说明 | 文件扩展名 | 典型用途 |
|------|------|-----------|----------|
| **Parquet** | Apache Parquet 列式存储，自带压缩 | `.parquet` | 大规模数据分析、DataFrame |
| **Text** | Unicode 文本（配合 token_mapper） | `.txt` | 语言模型 DataLoader |
| **Binary** | 紧凑二进制原始转储 | `.bin` | 高性能 mmap 随机访问 |

可以在全局配置中设置默认格式，也可在单个任务中覆盖。命令行 `--format` 可覆盖全局默认值。

---

## 添加新生成器

StructGen-rs 的生成器系统基于 **注册表模式**，添加新生成器只需实现 `Generator` trait 并注册到 `GeneratorRegistry`。

### 步骤 1：实现 `Generator` trait

```rust
// src/generators/my_generator.rs
use crate::core::*;
use crate::generators::rng::SeedRng;

/// 你的生成器（可以是任意结构体）
pub struct MyGenerator {
    my_param: f64,
}

impl Generator for MyGenerator {
    /// 返回生成器的唯一标识名称
    fn name(&self) -> &'static str {
        "my_generator"
    }

    /// 核心逻辑：根据种子和参数生成帧迭代器
    fn generate_stream(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let seq_limit = params.seq_length;
        let my_param = self.my_param;
        let mut rng = SeedRng::new(seed);
        let mut step: u64 = 0;

        // 使用 from_fn 构造惰性迭代器
        let iter = std::iter::from_fn(move || {
            if seq_limit > 0 && step >= seq_limit as u64 {
                return None;
            }

            // 生成一帧数据
            let value = FrameState::Float(my_param * rng.next_f64());
            let frame = SequenceFrame::new(step, FrameData {
                values: vec![value],
            });

            step += 1;
            Some(frame)
        });

        // 应用序列长度限制
        Ok(if seq_limit > 0 {
            Box::new(iter.take(seq_limit))
        } else {
            Box::new(iter)
        })
    }
}

// 也可实现 generate_batch 以优化批量生成（可选）
// fn generate_batch(&self, seed: u64, params: &GenParams, batch_size: usize)
//     -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>>
```

### 步骤 2：定义参数结构体并编写工厂函数

```rust
use serde::Deserialize;
use std::collections::HashMap;
use serde_json::Value;

/// 你的生成器专用参数
#[derive(Debug, Clone, Deserialize)]
struct MyGeneratorParams {
    #[serde(default = "default_my_param")]
    my_param: f64,
}

fn default_my_param() -> f64 { 0.5 }

/// 工厂函数：从 extensions 映射构造生成器实例
pub fn my_generator_factory(
    extensions: &HashMap<String, Value>
) -> CoreResult<Box<dyn Generator>> {
    let params: MyGeneratorParams = if extensions.is_empty() {
        MyGeneratorParams { my_param: default_my_param() }
    } else {
        // 将 HashMap 序列化为 JSON，再反序列化为参数结构体
        let obj = serde_json::to_value(extensions).map_err(|e| {
            CoreError::SerializationError(format!("序列化失败: {}", e))
        })?;
        serde_json::from_value(obj).map_err(|e| {
            CoreError::SerializationError(format!("参数反序列化失败: {}", e))
        })?
    };

    // 校验参数
    if params.my_param <= 0.0 {
        return Err(CoreError::InvalidParams("my_param 必须大于 0".into()));
    }

    Ok(Box::new(MyGenerator { my_param: params.my_param }))
}
```

### 步骤 3：在注册表中注册

编辑 `src/generators/mod.rs`，在 `register_all()` 中添加：

```rust
// 添加 mod 声明（如果是新文件）
pub mod my_generator;

pub fn register_all(registry: &mut GeneratorRegistry) -> Result<(), CoreError> {
    // ... 已有注册 ...
    registry.register("my_generator", my_generator::my_generator_factory)?;       // 短名
    registry.register("my_generator_long_name", my_generator::my_generator_factory)?; // 长名别名
    Ok(())
}
```

### 步骤 4：在 YAML 中使用

```yaml
tasks:
  - name: "my_new_task"
    generator: "my_generator"
    params:
      seq_length: 500
      extensions:
        my_param: 0.8      # 传递到工厂函数的 extionsions
    count: 100
    seed: 42
```

### 设计要点总结

| 组件 | 职责 |
|------|------|
| `struct Generator` | 保存内部状态和无参配置 |
| `impl Generator` | 实现 `generate_stream()` 生成帧迭代器 |
| `Params struct` | 定义可反序列化的配置参数（含 `#[serde(default)]` 默认值） |
| `factory fn` | 从 `HashMap<String, Value>` 构造生成器，做参数校验 |
| `register_all()` | 将工厂函数注册到 `GeneratorRegistry`（一行代码） |

> **关键约束**：工厂函数签名必须为 `fn(&HashMap<String, Value>) -> CoreResult<Box<dyn Generator>>`。
> 使用 `SeedRng::new(seed)` 保证确定性——相同种子必须产生相同输出。

---

## 项目架构

```
CLI (main.rs)
  |
  ├─ 解析参数 (clap)
  ├─ 加载 YAML 清单 (scheduler/manifest.rs)
  ├─ 校验清单
  └─ 调度执行 (scheduler/run_manifest)
       |
       ├─ 分片任务 (scheduler/shard.rs)
       ├─ 并行执行 (rayon)
       └─ 单分片流水线 (scheduler/executor.rs)
            |
            ├─ 生成器 (generators/) → Generator trait → GeneratorRegistry
            ├─ 处理器 (pipeline/)    → Processor trait → ProcessorRegistry
            └─ 输出   (sink/)        → SinkAdapter trait → 3 种适配器
                                   ├─ ParquetAdapter
                                   ├─ TextAdapter
                                   └─ BinaryAdapter
```

```
src/
├── main.rs              # CLI 入口
├── core/                # 核心抽象
│   ├── error.rs         # 统一错误类型
│   ├── frame.rs         # 帧数据模型
│   ├── generator.rs     # Generator trait 定义
│   ├── params.rs        # 参数和配置类型
│   └── registry.rs      # 生成器注册表
├── generators/          # 12 种内置生成器
├── pipeline/            # 6 种后处理器
├── sink/                # 3 种输出格式适配器
├── scheduler/           # 任务调度层
│   ├── manifest.rs      # YAML 清单解析
│   ├── shard.rs         # 分片算法
│   ├── executor.rs      # 分片执行器
│   └── seed.rs          # SHA2 种子派生
└── metadata/            # 元数据与日志
    ├── types.rs         # 元数据结构
    ├── progress.rs      # 进度追踪
    ├── recorder.rs      # metadata.json 写入
    └── logger.rs        # tracing 日志初始化
```

## 运行测试

```bash
# 单元测试 + 集成测试
cargo test

# Release 模式测试
cargo test --release

# 运行 CLI 测试（需要先编译）
cargo build --release
./target/release/structgen-rs --manifest tests/manifests/minimal_ca.yaml --no-progress
```

## 许可证

MIT License 或 Apache License 2.0（任选其一）
