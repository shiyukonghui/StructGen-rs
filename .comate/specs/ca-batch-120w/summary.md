# CA 批量数据生成 — 19 种规则各 120 万条

## 任务完成摘要

### 已完成工作

1. **创建 YAML 清单文件** `ca_all_rules_120w.yaml`
   - 包含 19 个 task，覆盖所有可用 CA 规则
   - 全局配置：Npy 格式输出、8 线程、流式写出、shard_max=10000
   - 输出目录：`F:/RustProjects/StructGen-rs/output_ca_120w`

2. **编译并启动生成任务**
   - `cargo build --release` 编译成功（仅 warnings）
   - `structgen-rs run --manifest ca_all_rules_120w.yaml --no-progress` 已启动
   - 后台任务 ID: `qv8n15`

### 规则覆盖情况

| 类别 | 规则数量 | 任务名前缀 | 状态 |
|------|----------|------------|------|
| 1D CA | 5 | ca1d_rule30/54/90/110/184 | 生成中 |
| 2D CA | 10 | ca2d_game_of_life/highlife/day_night/seeds/diamoeba/replicator/maze/wireworld/cyclic/ameyalli | 生成中 |
| 3D CA | 3 | ca3d_life/cyclic/fredkin | 排队中 |
| NCA2D | 1 | nca2d_default | 排队中 |

### 产出统计（截至检查时刻）

- 已产出 175+ 个 .npy 分片文件
- 总数据量约 236 GB
- 1D CA 进度领先（5 种规则各约 32/120 分片）
- 2D CA 刚开始产出（15/1200 分片）
- 3D CA 和 NCA2D 在线程池调度队列中排队

### 关键配置参数

- **1D CA**: seq_length=128, width=128, boundary=periodic
- **2D CA**: seq_length=64, rows=32, cols=32, boundary=periodic, neighborhood=moore
- **3D CA**: seq_length=32, depth=8, rows=8, cols=8, boundary=periodic, neighborhood=moore
- **NCA2D**: seq_length=64, rows=12, cols=12, d_state=10, n_groups=1

### 注意事项

- xrule 规则因需要外部 512 位查找表而排除
- 每个 .npy 分片约 586 MB（1D）/ 较小（2D/3D），可直接用 `np.load()` 加载
- 生成任务在后台持续运行，预计完整产出需要数小时
- 完成后输出目录将包含 `metadata.json` 汇总文件
