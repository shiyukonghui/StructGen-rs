# CA 批量数据生成 — 19 种规则各 120 万条

- [x] Task 1: 创建 ca_all_rules_120w.yaml 清单文件
    - 1.1: 定义全局配置（output_dir=Npy格式, 8线程, 流式写出, shard_max=10000）
    - 1.2: 定义 5 个 1D CA 任务（rule30/54/90/110/184, seq_length=128, width=128）
    - 1.3: 定义 10 个 2D CA 任务（game_of_life/highlife/day_night/seeds/diamoeba/replicator_2d/maze/wireworld/cyclic/ameyalli, seq_length=64, rows=32, cols=32）
    - 1.4: 定义 3 个 3D CA 任务（life_3d/cyclic_3d/fredkin_3d, seq_length=32, depth=8, rows=8, cols=8）
    - 1.5: 定义 1 个 NCA2D 任务（nca2d, seq_length=64, rows=12, cols=12）
    - 1.6: 验证 YAML 语法正确性（cargo test 或项目校验）

- [x] Task 2: 运行生成并验证
    - 2.1: 编译项目（cargo build --release）
    - 2.2: 执行 `structgen-rs run --manifest ca_all_rules_120w.yaml` 生成数据
    - 2.3: 检查输出目录结构和 metadata.json 确认 19 种规则数据完整
