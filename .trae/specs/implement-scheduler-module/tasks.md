# Tasks: 实现第三阶段调度协调模块 (scheduler)

## 依赖准备

- [x] Task 0: 更新 Cargo.toml 和模块声明
  - [x] 在 Cargo.toml 中添加 `rayon = "1"` 和 `serde_yaml = "0.9"` 依赖
  - [x] 运行 `cargo check` 确认新依赖拉取成功
  - [x] 在 `src/sink/mod.rs` 中重导出 `OutputStats`
  - [x] 在 `src/main.rs` 中添加 `mod scheduler;`

## Scheduler 模块核心实现

- [x] Task 1: 实现种子派生 (seed.rs)
  - [x] 创建 `src/scheduler/seed.rs`：实现 `derive_seed(base_seed, shard_idx)` 函数
  - [x] 使用 `wrapping_add` 确保确定性派生
  - [x] 编写单元测试：确定性验证、wrapping 行为验证
  - [x] 运行 `cargo test scheduler` 确认测试通过

- [x] Task 2: 实现清单数据结构 (manifest.rs)
  - [x] 创建 `src/scheduler/manifest.rs`：定义 `Manifest`、`TaskSpec` 结构体
  - [x] 实现 Manifest 的 YAML 反序列化（serde_yaml）
  - [x] 实现 Manifest 校验逻辑：output_dir 非空、task 名称唯一、count > 0、generator 在注册表中存在、pipeline 名称在注册表中存在
  - [x] 编写单元测试：合法 YAML 解析、零样本数拒绝、空输出目录拒绝、未注册生成器拒绝
  - [x] 运行 `cargo test scheduler` 确认测试通过

- [x] Task 3: 实现分片数据结构与切分算法 (shard.rs)
  - [x] 创建 `src/scheduler/shard.rs`：定义 `Shard`、`ShardResult` 结构体
  - [x] 实现 `shard_tasks()` 分片切分算法（均匀/不均匀分布、自动分片大小计算）
  - [x] 实现自动分片大小计算（target_shards = num_cpus * 4）
  - [x] 编写单元测试：均匀分片、不均匀分片、自动分片大小
  - [x] 运行 `cargo test scheduler` 确认测试通过

- [x] Task 4: 实现分片执行器 (executor.rs)
  - [x] 创建 `src/scheduler/executor.rs`：实现 `execute_shard()` 函数
  - [x] 实现完整数据流：生成器实例化 → 后处理管道组装 → 适配器创建 → 逐帧写入 → 收集统计
  - [x] 实现容错捕获（panic 和错误转为 ShardResult.error）
  - [x] 编写集成测试：mock 生成器端到端数据流
  - [x] 编写容错测试：注入 panic 生成器验证隔离
  - [x] 运行 `cargo test scheduler` 确认测试通过

- [x] Task 5: 实现模块根与并行调度 (mod.rs)
  - [x] 创建 `src/scheduler/mod.rs`：声明子模块，实现 `run_manifest()` 主入口
  - [x] 使用 rayon par_iter 并行执行所有分片
  - [x] 实现容错汇总：收集所有 ShardResult，不因个别失败而中止
  - [x] 编写集成测试：多分片并行执行、确定性验证（相同清单两次运行结果一致）
  - [x] 编写容错测试：部分分片失败不影响其他分片
  - [x] 运行全量 `cargo test scheduler` 确认全部测试通过

## 最终集成验证

- [x] Task Z1: 运行全量测试和代码质量检查
  - [x] `cargo test` 全部通过（239 passed, 0 failed），覆盖率 ≥ 80%
  - [x] `cargo clippy` 无实质性警告（仅 dead_code，符合自底向上开发预期）
  - [x] `cargo build` 编译通过

# Task Dependencies

- Task 0 (Cargo.toml + 模块声明) 必须先完成
- Task 1 (seed.rs) 无依赖，可立即开始
- Task 2 (manifest.rs) 依赖 Task 0（需要 serde_yaml）
- Task 3 (shard.rs) 依赖 Task 1（需要 derive_seed）和 Task 2（需要 TaskSpec）
- Task 4 (executor.rs) 依赖 Task 1/2/3（需要全部数据结构）和 Task 0（需要 OutputStats 重导出）
- Task 5 (mod.rs) 依赖 Task 1/2/3/4 全部完成
- Task Z1 依赖所有任务完成
