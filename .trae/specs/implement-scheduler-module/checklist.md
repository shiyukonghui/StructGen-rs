# Checklist: 第三阶段调度协调模块验证

## 编译与代码质量
- [x] `cargo check` 无错误
- [x] `cargo build` 编译通过
- [x] `cargo clippy` 无实质性警告（仅 dead_code，符合自底向上开发预期）
- [x] `cargo test` 全部通过（239 passed, 0 failed），覆盖率 ≥ 80%

## 依赖与模块声明
- [x] `Cargo.toml` 包含 `rayon = "1"` 和 `serde_yaml = "0.9"` 依赖
- [x] `src/main.rs` 声明 `mod scheduler;`
- [x] `src/sink/mod.rs` 重导出 `OutputStats`

## 种子派生 (seed.rs)
- [x] `derive_seed()` 函数签名正确：`fn(u64, usize) -> u64`
- [x] 确定性测试通过：相同输入→相同输出
- [x] wrapping 行为测试通过：u64::MAX + 1 = 0

## 清单数据结构 (manifest.rs)
- [x] `Manifest` 结构体包含 `tasks: Vec<TaskSpec>` 和 `global: GlobalConfig`
- [x] `TaskSpec` 包含 name/generator/params/count/seed/pipeline/output_format/shard_size
- [x] YAML 反序列化正确：合法 YAML 解析成功
- [x] 零样本数拒绝测试通过
- [x] 空输出目录拒绝测试通过
- [x] 未注册生成器拒绝测试通过
- [x] 未注册处理器拒绝测试通过

## 分片数据结构 (shard.rs)
- [x] `Shard` 结构体包含 task_idx/shard_idx/seed/sample_count
- [x] `ShardResult` 结构体包含 task_name/shard_idx/seed/sample_count/format/stats/error
- [x] `shard_tasks()` 均匀分片测试通过
- [x] `shard_tasks()` 不均匀分片测试通过
- [x] 自动分片大小计算正确（target_shards = num_cpus * 4）

## 分片执行器 (executor.rs)
- [x] `execute_shard()` 实现完整数据流：生成→管道→写出
- [x] mock 生成器端到端测试通过（frames_written 正确）
- [x] panic 容错测试通过：错误记录在 ShardResult.error 中
- [x] 多处理器链式应用正确

## 模块根与并行调度 (mod.rs)
- [x] `run_manifest()` 接受 Manifest/GeneratorRegistry/ProcessorRegistry/SinkAdapterFactory
- [x] rayon 并行执行所有分片
- [x] 相同清单两次运行结果确定性一致
- [x] 部分分片失败不影响其他分片（容错隔离）
- [x] 返回结果包含成功和失败的分片结果
