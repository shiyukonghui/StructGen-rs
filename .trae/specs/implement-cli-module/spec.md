# CLI模块与端到端集成 Spec

## Why
StructGen-rs 的核心领域模块（core/sink/pipeline/generators/scheduler/metadata）已全部实现并通过测试。用户需要一个命令行入口来编排这些模块，完成从 YAML 清单到数据产出 + metadata.json 的全流程。这是 M5 里程碑——端到端交付的最后一块拼图。

## What Changes
- 在 `Cargo.toml` 中添加 `clap` 依赖和 `[[bin]]` 目标配置
- 重写 `src/main.rs`：实现 `CliArgs` 参数定义、`main()` 入口和 `run()` 装配函数
- 提供真实的 SinkAdapter 工厂函数，将 OutputFormat 映射到 ParquetAdapter/TextAdapter/BinaryAdapter
- 实现进度条显示（stderr 覆盖式刷新）
- 实现运行结束后的汇总报告打印
- 实现退出码逻辑（0=全成功, 1=部分失败, 2=全失败）
- 编写 CLI 单元测试（参数解析）和端到端集成测试

## Impact
- Affected specs: 无（新增模块）
- Affected code: `Cargo.toml`, `src/main.rs`

## ADDED Requirements

### Requirement: CLI参数解析
系统 SHALL 通过 clap derive 方式定义 `CliArgs` 结构体，解析以下命令行参数：

| 短参数 | 长参数 | 类型 | 默认值 | 说明 |
|--------|--------|------|--------|------|
| `-m` | `--manifest` | PathBuf | 必需 | 任务清单 YAML 文件路径 |
| `-o` | `--output-dir` | PathBuf | 无 | 输出根目录（覆盖清单配置） |
| `-f` | `--format` | CliFormat 枚举 | 无 | 输出格式（覆盖清单配置） |
| `-t` | `--num-threads` | usize | 无 | 并行线程数（覆盖清单配置） |
| `-l` | `--log-level` | String | "info" | 日志级别 |
| | `--stream-write` | bool | 无 | 强制流式写出（覆盖清单配置） |
| | `--no-progress` | flag | false | 禁用进度条 |

#### Scenario: 最简参数解析成功
- **WHEN** 用户传入 `--manifest tasks.yaml`
- **THEN** `CliArgs::parse()` 成功解析，`manifest` 为 `PathBuf::from("tasks.yaml")`，`log_level` 为 `"info"`，其余可选参数均为 None/默认值

#### Scenario: 缺少 --manifest 报错
- **WHEN** 用户未传入 `--manifest`
- **THEN** clap 自动输出错误信息和帮助，程序退出

#### Scenario: 完整参数解析
- **WHEN** 用户传入所有可选参数
- **THEN** 所有字段正确填充

### Requirement: 主装配函数 run()
系统 SHALL 实现 `run(args: CliArgs) -> i32` 函数，按以下流程编排全模块：

1. 初始化日志系统（metadata::init_logger）
2. 验证清单文件存在且可读
3. 加载并解析 YAML 清单（scheduler::Manifest::from_yaml）
4. 应用命令行参数覆盖到清单全局配置
5. 设置 rayon 线程池大小
6. 初始化生成器注册表并注册所有内置生成器（generators::register_all）
7. 初始化处理器注册表并注册所有内置处理器（pipeline::register_all）
8. 校验清单合法性（manifest.validate）
9. 创建输出目录
10. 创建进度追踪器（metadata::ProgressTracker）
11. 启动进度显示线程（若未禁用）
12. 调用 scheduler::run_manifest 执行任务
13. 停止进度显示线程
14. 写入元数据（metadata::write_metadata）
15. 打印汇总报告
16. 返回退出码

#### Scenario: 正常执行全流程
- **WHEN** 提供有效的 YAML 清单
- **THEN** 输出数据文件被写入到输出目录，metadata.json 被生成，进度条正常显示，汇总报告打印到 stdout

#### Scenario: 清单文件不存在
- **WHEN** --manifest 指向不存在的文件
- **THEN** 打印 `Error: manifest file not found: <path>` 到 stderr，返回退出码 2

#### Scenario: YAML解析失败
- **WHEN** 清单文件内容不是合法 YAML
- **THEN** 打印可读错误信息到 stderr，返回退出码 2

#### Scenario: 输出目录无法创建
- **WHEN** 输出目录路径指向无权限位置
- **THEN** 打印错误信息到 stderr，返回退出码 2

### Requirement: 真实适配器工厂
系统 SHALL 提供 `create_adapter(format: OutputFormat) -> CoreResult<Box<dyn SinkAdapter>>` 函数：
- Parquet → `ParquetAdapter::new()`
- Text → `TextAdapter::new()`
- Binary → `BinaryAdapter::new()`

### Requirement: 进度显示
系统 SHALL 在 --no-progress 未设置时，启动独立线程定期从 ProgressTracker 读取进度快照，以 `\r` 覆盖方式输出到 stderr：
```
[=====>    ] 52% (520/1000 samples) | 5200 frames | ETA: 00:03:45
```
若 stderr 不是 TTY（重定向场景），降级为每 5 秒打印一行日志。

#### Scenario: 进度条正常显示
- **WHEN** 运行时未设置 --no-progress 且 stderr 是 TTY
- **THEN** 每 200ms 刷新一次进度条到 stderr

#### Scenario: 禁用进度条
- **WHEN** 设置 --no-progress
- **THEN** 不启动进度显示线程，不输出进度信息到 stderr

### Requirement: 汇总报告
系统 SHALL 在运行结束后打印格式化汇总报告到 stdout，包含：
- StructGen-rs 版本号
- 输出目录路径
- 总耗时（格式 HH:MM:SS.mmm）
- 每个任务的名称、样本数、帧数、数据量和状态
- 总计行：总样本数、总帧数、总数据量、总文件数
- metadata.json 路径
- 退出码说明

#### Scenario: 全部成功时的汇总报告
- **WHEN** 所有分片执行成功
- **THEN** 每个任务显示 [OK]，退出码为 0

#### Scenario: 部分失败时的汇总报告
- **WHEN** 部分分片失败
- **THEN** 相应任务显示失败分片数，退出码为 1

### Requirement: 退出码
系统 SHALL 返回以下退出码：
- 0: 所有分片成功完成
- 1: 部分分片失败（至少有一个成功）
- 2: 全部失败或启动阶段致命错误（清单解析失败、生成器未注册等）

### Requirement: Panic Hook
系统 SHALL 设置自定义 panic hook，将 panic 信息以结构化日志格式输出到 log::error! 和 stderr。

#### Scenario: 运行时 panic 被优雅捕获
- **WHEN** 代码发生 panic
- **THEN** panic 信息通过 log::error! 记录，同时打印到 stderr，提示用户报告此 bug

### Requirement: Cargo.toml 配置
Cargo.toml SHALL 添加：
- `clap = { version = "4", features = ["derive"] }` 依赖
- `[[bin]]` 节，name = "structgen-rs"，path = "src/main.rs"

### Requirement: 端到端集成测试
系统 SHALL 提供端到端集成测试：
- `--help` 输出包含所有参数说明
- 缺少 `--manifest` 时报错退出
- 最小 YAML 清单 → 产出数据文件 + metadata.json
- 退出码验证：全成功=0，部分失败=1，全失败=2
