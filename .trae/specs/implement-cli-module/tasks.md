# Tasks: CLI模块与端到端集成

- [x] Task 1: 更新 Cargo.toml 依赖和配置
  - [x] 添加 `clap = { version = "4", features = ["derive"] }` 到 `[dependencies]`
  - [x] 添加 `[[bin]]` 二进制目标配置（name = "structgen-rs", path = "src/main.rs"）

- [x] Task 2: 实现 CLI 参数解析和适配器工厂
  - [x] 定义 `CliFormat` 枚举（实现 `clap::ValueEnum` + `From<CliFormat> for OutputFormat`）
  - [x] 定义 `CliArgs` 结构体（clap derive），含 `--manifest`、`--output-dir`、`--format`、`--num-threads`、`--log-level`、`--stream-write`、`--no-progress` 参数
  - [x] 定义 `exit_code` 常量模块（SUCCESS=0, PARTIAL_FAILURE=1, FULL_FAILURE=2）
  - [x] 实现 `create_adapter()` 工厂函数（映射 OutputFormat → ParquetAdapter/TextAdapter/BinaryAdapter）
  - [x] 编写 CliArgs 解析的单元测试（最简参数、完整参数）

- [x] Task 3: 实现 main() 入口和 panic hook
  - [x] 设置自定义 panic hook（log::error + eprintln）
  - [x] 实现 `main()` 函数：解析参数 → 调用 run() → process::exit()

- [x] Task 4: 实现 run() 装配函数
  - [x] 步骤1：初始化日志（metadata::init_logger，使用 Once 幂等）
  - [x] 步骤2：验证清单文件存在（不存在则打印错误并返回2）
  - [x] 步骤3：加载并解析 YAML 清单（Manifest::from_yaml，失败则返回2）
  - [x] 步骤4：应用命令行参数覆盖到 manifest.global
  - [x] 步骤5：设置 rayon 全局线程池（使用 Once 幂等）
  - [x] 步骤6：初始化 GeneratorRegistry 并注册所有内置生成器
  - [x] 步骤7：初始化 ProcessorRegistry 并注册所有内置处理器
  - [x] 步骤8：调用 manifest.validate() 校验清单合法性（失败返回2）
  - [x] 步骤9：创建输出目录（失败返回2）
  - [x] 步骤10：创建 ProgressTracker
  - [x] 步骤11：启动进度显示线程（若未禁用）
  - [x] 步骤12：记录开始时间，调用 scheduler::run_manifest()
  - [x] 步骤13：停止进度显示线程
  - [x] 步骤14：写入 metadata.json（失败打印警告，不中断）
  - [x] 步骤15：打印汇总报告
  - [x] 步骤16：返回退出码

- [x] Task 5: 实现进度条显示函数
  - [x] 实现 `progress_display_loop()` 函数：循环读取进度，`\r` 覆盖输出到 stderr
  - [x] 实现 TTY 检测：若非 TTY 环境则降级为每 5 秒日志输出
  - [x] 格式化进度条：`[====>    ] 52% (520/1000 samples) | 5200 frames | ETA: 00:03:45`

- [x] Task 6: 实现汇总报告打印函数
  - [x] 实现 `print_summary()` 函数：格式化输出任务统计、总计行、metadata.json 路径、退出码
  - [x] 格式化耗时：HH:MM:SS.mmm
  - [x] 格式化字节数：自动选择 B/KB/MB/GB 单位
  - [x] 每个任务行显示：名称、样本数、帧数、数据量、状态（[OK] 或 [N shards FAILED]）

- [x] Task 7: 编写集成测试
  - [x] 创建测试用 demo YAML 清单文件内容（行内字符串）
  - [x] 端到端测试：从 YAML 字符串 → 写入临时文件 → 模拟 CLI 调用 run() → 验证产出文件和 metadata.json
  - [x] 测试退出码：全成功=0
  - [x] 测试退出码：部分失败=1（使用 MockAdapter 模拟失败）
  - [x] 测试退出码：清单不存在=2
  - [x] 测试退出码：YAML 解析失败=2
  - [x] 测试 --no-progress 参数禁用进度条
  - [x] 测试命令行参数覆盖清单配置

- [x] Task 8: 最终全功能验证
  - [x] 运行 `cargo test` 确保全部测试通过（286 passed, 0 failed）
  - [x] 运行 `cargo clippy` 确保无实质性警告（0 new warnings）
  - [x] 运行 `cargo build` 确保编译成功

# Task Dependencies
- Task 2 依赖 Task 1（需要 clap 依赖就位）
- Task 3 依赖 Task 2
- Task 4 依赖 Task 2、Task 3
- Task 5 依赖 Task 4（需要 run() 中的 ProgressTracker）
- Task 6 依赖 Task 4（需要 ShardResult 数据结构）
- Task 7 依赖 Task 4、Task 5、Task 6（端到端需要完整流程）
- Task 8 依赖 Task 1-7 全部完成
