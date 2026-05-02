# StructGen-rs 命令行接口 (CLI) 详细设计规格

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| 1.0 | 2026-05-01 | - | 初始版本，定义命令行参数、装配流程和入口逻辑 |

## 1. 模块概述

命令行接口（CLI）是 StructGen-rs 系统与用户交互的唯一入口。它是一个独立的二进制目标（`src/main.rs`），负责解析命令行参数、加载清单文件、初始化各子系统（日志、生成器注册表、处理器注册表），并装配完整的生成流水线。

该模块的职责包括：
- 解析命令行参数（清单路径、输出目录、输出格式、线程数、日志级别等）。
- 验证参数的有效性（清单文件存在、输出目录可写、格式合法等）。
- 按需加载 YAML 清单文件并反序列化为 `Manifest`。
- 初始化全局生成器注册表和处理器注册表（注册所有内置实现）。
- 装配并启动调度器，监控进度，等待完成。
- 运行结束后调用元数据层生成 `metadata.json`，并打印汇总报告。
- 设置进程退出码（0 = 成功，1 = 部分失败，2 = 完全失败）。

**核心原则**：CLI 层是系统中唯一直接与用户交互的模块。它不应该包含任何业务逻辑——所有领域逻辑由 scheduler、generators、pipeline、sink、metadata 等模块承载。

## 2. 设计目标与原则

- **简洁**：命令行参数最少化，所有复杂配置通过 YAML 清单承载。
- **明确的错误信息**：参数解析失败或清单校验失败时，输出人类可读的错误提示，包含修复建议。
- **进度可视化**：长时间运行时通过标准错误输出实时进度条；短任务则直接输出结果不刷屏。
- **向前兼容**：新增参数时提供合理的默认值，不破坏已有调用脚本。
- **单一入口**：`main()` 函数作为唯一入口，不提供子命令（所有功能通过清单中的任务类型区分）。

## 3. 模块内部结构组织

```
src/
├── main.rs       # 二进制入口，main() 函数

// 可选：若 CLI 逻辑较复杂，可进一步拆分
├── cli/
│   ├── mod.rs    # 模块根
│   ├── args.rs   # CliArgs 定义与解析（基于 clap）
│   └── run.rs    # run() 函数：装配并启动全流程
```

若 CLI 逻辑足够简单，全部放在 `main.rs` 中即可；以上拆分适用于参数较多、逻辑复杂的场景。本文档按简单方案设计——所有 CLI 逻辑集中在 `src/main.rs`。

## 4. 公开接口定义

CLI 模块本身不暴露公开 Rust API（它是二进制入口，不被其他模块调用），但其内部函数划分如下：

```rust
use clap::Parser;
use std::path::PathBuf;
use std::process;

/// 命令行参数定义。
#[derive(Parser, Debug)]
#[command(name = "StructGen-rs")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "高性能程序化结构化数据生成器", long_about = None)]
pub struct CliArgs {
    /// 任务清单 YAML 文件路径（必需参数）。
    #[arg(short, long, value_name = "FILE")]
    pub manifest: PathBuf,

    /// 输出根目录（覆盖清单中的 global.output_dir）。
    #[arg(short = 'o', long, value_name = "DIR")]
    pub output_dir: Option<PathBuf>,

    /// 输出格式: parquet | text | binary（覆盖清单中的格式设置）。
    #[arg(short, long, value_enum)]
    pub format: Option<OutputFormat>,

    /// 并行线程数（默认为 CPU 逻辑核心数）。
    #[arg(short = 't', long, value_name = "N")]
    pub num_threads: Option<usize>,

    /// 日志级别: trace | debug | info | warn | error。
    #[arg(short = 'l', long, value_name = "LEVEL", default_value = "info")]
    pub log_level: String,

    /// 是否强制使用流式写出模式（覆盖清单设置）。
    #[arg(long)]
    pub stream_write: Option<bool>,

    /// 是否禁用进度条（适用于 CI/重定向场景）。
    #[arg(long)]
    pub no_progress: bool,
}
```

### 4.1 退出码

```rust
/// 进程退出码常量。
pub mod exit_code {
    pub const SUCCESS: i32 = 0;              // 所有任务成功完成
    pub const PARTIAL_FAILURE: i32 = 1;      // 部分分片失败，但部分成功
    pub const FULL_FAILURE: i32 = 2;         // 所有任务或清单解析失败
}
```

### 4.2 主入口函数签名（逻辑概要）

```rust
fn main() {
    let args = CliArgs::parse();
    let exit_code = run(args);
    process::exit(exit_code);
}

fn run(args: CliArgs) -> i32 {
    // 1. 初始化日志
    // 2. 验证并加载清单
    // 3. 应用命令行覆盖到全局配置
    // 4. 初始化生成器注册表和处理器注册表
    // 5. 创建进度追踪器
    // 6. 调用 scheduler::run_manifest()
    // 7. 写入元数据
    // 8. 打印汇总报告
    // 9. 返回退出码
}
```

## 5. 核心逻辑详解

### 5.1 完整执行流程

```
main()
  ↓
1. 解析命令行参数: CliArgs::parse()
  ↓
2. 初始化日志: metadata::init_logger(&args.log_level, log_file_path)
   - 若 --no-progress，日志级别至少为 WARN
   - 同时在输出目录创建 structgen_run.log
  ↓
3. 验证清单文件存在且可读:
   if !args.manifest.exists() → 打印错误，退出码 2
  ↓
4. 加载并解析清单:
   let manifest: Manifest = serde_yaml::from_str(&fs::read_to_string(&args.manifest)?)?;
   - 解析失败 → 打印详细错误，退出码 2
  ↓
5. 应用命令行覆盖（命令行参数优先级 > 清单设置）:
   if let Some(dir) = args.output_dir { manifest.global.output_dir = dir; }
   if let Some(fmt) = args.format { manifest.global.default_format = fmt; }
   if let Some(n) = args.num_threads { manifest.global.num_threads = Some(n); }
   if let Some(sw) = args.stream_write { manifest.global.stream_write = sw; }
  ↓
6. 设置 rayon 线程池大小:
   rayon::ThreadPoolBuilder::new()
       .num_threads(num_threads)
       .build_global()?;
  ↓
7. 初始化生成器注册表:
   let mut gen_registry = GeneratorRegistry::new();
   generators::register_all(&mut gen_registry);  // 注册所有内置生成器
  ↓
8. 初始化处理器注册表:
   let mut proc_registry = ProcessorRegistry::new();
   pipeline::register_all(&mut proc_registry);  // 注册所有内置处理器
  ↓
9. 校验清单中引用的所有生成器和处理器名称均已注册:
   for task in &manifest.tasks {
       if !gen_registry.contains(&task.generator) {
           eprintln!("Error: generator '{}' not found.", task.generator);
           eprintln!("Available: {:?}", gen_registry.list_names());
           return exit_code::FULL_FAILURE;
       }
       for proc_name in &task.pipeline {
           if !proc_registry.contains(proc_name) {
               eprintln!("Error: processor '{}' not found.", proc_name);
               return exit_code::FULL_FAILURE;
           }
       }
   }
  ↓
10. 创建输出目录:
    fs::create_dir_all(&manifest.global.output_dir)?;
  ↓
11. 计算总样本数（用于进度追踪）:
    let total_samples: usize = manifest.tasks.iter().map(|t| t.count).sum();
    let progress = ProgressTracker::new(total_samples);
  ↓
12. 启动进度显示线程（若非 --no-progress）:
    let progress_handle = if !args.no_progress {
        Some(std::thread::spawn(move || { progress_display_loop(progress.clone()); }))
    } else { None };
  ↓
13. 记录开始时间，调用调度器:
    let start = Utc::now();
    let results = scheduler::run_manifest(&manifest, &gen_registry)?;
    let end = Utc::now();
  ↓
14. 停止进度显示线程:
    if let Some(handle) = progress_handle {
        progress.signal_done();
        handle.join().unwrap();
    }
  ↓
15. 写入元数据:
    metadata::write_metadata(
        Path::new(&manifest.global.output_dir),
        &manifest, &results, start, end
    )?;
  ↓
16. 打印汇总报告:
    print_summary(&results, end - start);
  ↓
17. 确定退出码:
    if results.iter().all(|r| r.error.is_none()) {
        exit_code::SUCCESS
    } else if results.iter().any(|r| r.error.is_none()) {
        exit_code::PARTIAL_FAILURE  // 部分成功
    } else {
        exit_code::FULL_FAILURE  // 全部失败
    }
```

### 5.2 覆盖策略

命令行参数与清单配置的覆盖优先级（高到低）：

```
命令行参数 > 清单 global 字段 > 清单任务级字段 > 硬编码默认值
```

具体覆盖规则：

| 命令行参数 | 覆盖的清单字段 |
|-----------|---------------|
| `--output-dir` | `manifest.global.output_dir` |
| `--format` | `manifest.global.default_format`（未在任务级指定的任务也会使用此值） |
| `--num-threads` | `manifest.global.num_threads` |
| `--log-level` | `manifest.global.log_level` |
| `--stream-write` | `manifest.global.stream_write` |

### 5.3 进度显示

```
progress_display_loop(tracker: ProgressTracker):
    loop:
        let info = tracker.progress();
        输出到 stderr（一行，覆盖式更新）:
        "\r[=====>    ] 52% (520000/1000000 samples) | 5.2M frames | ETA: 00:03:45"
        sleep 200ms
        if tracker.is_done(): break
```

- 进度条通过 `\r` 回车符实现同行刷新。
- 若输出被重定向到文件（`!stderr.is_tty()`），降级为每 5 秒打印一行日志，避免产生大量进度行。

### 5.4 汇总报告格式

```
============================================================
 StructGen-rs v0.1.0 - Run Summary
============================================================
 Output directory: ./output
 Elapsed time:      00:05:23.456

 Tasks:
   rule30_ca             50,000 samples |  50,000,000 frames |  2.3 GB  [OK]
   lorenz_chaos         100,000 samples | 200,000,000 frames |  8.1 GB  [OK]
   nbody_sim             10,000 samples |          --        |    --    [2 shards FAILED]

 Totals:
   Samples:  150,000 (10,000 failed)
   Frames:   250,000,000
   Data:     10.4 GB
   Files:    35

 Metadata: ./output/metadata.json
 Exit code: 1 (partial failure — see metadata.json for details)
============================================================
```

## 6. 与其他模块的交互

### 6.1 依赖关系

```
CLI (main.rs)
  ├── clap                 -- 命令行参数解析
  ├── core                 -- CliArgs、OutputFormat、GeneratorRegistry、CoreError
  ├── generators           -- register_all(&mut GeneratorRegistry)
  ├── scheduler            -- run_manifest()
  ├── pipeline             -- register_all(&mut ProcessorRegistry)
  ├── metadata             -- init_logger()、ProgressTracker、write_metadata()
  └── rayon                -- 设置线程池
```

### 6.2 CLI 是唯一的编排者

```
CLI 不实现任何业务逻辑，它只做编排：

  CLI
   ├→ metadata::init_logger()        -- 设置可观测性
   ├→ generators::register_all()     -- 注册能力
   ├→ pipeline::register_all()       -- 注册能力
   ├→ scheduler::run_manifest()      -- 执行任务
   ├→ metadata::write_metadata()     -- 记录结果
   └→ 打印汇总                        -- 通知用户
```

## 7. 错误处理策略

### 7.1 错误分类与处理

| 错误 | 发生阶段 | 输出 | 退出码 |
|------|---------|------|--------|
| 缺少 --manifest 参数 | 参数解析 | clap 自动输出帮助信息 | 2 |
| 清单文件不存在 | 启动校验 | `Error: manifest file not found: <path>` | 2 |
| YAML 解析失败 | 清单加载 | `Error: YAML parse error at line <N>` | 2 |
| 输出目录无法创建 | 启动校验 | `Error: cannot create output directory: <reason>` | 2 |
| 生成器/处理器未注册 | 启动校验 | `Error: generator 'x' not found. Available: [...]` | 2 |
| 线程数设为 0 | 启动校验 | `Error: num_threads must be >= 1` | 2 |
| 运行时部分分片失败 | 执行中 | 日志记录详细错误，进度继续 | 1 |
| 运行时全部失败 | 执行中 | 日志记录错误，打印失败汇总 | 2 |
| metadata.json 写入失败 | 收尾 | 打印警告，但不影响已生成数据 | 1 |

### 7.2 Panic 时的不安全处理

```rust
fn main() {
    // 设置自定义 panic hook，让 panic 信息以结构化日志格式输出
    std::panic::set_hook(Box::new(|info| {
        log::error!("FATAL: {}", info);
        eprintln!("FATAL ERROR: {}\nPlease report this bug.", info);
    }));
    // ...
}
```

## 8. 性能考量

- **CLI 本身零开销**：全部工作委托给子模块，CLI 层只负责启动和收尾。
- **使用 jemalloc**：可考虑在 `Cargo.toml` 中启用 jemalloc 内存分配器以降低多线程分配开销。
- **rayon 线程池单次初始化**：在程序入口一次性设置，避免重复创建。
- **进度条避免频繁刷新**：200ms 周期保证了人眼感知度，同时几乎不消耗 CPU。

## 9. 可测试性设计

CLI 作为二进制入口，测试主要通过两种方式：

### 9.1 单元测试（args 解析）

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_parse_minimal_args() {
        let args = CliArgs::parse_from([
            "structgen-rs",
            "--manifest", "tasks.yaml",
        ]);
        assert_eq!(args.manifest, PathBuf::from("tasks.yaml"));
        assert_eq!(args.log_level, "info");
        assert!(args.output_dir.is_none());
    }

    #[test]
    fn test_parse_full_args() {
        let args = CliArgs::parse_from([
            "structgen-rs",
            "--manifest", "tasks.yaml",
            "--output-dir", "/tmp/out",
            "--format", "text",
            "--num-threads", "4",
            "--log-level", "debug",
            "--stream-write",
        ]);
        assert_eq!(args.output_dir, Some(PathBuf::from("/tmp/out")));
        assert!(args.stream_write.unwrap());
    }
}
```

### 9.2 集成测试（端到端）

```rust
#[cfg(test)]
mod integration_tests {
    use std::process::Command;

    #[test]
    fn test_help_flag() {
        let output = Command::new(env!("CARGO_BIN_EXE_structgen-rs"))
            .arg("--help")
            .output()
            .unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("--manifest"));
    }

    #[test]
    fn test_missing_manifest() {
        let output = Command::new(env!("CARGO_BIN_EXE_structgen-rs"))
            .output()
            .unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("--manifest"));
    }
}
```

## 10. 配置与参数

### 10.1 命令行参数完整列表

| 短参数 | 长参数 | 取值 | 默认 | 说明 |
|--------|--------|------|------|------|
| `-m` | `--manifest` | FILE | 无（必需） | 任务清单 YAML 文件路径 |
| `-o` | `--output-dir` | DIR | 来自清单 | 输出根目录 |
| `-f` | `--format` | parquet\|text\|binary | 来自清单 | 输出格式 |
| `-t` | `--num-threads` | N | CPU 核心数 | 并行线程数 |
| `-l` | `--log-level` | trace\|debug\|info\|warn\|error | info | 日志级别 |
| | `--stream-write` | bool | 来自清单 | 强制流式写出 |
| | `--no-progress` | flag | false | 禁用进度条 |

### 10.2 用法示例

```bash
# 最简调用
structgen-rs --manifest tasks.yaml

# 指定输出目录和格式
structgen-rs --manifest tasks.yaml --output-dir ./data --format parquet

# 限制线程数，启用详细日志
structgen-rs -m tasks.yaml -t 4 -l debug

# CI 环境（无进度条，输出到文本格式）
structgen-rs -m tasks.yaml -f text --no-progress

# 查看版本
structgen-rs --version
```

### 10.3 Cargo.toml 二进制目标配置

```toml
[[bin]]
name = "structgen-rs"
path = "src/main.rs"

[package.metadata.bin]
# 可在发布时嵌入构建信息
```

---

通过以上设计，命令行接口以简洁、鲁棒的单一入口方式将用户意图转换为可执行的生成流水线，确保从命令敲下到 `metadata.json` 写出的全流程清晰可控。
