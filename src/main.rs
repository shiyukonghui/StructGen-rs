//! StructGen-rs 命令行入口
//!
//! 本模块是系统的唯一用户交互入口，负责：
//! - 解析命令行参数（基于 clap）
//! - 初始化各子系统（日志、注册表、线程池）
//! - 加载并校验 YAML 清单
//! - 装配并启动调度器执行全流程
//! - 显示进度条、打印汇总报告、返回退出码
//!
//! CLI 层不包含任何业务逻辑——所有领域逻辑由下层模块承载。

mod core;
mod generators;
mod metadata;
mod pipeline;
mod scheduler;
mod sink;
mod view;

use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Once};

use chrono::Utc;
use clap::{Parser, Subcommand, ValueEnum};

use crate::core::{CoreResult, OutputFormat};
use crate::generators::register_all as register_all_generators;
use crate::metadata::{init_logger, write_metadata, ProgressTracker};
use crate::pipeline::register_all as register_all_processors;
use crate::sink::{BinaryAdapter, NpyAdapter, ParquetAdapter, SinkAdapter, TextAdapter};

// ============================================================================
// 退出码常量
// ============================================================================

/// 进程退出码常量
pub mod exit_code {
    /// 所有分片成功完成
    pub const SUCCESS: i32 = 0;
    /// 部分分片失败（至少有一个成功）
    pub const PARTIAL_FAILURE: i32 = 1;
    /// 全部失败或启动阶段致命错误
    pub const FULL_FAILURE: i32 = 2;
}

// ============================================================================
// 命令行参数定义
// ============================================================================

/// CLI 输出格式枚举，映射到 core::OutputFormat
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum CliFormat {
    /// Apache Parquet 列式存储
    Parquet,
    /// Unicode 文本格式
    Text,
    /// 紧凑二进制格式
    Binary,
    /// NumPy .npy 格式
    Npy,
}

impl From<CliFormat> for OutputFormat {
    fn from(f: CliFormat) -> Self {
        match f {
            CliFormat::Parquet => OutputFormat::Parquet,
            CliFormat::Text => OutputFormat::Text,
            CliFormat::Binary => OutputFormat::Binary,
            CliFormat::Npy => OutputFormat::Npy,
        }
    }
}

/// 命令行参数定义，基于 clap derive 宏
#[derive(Parser, Debug)]
#[command(name = "StructGen-rs")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "高性能程序化结构化数据生成器", long_about = None)]
pub struct CliArgs {
    /// 任务清单 YAML 文件路径（必需参数）
    #[arg(short, long, value_name = "FILE")]
    pub manifest: PathBuf,

    /// 输出根目录（覆盖清单中的 global.output_dir）
    #[arg(short = 'o', long, value_name = "DIR")]
    pub output_dir: Option<PathBuf>,

    /// 输出格式: parquet | text | binary（覆盖清单中的格式设置）
    #[arg(short = 'f', long, value_enum)]
    pub format: Option<CliFormat>,

    /// 并行线程数（默认为 CPU 逻辑核心数）
    #[arg(short = 't', long, value_name = "N")]
    pub num_threads: Option<usize>,

    /// 日志级别: trace | debug | info | warn | error
    #[arg(short = 'l', long, value_name = "LEVEL", default_value = "info")]
    pub log_level: String,

    /// 是否强制使用流式写出模式（覆盖清单设置）
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub stream_write: Option<bool>,

    /// 是否禁用进度条（适用于 CI/重定向场景）
    #[arg(long, default_value_t = false)]
    pub no_progress: bool,
}

/// 顶层 CLI 命令枚举
#[derive(Parser, Debug)]
#[command(name = "StructGen-rs")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "高性能程序化结构化数据生成器", long_about = None)]
pub struct CliCommand {
    #[command(subcommand)]
    pub command: Commands,
}

/// CLI 子命令
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// 运行生成流水线（从 YAML 清单）
    Run(CliArgs),

    /// 启动元胞自动机实时可视化
    View(ViewCliArgs),
}

/// View 子命令参数
#[derive(Parser, Debug, Clone)]
pub struct ViewCliArgs {
    /// 生成器名称: ca | ca2d | ca3d
    #[arg(short, long)]
    pub generator: String,

    /// 规则号(1D)或预设名: rule30, game_of_life, wireworld ...
    #[arg(short, long)]
    pub rule: Option<String>,

    /// 随机种子
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// 演化步数 (0=无限)
    #[arg(long, default_value = "200")]
    pub steps: usize,

    /// 1D 网格宽度
    #[arg(long, default_value = "128")]
    pub width: usize,

    /// 2D/3D 行数
    #[arg(long, default_value = "64")]
    pub rows: usize,

    /// 2D/3D 列数
    #[arg(long, default_value = "64")]
    pub cols: usize,

    /// 3D 深度
    #[arg(long, default_value = "16")]
    pub depth: usize,

    /// 动画速度(毫秒/帧)
    #[arg(long, default_value = "100")]
    pub speed: u64,

    /// 边界条件: periodic | fixed | reflective
    #[arg(long, default_value = "periodic")]
    pub boundary: String,

    /// 初始化模式: random | single_center
    #[arg(long, default_value = "random")]
    pub init: String,
}

impl From<ViewCliArgs> for view::ViewArgs {
    fn from(args: ViewCliArgs) -> Self {
        view::ViewArgs {
            generator: args.generator,
            rule: args.rule,
            seed: args.seed,
            steps: args.steps,
            width: args.width,
            rows: args.rows,
            cols: args.cols,
            depth: args.depth,
            speed: args.speed,
            boundary: args.boundary,
            init: args.init,
        }
    }
}

// ============================================================================
// 适配器工厂
// ============================================================================

/// 创建输出适配器：将 OutputFormat 映射到具体适配器实现
fn create_adapter(format: OutputFormat) -> CoreResult<Box<dyn SinkAdapter>> {
    match format {
        OutputFormat::Parquet => Ok(Box::new(ParquetAdapter::new())),
        OutputFormat::Text => Ok(Box::new(TextAdapter::new())),
        OutputFormat::Binary => Ok(Box::new(BinaryAdapter::new())),
        OutputFormat::Npy => Ok(Box::new(NpyAdapter::new())),
    }
}

// ============================================================================
// 主入口
// ============================================================================

/// 全局日志初始化守卫（整个进程生命周期仅初始化一次）
static LOGGER_ONCE: Once = Once::new();

/// 全局线程池初始化守卫（整个进程生命周期仅初始化一次）
static THREADPOOL_ONCE: Once = Once::new();

fn main() {
    // 设置自定义 panic hook，将 panic 信息以结构化格式输出
    std::panic::set_hook(Box::new(|info| {
        let msg = if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else {
            "Unknown panic".to_string()
        };
        log::error!("FATAL: {}", msg);
        eprintln!("FATAL ERROR: {}\nPlease report this bug.", msg);
    }));

    let cli = CliCommand::parse();
    let code = match cli.command {
        Commands::Run(args) => run(args),
        Commands::View(args) => view::launch_viewer(args.into()),
    };
    std::process::exit(code);
}

// ============================================================================
// 核心装配函数
// ============================================================================

/// 装配并运行完整的生成流水线
///
/// 负责按顺序初始化各子系统、加载清单、调度执行、输出元数据和汇总报告。
/// 返回进程退出码。
fn run(args: CliArgs) -> i32 {
    // 步骤1：初始化日志系统（进程生命周期内仅初始化一次）
    let mut init_err: Option<String> = None;
    let log_file = args.output_dir.as_ref().map(|d| d.join("structgen_run.log"));
    let log_path_ref = log_file.as_deref();
    let log_level = args.log_level.clone();
    LOGGER_ONCE.call_once(|| {
        if let Err(e) = init_logger(&log_level, log_path_ref) {
            init_err = Some(e.to_string());
        }
    });
    if let Some(e) = init_err {
        eprintln!("Error: failed to initialize logger: {}", e);
        return exit_code::FULL_FAILURE;
    }

    // 步骤2：验证清单文件存在
    if !args.manifest.exists() {
        eprintln!(
            "Error: manifest file not found: {}",
            args.manifest.display()
        );
        return exit_code::FULL_FAILURE;
    }

    // 步骤3：加载并解析 YAML 清单
    let yaml_content = match std::fs::read_to_string(&args.manifest) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error: cannot read manifest file: {}", e);
            return exit_code::FULL_FAILURE;
        }
    };

    let mut manifest = match scheduler::manifest::Manifest::from_yaml(&yaml_content) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: YAML parse error: {}", e);
            return exit_code::FULL_FAILURE;
        }
    };

    // 步骤4：应用命令行参数覆盖清单全局配置
    if let Some(ref dir) = args.output_dir {
        manifest.global.output_dir = dir.to_string_lossy().to_string();
    }
    if let Some(fmt) = args.format {
        manifest.global.default_format = fmt.into();
    }
    if let Some(n) = args.num_threads {
        if n == 0 {
            eprintln!("Error: num_threads must be >= 1");
            return exit_code::FULL_FAILURE;
        }
        manifest.global.num_threads = Some(n);
    }
    if let Some(sw) = args.stream_write {
        manifest.global.stream_write = sw;
    }

    // 确保输出目录已设置
    if manifest.global.output_dir.is_empty() {
        eprintln!("Error: output_dir is required (via --output-dir or manifest)");
        return exit_code::FULL_FAILURE;
    }

    // 步骤5：设置 rayon 全局线程池（进程生命周期仅一次）
    let num_threads = manifest.global.num_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });
    let mut tp_err: Option<String> = None;
    THREADPOOL_ONCE.call_once(|| {
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
        {
            tp_err = Some(e.to_string());
        }
    });
    if let Some(e) = tp_err {
        eprintln!("Error: failed to set thread pool: {}", e);
        return exit_code::FULL_FAILURE;
    }

    // 步骤6：初始化生成器注册表
    let mut gen_registry = crate::core::GeneratorRegistry::new();
    if let Err(e) = register_all_generators(&mut gen_registry) {
        eprintln!("Error: failed to register generators: {}", e);
        return exit_code::FULL_FAILURE;
    }

    // 步骤7：初始化处理器注册表
    let mut proc_registry = crate::pipeline::ProcessorRegistry::new();
    if let Err(e) = register_all_processors(&mut proc_registry) {
        eprintln!("Error: failed to register processors: {}", e);
        return exit_code::FULL_FAILURE;
    }

    // 步骤8：校验清单合法性（生成器/处理器引用等）
    if let Err(e) = manifest.validate(&gen_registry, &proc_registry) {
        eprintln!("Error: {}", e);
        return exit_code::FULL_FAILURE;
    }

    // 步骤9：创建输出目录
    let output_dir = std::path::Path::new(&manifest.global.output_dir);
    if let Err(e) = std::fs::create_dir_all(output_dir) {
        eprintln!("Error: cannot create output directory: {}", e);
        return exit_code::FULL_FAILURE;
    }

    // 步骤10：创建进度追踪器
    let total_samples: usize = manifest.tasks.iter().map(|t| t.count).sum();
    let progress = ProgressTracker::new(total_samples);

    // 步骤11：启动进度显示线程（若非 --no-progress）
    let done_flag = Arc::new(AtomicBool::new(false));
    let progress_handle = if !args.no_progress {
        let p = progress.clone();
        let done = done_flag.clone();
        Some(std::thread::spawn(move || {
            progress_display_loop(p, done);
        }))
    } else {
        None
    };

    // 步骤12：记录开始时间，调用调度器
    let start_time = Utc::now();
    let results = match scheduler::run_manifest(
        &manifest,
        &gen_registry,
        &proc_registry,
        create_adapter,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: scheduler failed: {}", e);
            done_flag.store(true, Ordering::SeqCst);
            if let Some(handle) = progress_handle {
                let _ = handle.join();
            }
            return exit_code::FULL_FAILURE;
        }
    };
    let end_time = Utc::now();

    // 步骤13：停止进度显示线程
    done_flag.store(true, Ordering::SeqCst);
    if let Some(handle) = progress_handle {
        let _ = handle.join();
    }

    // 步骤14：写入元数据
    if let Err(e) = write_metadata(output_dir, &manifest, &results, start_time, end_time) {
        eprintln!("Warning: failed to write metadata.json: {}", e);
    } else {
        tracing::info!("metadata.json 已写入 {}", output_dir.display());
    }

    // 步骤15：打印汇总报告
    print_summary(&manifest.global.output_dir, &results, start_time, end_time);

    // 步骤16：确定退出码
    let all_success = results.iter().all(|r| r.error.is_none());
    let any_success = results.iter().any(|r| r.error.is_none());

    if all_success {
        exit_code::SUCCESS
    } else if any_success {
        exit_code::PARTIAL_FAILURE
    } else {
        exit_code::FULL_FAILURE
    }
}

// ============================================================================
// 进度条显示
// ============================================================================

/// 进度条显示循环，通过 `\r` 覆盖式输出到 stderr
///
/// 每 200ms 刷新一次。若 stderr 不是 TTY（如管道重定向），降级为每 5 秒一行日志。
fn progress_display_loop(tracker: ProgressTracker, done: Arc<AtomicBool>) {
    let stderr_is_tty = std::io::stderr().is_terminal();
    let mut last_log_time = std::time::Instant::now();

    loop {
        if done.load(Ordering::Relaxed) {
            break;
        }

        let info = tracker.progress();

        if stderr_is_tty {
            // TTY 环境：覆盖式进度条
            let bar_width = 20usize;
            let filled = if info.total_samples > 0 {
                (info.percent / 100.0 * bar_width as f64) as usize
            } else {
                0
            };
            let filled = filled.min(bar_width);
            let empty = bar_width - filled;

            let eta_str = format_eta(info.eta_secs);

            eprint!(
                "\r[{}{}] {:>5.1}% ({}/{} samples) | {} frames | ETA: {}",
                "=".repeat(filled),
                " ".repeat(empty),
                info.percent,
                info.completed_samples,
                info.total_samples,
                info.total_frames,
                eta_str,
            );
            let _ = std::io::stderr().flush();
        } else {
            // 非 TTY 环境：每 5 秒一行日志
            let elapsed = last_log_time.elapsed().as_secs_f64();
            if elapsed >= 5.0 || info.completed_samples >= info.total_samples {
                tracing::info!(
                    "进度: {:.1}% ({}/{} samples) | {} frames",
                    info.percent,
                    info.completed_samples,
                    info.total_samples,
                    info.total_frames,
                );
                last_log_time = std::time::Instant::now();
            }
        }

        // 检查是否已完成
        if info.total_samples > 0 && info.completed_samples >= info.total_samples {
            break;
        }

        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    // 最终换行
    if stderr_is_tty {
        eprintln!();
    }
}

/// 格式化 ETA 秒数为 HH:MM:SS 格式
fn format_eta(total_seconds: f64) -> String {
    if total_seconds <= 0.0 || !total_seconds.is_finite() {
        return "--:--:--".to_string();
    }
    let total = total_seconds as u64;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

// ============================================================================
// 汇总报告
// ============================================================================

/// 打印格式化汇总报告到 stdout
fn print_summary(
    output_dir: &str,
    results: &[crate::scheduler::shard::ShardResult],
    start_time: chrono::DateTime<Utc>,
    end_time: chrono::DateTime<Utc>,
) {
    let elapsed = end_time - start_time;
    let elapsed_str = format_duration(elapsed);

    println!();
    println!("============================================================");
    println!(
        " StructGen-rs v{} - Run Summary",
        env!("CARGO_PKG_VERSION")
    );
    println!("============================================================");
    println!(" Output directory: {}", output_dir);
    println!(" Elapsed time:     {}", elapsed_str);
    println!();

    // 汇总存储
    let total_samples: usize = results.iter().map(|r| r.sample_count).sum();
    let total_frames: u64 = results.iter().map(|r| r.stats.frames_written).sum();
    let total_bytes: u64 = results.iter().map(|r| r.stats.bytes_written).sum();
    let succeeded: Vec<_> = results.iter().filter(|r| r.error.is_none()).collect();
    let total_files = succeeded.len();

    // 按任务分组统计
    use std::collections::BTreeMap;
    let mut task_map: BTreeMap<String, Vec<&crate::scheduler::shard::ShardResult>> =
        BTreeMap::new();
    for r in results {
        task_map
            .entry(r.task_name.clone())
            .or_default()
            .push(r);
    }

    println!(" Tasks:");
    for (task_name, task_results) in &task_map {
        let task_samples: usize = task_results.iter().map(|r| r.sample_count).sum();
        let task_frames: u64 = task_results.iter().map(|r| r.stats.frames_written).sum();
        let task_bytes: u64 = task_results.iter().map(|r| r.stats.bytes_written).sum();
        let failed_count = task_results.iter().filter(|r| r.error.is_some()).count();

        let status = if failed_count == 0 {
            "[OK]".to_string()
        } else {
            format!("[{} shards FAILED]", failed_count)
        };

        println!(
            "   {:<20} {:>12} samples | {:>14} frames | {:>8}  {}",
            task_name,
            task_samples,
            task_frames,
            format_bytes(task_bytes),
            status,
        );
    }

    let failed_count: usize = results.iter().filter(|r| r.error.is_some()).count();

    println!();
    println!(" Totals:");
    println!("   Samples:  {} ({} failed)", total_samples, failed_count);
    println!("   Frames:   {}", total_frames);
    println!("   Data:     {}", format_bytes(total_bytes));
    println!("   Files:    {}", total_files);
    println!();
    println!(" Metadata: {}/metadata.json", output_dir);

    let exit_code = if failed_count == 0 {
        exit_code::SUCCESS
    } else if succeeded.is_empty() {
        exit_code::FULL_FAILURE
    } else {
        exit_code::PARTIAL_FAILURE
    };

    let exit_desc = match exit_code {
        exit_code::SUCCESS => "success",
        exit_code::PARTIAL_FAILURE => "partial failure — see metadata.json for details",
        exit_code::FULL_FAILURE => "full failure",
        _ => "unknown",
    };
    println!(" Exit code: {} ({})", exit_code, exit_desc);
    println!("============================================================");
}

/// 格式化 chrono::Duration 为 HH:MM:SS.mmm
fn format_duration(dur: chrono::TimeDelta) -> String {
    let total_ms = dur.num_milliseconds();
    let total_secs = dur.num_seconds();
    let hours = (total_secs / 3600) as u64;
    let minutes = ((total_secs % 3600) / 60) as u64;
    let seconds = (total_secs % 60) as u64;
    let millis = (total_ms % 1000) as u64;
    if hours > 0 {
        format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
    } else {
        format!("{:02}:{:02}.{:03}", minutes, seconds, millis)
    }
}

/// 格式化字节数为人类可读的字符串
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[unit_idx])
    } else {
        format!("{:.1} {}", size, UNITS[unit_idx])
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    // ------------------------------------------------------------------
    // CliArgs 解析测试
    // ------------------------------------------------------------------

    #[test]
    fn test_parse_minimal_args() {
        let args = CliArgs::parse_from(["structgen-rs", "--manifest", "tasks.yaml"]);
        assert_eq!(args.manifest, PathBuf::from("tasks.yaml"));
        assert_eq!(args.log_level, "info");
        assert!(args.output_dir.is_none());
        assert!(args.format.is_none());
        assert!(args.num_threads.is_none());
        assert!(args.stream_write.is_none());
        assert!(!args.no_progress);
    }

    #[test]
    fn test_parse_short_args() {
        let args = CliArgs::parse_from([
            "structgen-rs",
            "-m", "tasks.yaml",
            "-o", "/tmp/out",
            "-f", "text",
            "-t", "4",
            "-l", "debug",
        ]);
        assert_eq!(args.manifest, PathBuf::from("tasks.yaml"));
        assert_eq!(args.output_dir, Some(PathBuf::from("/tmp/out")));
        assert!(matches!(args.format, Some(CliFormat::Text)));
        assert_eq!(args.num_threads, Some(4));
        assert_eq!(args.log_level, "debug");
    }

    #[test]
    fn test_parse_full_args() {
        let args = CliArgs::parse_from([
            "structgen-rs",
            "--manifest", "tasks.yaml",
            "--output-dir", "/tmp/out",
            "--format", "binary",
            "--num-threads", "8",
            "--log-level", "trace",
            "--stream-write",
            "--no-progress",
        ]);
        assert_eq!(args.output_dir, Some(PathBuf::from("/tmp/out")));
        assert!(matches!(args.format, Some(CliFormat::Binary)));
        assert_eq!(args.num_threads, Some(8));
        assert_eq!(args.log_level, "trace");
        assert_eq!(args.stream_write, Some(true));
        assert!(args.no_progress);
    }

    #[test]
    fn test_parse_stream_write_false() {
        let args = CliArgs::parse_from([
            "structgen-rs",
            "--manifest", "tasks.yaml",
            "--stream-write=false",
        ]);
        assert_eq!(args.stream_write, Some(false));
    }

    // ------------------------------------------------------------------
    // CliFormat 映射测试
    // ------------------------------------------------------------------

    #[test]
    fn test_cli_format_to_output_format() {
        assert_eq!(OutputFormat::from(CliFormat::Parquet), OutputFormat::Parquet);
        assert_eq!(OutputFormat::from(CliFormat::Text), OutputFormat::Text);
        assert_eq!(OutputFormat::from(CliFormat::Binary), OutputFormat::Binary);
    }

    // ------------------------------------------------------------------
    // create_adapter 工厂测试
    // ------------------------------------------------------------------

    #[test]
    fn test_create_adapter_factory() {
        let adapter = create_adapter(OutputFormat::Text).unwrap();
        assert_eq!(adapter.format(), OutputFormat::Text);

        let adapter = create_adapter(OutputFormat::Parquet).unwrap();
        assert_eq!(adapter.format(), OutputFormat::Parquet);

        let adapter = create_adapter(OutputFormat::Binary).unwrap();
        assert_eq!(adapter.format(), OutputFormat::Binary);
    }

    // ------------------------------------------------------------------
    // format_eta 测试
    // ------------------------------------------------------------------

    #[test]
    fn test_format_eta_zero() {
        assert_eq!(format_eta(0.0), "--:--:--");
    }

    #[test]
    fn test_format_eta_negative() {
        assert_eq!(format_eta(-1.0), "--:--:--");
    }

    #[test]
    fn test_format_eta_seconds_only() {
        assert_eq!(format_eta(45.0), "00:00:45");
    }

    #[test]
    fn test_format_eta_minutes_and_seconds() {
        assert_eq!(format_eta(125.0), "00:02:05");
    }

    #[test]
    fn test_format_eta_hours() {
        assert_eq!(format_eta(3661.0), "01:01:01");
    }

    // ------------------------------------------------------------------
    // format_bytes 测试
    // ------------------------------------------------------------------

    #[test]
    fn test_format_bytes_zero() {
        assert_eq!(format_bytes(0), "0 B");
    }

    #[test]
    fn test_format_bytes_bytes() {
        assert_eq!(format_bytes(512), "512 B");
    }

    #[test]
    fn test_format_bytes_kb() {
        assert_eq!(format_bytes(2048), "2.0 KB");
    }

    #[test]
    fn test_format_bytes_mb() {
        assert_eq!(format_bytes(5_242_880), "5.0 MB");
    }

    #[test]
    fn test_format_bytes_gb() {
        assert_eq!(format_bytes(3_221_225_472), "3.0 GB");
    }

    // ------------------------------------------------------------------
    // format_duration 测试
    // ------------------------------------------------------------------

    #[test]
    fn test_format_duration_short() {
        let dur = chrono::TimeDelta::milliseconds(5123);
        let s = format_duration(dur);
        assert!(s.contains("05.123"), "got: {}", s);
    }

    #[test]
    fn test_format_duration_long() {
        let dur = chrono::TimeDelta::milliseconds(3_661_789);
        let s = format_duration(dur);
        assert!(s.contains("01:01:01"), "got: {}", s);
    }

    // ------------------------------------------------------------------
    // exit_code 常量测试
    // ------------------------------------------------------------------

    #[test]
    fn test_exit_code_values() {
        assert_eq!(exit_code::SUCCESS, 0);
        assert_eq!(exit_code::PARTIAL_FAILURE, 1);
        assert_eq!(exit_code::FULL_FAILURE, 2);
    }
}

// ============================================================================
// 集成测试
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::io::Write as _;
    use tempfile::TempDir;

    /// 构建最小可用的 YAML 清单内容
    fn make_minimal_yaml(output_dir: &str) -> String {
        // Windows 路径使用反斜杠，在 YAML 中会被解释为转义序列
        // 使用正斜杠（Windows 也接受）来避免 YAML 解析错误
        let safe_dir = output_dir.replace('\\', "/");
        format!(
            r#"
global:
  output_dir: "{}"
  default_format: "Text"
tasks:
  - name: "test_task"
    generator: "ca"
    params:
      seq_length: 10
    count: 10
    seed: 42
    pipeline: []
    output_format: "Text"
"#,
            safe_dir
        )
    }

    /// 写入临时 YAML 文件并返回路径
    fn write_temp_yaml(dir: &std::path::Path, content: &str) -> std::path::PathBuf {
        let path = dir.join("test_manifest.yaml");
        std::fs::write(&path, content).unwrap();
        path
    }

    // ------------------------------------------------------------------
    // 退出码：清单文件不存在
    // ------------------------------------------------------------------

    #[test]
    fn test_exit_code_manifest_not_found() {
        let args = CliArgs::parse_from([
            "structgen-rs",
            "--manifest", "nonexistent_file.yaml",
            "--output-dir", "/tmp/does_not_matter",
            "--no-progress",
        ]);
        let code = run(args);
        assert_eq!(code, exit_code::FULL_FAILURE);
    }

    // ------------------------------------------------------------------
    // 退出码：YAML 解析失败
    // ------------------------------------------------------------------

    #[test]
    fn test_exit_code_yaml_parse_error() {
        let tmp = TempDir::new().unwrap();
        let manifest_path = write_temp_yaml(tmp.path(), "this is not valid yaml: [[[");
        let output_dir = tmp.path().join("output");

        let args = CliArgs::parse_from([
            "structgen-rs",
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--no-progress",
        ]);
        let code = run(args);
        assert_eq!(code, exit_code::FULL_FAILURE);
    }

    // ------------------------------------------------------------------
    // 退出码：无效 YAML 内容（空任务列表）
    // ------------------------------------------------------------------

    #[test]
    fn test_exit_code_empty_tasks() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("output");
        let yaml = format!(
            r#"
global:
  output_dir: "{}"
tasks: []
"#,
            output_dir.display().to_string().replace('\\', "/")
        );
        let manifest_path = write_temp_yaml(tmp.path(), &yaml);

        let args = CliArgs::parse_from([
            "structgen-rs",
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--no-progress",
        ]);
        let code = run(args);
        assert_eq!(code, exit_code::FULL_FAILURE);
    }

    // ------------------------------------------------------------------
    // 退出码：未注册的生成器
    // ------------------------------------------------------------------

    #[test]
    fn test_exit_code_unknown_generator() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("output");
        let yaml = format!(
            r#"
global:
  output_dir: "{}"
tasks:
  - name: "bad_task"
    generator: "nonexistent_gen"
    params:
      seq_length: 10
    count: 10
    seed: 42
"#,
            output_dir.display().to_string().replace('\\', "/")
        );
        let manifest_path = write_temp_yaml(tmp.path(), &yaml);

        let args = CliArgs::parse_from([
            "structgen-rs",
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--no-progress",
        ]);
        let code = run(args);
        assert_eq!(code, exit_code::FULL_FAILURE);
    }

    // ------------------------------------------------------------------
    // 端到端：最小 YAML → 产出数据文件 + metadata.json
    // ------------------------------------------------------------------

    #[test]
    fn test_end_to_end_success() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("output");
        let yaml = make_minimal_yaml(&output_dir.display().to_string());
        let manifest_path = write_temp_yaml(tmp.path(), &yaml);

        let args = CliArgs::parse_from([
            "structgen-rs",
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--no-progress",
        ]);
        let code = run(args);

        // 验证退出码为成功
        assert_eq!(code, exit_code::SUCCESS, "全成功应以退出码 0 退出");

        // 验证输出目录中有文件
        let entries: Vec<_> = std::fs::read_dir(&output_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert!(!entries.is_empty(), "输出目录应该有文件产出");

        // 验证 metadata.json 存在且包含关键字段
        let metadata_path = output_dir.join("metadata.json");
        assert!(metadata_path.exists(), "metadata.json 应该被生成");

        let content = std::fs::read_to_string(&metadata_path).unwrap();
        assert!(content.contains("\"software_version\""), "应包含 software_version");
        assert!(content.contains("\"start_time\""), "应包含 start_time");
        assert!(content.contains("\"end_time\""), "应包含 end_time");
        assert!(content.contains("\"global_config\""), "应包含 global_config");
        assert!(content.contains("\"summary\""), "应包含 summary");
        assert!(content.contains("\"tasks\""), "应包含 tasks");
        assert!(content.contains("\"test_task\""), "应包含任务名称 test_task");
    }

    // ------------------------------------------------------------------
    // 命令行参数覆盖清单配置
    // ------------------------------------------------------------------

    #[test]
    fn test_cli_overrides_manifest_config() {
        let tmp = TempDir::new().unwrap();
        let default_output = tmp.path().join("default_output");
        let override_output = tmp.path().join("override_output");
        let yaml = make_minimal_yaml(&default_output.display().to_string());
        let manifest_path = write_temp_yaml(tmp.path(), &yaml);

        // 用 --output-dir 覆盖清单中的输出目录
        let args = CliArgs::parse_from([
            "structgen-rs",
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--output-dir",
            override_output.to_str().unwrap(),
            "--no-progress",
        ]);
        let code = run(args);

        assert_eq!(code, exit_code::SUCCESS);

        // 验证文件产出在覆盖后的目录
        assert!(override_output.exists(), "覆盖后的输出目录应存在");
        assert!(override_output.join("metadata.json").exists(), "metadata.json 应在覆盖后的目录");
    }

    // ------------------------------------------------------------------
    // 退出码：num_threads 为 0
    // ------------------------------------------------------------------

    #[test]
    fn test_exit_code_zero_threads() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("output");
        let yaml = make_minimal_yaml(&output_dir.display().to_string());
        let manifest_path = write_temp_yaml(tmp.path(), &yaml);

        let args = CliArgs::parse_from([
            "structgen-rs",
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--num-threads", "0",
            "--no-progress",
        ]);
        let code = run(args);
        assert_eq!(code, exit_code::FULL_FAILURE);
    }
}
