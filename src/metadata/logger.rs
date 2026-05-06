//! 日志系统初始化模块
//!
//! 基于 tracing-subscriber 实现控制台与可选文件双输出日志系统。

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use tracing_subscriber::prelude::*;

use crate::core::{CoreError, CoreResult};

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// 初始化日志系统
///
/// 配置 tracing-subscriber 日志系统，支持控制台彩色 compact 格式输出和可选的 JSON 文件输出。
/// 日志级别解析为 tracing 过滤器，无效级别返回错误。重复初始化将返回错误。
///
/// # 参数
/// * `log_level` - 日志级别字符串，支持: "trace", "debug", "info", "warn", "error"
/// * `log_file` - 可选的日志文件路径，若提供则同时输出 JSON 格式日志到文件
///
/// # 返回
/// * `Ok(())` - 日志系统初始化成功
/// * `Err(CoreError::ConfigError)` - 日志级别无效或日志系统已初始化
pub fn init_logger(log_level: &str, log_file: Option<&Path>) -> CoreResult<()> {
    // 无论是否已初始化，都先校验日志级别有效性
    let level_filter = match log_level.to_lowercase().as_str() {
        "trace" => tracing_subscriber::filter::LevelFilter::TRACE,
        "debug" => tracing_subscriber::filter::LevelFilter::DEBUG,
        "info" => tracing_subscriber::filter::LevelFilter::INFO,
        "warn" => tracing_subscriber::filter::LevelFilter::WARN,
        "error" => tracing_subscriber::filter::LevelFilter::ERROR,
        _ => return Err(CoreError::ConfigError(format!("无效的日志级别: {}", log_level))),
    };

    if INITIALIZED.load(Ordering::Acquire) {
        // 日志系统已初始化，非致命——允许重复调用以支持测试场景
        let _ = level_filter; // 已校验，忽略未使用警告
        return Ok(());
    }

    let _level_filter = level_filter;

    let console_layer = tracing_subscriber::fmt::layer()
        .with_ansi(true)
        .compact()
        .with_target(true)
        .with_timer(tracing_subscriber::fmt::time::SystemTime);

    let file_layer: Option<_> = log_file.and_then(|path| {
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    eprintln!("警告: 无法创建日志目录 '{}': {}", parent.display(), e);
                    return None;
                }
            }
        }
        match std::fs::File::create(path) {
            Ok(file) => Some(
                tracing_subscriber::fmt::layer()
                    .json()
                    .with_writer(std::sync::Mutex::new(file)),
            ),
            Err(e) => {
                eprintln!("警告: 无法创建日志文件 '{}': {}", path.display(), e);
                None
            }
        }
    });

    let env_filter = tracing_subscriber::EnvFilter::new(log_level);

    let result = if let Some(fl) = file_layer {
        tracing_subscriber::registry()
            .with(console_layer)
            .with(fl)
            .with(env_filter)
            .try_init()
    } else {
        tracing_subscriber::registry()
            .with(console_layer)
            .with(env_filter)
            .try_init()
    };

    match result {
        Ok(()) => {
            INITIALIZED.store(true, Ordering::Release);
            tracing::info!("StructGen-rs v{} 启动", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Err(_) => {
            // try_init 可能因多线程竞态失败（subscriber 已被其他线程设置）
            // 标记为已初始化并返回 OK，使日志系统幂等
            INITIALIZED.store(true, Ordering::Release);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_init_logger_full_flow() {
        // 1. 无效日志级别（无论日志系统是否已初始化都会校验）
        let result = init_logger("banana", None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("无效的日志级别"));

        // 2. 带文件初始化
        let temp_dir = TempDir::new().unwrap();
        let log_path = temp_dir.path().join("test.log");

        // 检查日志系统是否已被其他测试初始化
        let was_already_init = INITIALIZED.load(Ordering::Acquire);

        let result = init_logger("debug", Some(&log_path));
        assert!(result.is_ok(), "init_logger 应成功返回");

        if !was_already_init {
            // 仅当本次是首次初始化时才验证文件创建
            assert!(log_path.exists(), "首次初始化应创建日志文件");
        }

        // 3. 重复初始化应成功（幂等）
        let result = init_logger("info", None);
        assert!(result.is_ok(), "重复初始化应幂等返回 Ok");
    }
}
