use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::error::CoreError;

/// 输出文件格式枚举
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Apache Parquet 列式存储
    #[default]
    Parquet,
    /// 纯文本（令牌映射后的 Unicode 序列）
    Text,
    /// 内存映射二进制原始转储
    Binary,
}

/// 日志级别枚举
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

/// 全局配置，适用于整个运行而非单个任务
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfig {
    /// 并行线程数，None 表示自动检测（= CPU 逻辑核心数）
    #[serde(default)]
    pub num_threads: Option<usize>,
    /// 默认输出格式，可被任务级配置覆盖
    #[serde(default)]
    pub default_format: OutputFormat,
    /// 输出根目录
    #[serde(default)]
    pub output_dir: String,
    /// 日志级别
    #[serde(default)]
    pub log_level: LogLevel,
    /// 每个输出分片文件的最大序列数，超出后自动切分
    #[serde(default = "default_shard_max")]
    pub shard_max_sequences: usize,
    /// 流式写出模式（true） vs 阻塞收集模式（false）
    #[serde(default = "default_true")]
    pub stream_write: bool,
}

fn default_shard_max() -> usize {
    10000
}

fn default_true() -> bool {
    true
}

impl Default for GlobalConfig {
    fn default() -> Self {
        GlobalConfig {
            num_threads: None,
            default_format: OutputFormat::Parquet,
            output_dir: String::new(),
            log_level: LogLevel::default(),
            shard_max_sequences: default_shard_max(),
            stream_write: default_true(),
        }
    }
}

impl GlobalConfig {
    /// 校验配置的合法性
    ///
    /// # Errors
    /// 当配置项不合法时返回 [`CoreError::ConfigError`]
    pub fn validate(&self) -> Result<(), CoreError> {
        if let Some(n) = self.num_threads {
            if n == 0 {
                return Err(CoreError::ConfigError("num_threads cannot be 0".into()));
            }
        }
        if self.output_dir.is_empty() {
            return Err(CoreError::ConfigError("output_dir cannot be empty".into()));
        }
        if self.shard_max_sequences == 0 {
            return Err(CoreError::ConfigError(
                "shard_max_sequences cannot be 0".into(),
            ));
        }
        Ok(())
    }
}

/// 生成器的通用参数载体，包含所有生成器共享的字段以及动态扩展
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenParams {
    /// 目标序列长度（要生成的帧数），0 表示无限制（由外部迭代决定）
    #[serde(default)]
    pub seq_length: usize,
    /// 网格尺寸（对 CA、布尔网络等有空间维度概念的生成器），None 表示不适用
    #[serde(default)]
    pub grid_size: Option<GridSize>,
    /// 动态扩展字段，承载生成器特有参数（以 JSON Value 形式存储）
    #[serde(default)]
    pub extensions: HashMap<String, Value>,
}

/// 二维网格尺寸
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridSize {
    pub rows: usize,
    pub cols: usize,
}

impl GridSize {
    /// 校验网格尺寸的合法性
    ///
    /// # Errors
    /// 当 rows 或 cols 为 0 时返回 [`CoreError::InvalidParams`]
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.rows == 0 {
            return Err(CoreError::InvalidParams(
                "grid_size.rows cannot be 0".into(),
            ));
        }
        if self.cols == 0 {
            return Err(CoreError::InvalidParams(
                "grid_size.cols cannot be 0".into(),
            ));
        }
        Ok(())
    }
}

impl GenParams {
    /// 创建最简参数（仅指定序列长度）
    pub fn simple(seq_length: usize) -> Self {
        GenParams {
            seq_length,
            grid_size: None,
            extensions: HashMap::new(),
        }
    }

    /// 从扩展字段中提取并反序列化生成器特有参数
    pub fn get_extension<T: serde::de::DeserializeOwned>(&self, key: &str) -> Result<T, CoreError> {
        let value = self.extensions.get(key).ok_or_else(|| {
            CoreError::InvalidParams(format!("extension key '{}' not found", key))
        })?;
        serde_json::from_value(value.clone()).map_err(|e| {
            CoreError::SerializationError(format!(
                "failed to deserialize extension '{}': {}",
                key, e
            ))
        })
    }

    /// 向扩展字段中插入生成器特有参数
    pub fn set_extension<T: Serialize>(&mut self, key: &str, value: &T) -> Result<(), CoreError> {
        let json_value = serde_json::to_value(value).map_err(|e| {
            CoreError::SerializationError(format!(
                "failed to serialize extension '{}': {}",
                key, e
            ))
        })?;
        self.extensions.insert(key.to_string(), json_value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_grid_size_construction() {
        let gs = GridSize { rows: 10, cols: 20 };
        assert_eq!(gs.rows, 10);
        assert_eq!(gs.cols, 20);
    }

    #[test]
    fn test_grid_size_serialization() {
        let gs = GridSize { rows: 8, cols: 16 };
        let json = serde_json::to_string(&gs).unwrap();
        let restored: GridSize = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, gs);
    }

    #[test]
    fn test_grid_size_validate_valid() {
        let gs = GridSize { rows: 1, cols: 1 };
        assert!(gs.validate().is_ok());
    }

    #[test]
    fn test_grid_size_validate_zero_rows() {
        let gs = GridSize { rows: 0, cols: 10 };
        let result = gs.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            CoreError::InvalidParams(msg) => assert!(msg.contains("rows")),
            _ => panic!("expected InvalidParams"),
        }
    }

    #[test]
    fn test_grid_size_validate_zero_cols() {
        let gs = GridSize { rows: 10, cols: 0 };
        let result = gs.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            CoreError::InvalidParams(msg) => assert!(msg.contains("cols")),
            _ => panic!("expected InvalidParams"),
        }
    }

    #[test]
    fn test_output_format_serialization() {
        let fmt = OutputFormat::Parquet;
        let json = serde_json::to_string(&fmt).unwrap();
        assert!(json.contains("Parquet"));
        let restored: OutputFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, OutputFormat::Parquet);
    }

    #[test]
    fn test_log_level_serialization() {
        let level = LogLevel::Debug;
        let json = serde_json::to_string(&level).unwrap();
        assert!(json.contains("debug"));
        let restored: LogLevel = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, LogLevel::Debug);
    }

    #[test]
    fn test_log_level_default() {
        assert_eq!(LogLevel::default(), LogLevel::Info);
    }

    #[test]
    fn test_log_level_deserialization() {
        let level: LogLevel = serde_json::from_value(json!("warn")).unwrap();
        assert_eq!(level, LogLevel::Warn);
    }

    #[test]
    fn test_log_level_invalid_string() {
        let result: Result<LogLevel, _> = serde_json::from_value(json!("banana"));
        assert!(result.is_err());
    }

    #[test]
    fn test_gen_params_simple() {
        let params = GenParams::simple(100);
        assert_eq!(params.seq_length, 100);
        assert!(params.grid_size.is_none());
        assert!(params.extensions.is_empty());
    }

    #[test]
    fn test_gen_params_extension_roundtrip() {
        let mut params = GenParams::simple(100);
        params.set_extension("rule", &110u32).unwrap();
        let rule: u32 = params.get_extension("rule").unwrap();
        assert_eq!(rule, 110);
    }

    #[test]
    fn test_gen_params_extension_float() {
        let mut params = GenParams::simple(50);
        params.set_extension("sigma", &10.0f64).unwrap();
        let sigma: f64 = params.get_extension("sigma").unwrap();
        assert!((sigma - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_gen_params_extension_not_found() {
        let params = GenParams::simple(10);
        let result: Result<u32, _> = params.get_extension("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_gen_params_extension_wrong_type() {
        let mut params = GenParams::simple(10);
        params.set_extension("value", &"hello").unwrap();
        let result: Result<u32, _> = params.get_extension("value");
        assert!(result.is_err());
    }

    #[test]
    fn test_gen_params_serialization() {
        let mut params = GenParams::simple(256);
        params.set_extension("rule", &30u32).unwrap();
        params.grid_size = Some(GridSize { rows: 1, cols: 128 });
        let json = serde_json::to_string(&params).unwrap();
        let restored: GenParams = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.seq_length, 256);
        assert_eq!(
            restored.grid_size,
            Some(GridSize { rows: 1, cols: 128 })
        );
        let rule: u32 = restored.get_extension("rule").unwrap();
        assert_eq!(rule, 30);
    }

    #[test]
    fn test_global_config_default() {
        let config = GlobalConfig::default();
        assert_eq!(config.default_format, OutputFormat::Parquet);
        assert_eq!(config.log_level, LogLevel::Info);
        assert_eq!(config.shard_max_sequences, 10000);
        assert!(config.stream_write);
        assert!(config.num_threads.is_none());
    }

    #[test]
    fn test_global_config_deserialization() {
        let json = json!({
            "num_threads": 4,
            "default_format": "Text",
            "output_dir": "./output",
            "log_level": "debug",
            "shard_max_sequences": 5000,
            "stream_write": false
        });
        let config: GlobalConfig = serde_json::from_value(json).unwrap();
        assert_eq!(config.num_threads, Some(4));
        assert_eq!(config.default_format, OutputFormat::Text);
        assert_eq!(config.output_dir, "./output");
        assert_eq!(config.log_level, LogLevel::Debug);
        assert_eq!(config.shard_max_sequences, 5000);
        assert!(!config.stream_write);
    }

    #[test]
    fn test_global_config_validate_valid() {
        let config = GlobalConfig {
            output_dir: "./output".into(),
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_global_config_validate_auto_threads() {
        // num_threads = None (自动检测) 应合法
        let config = GlobalConfig {
            num_threads: None,
            output_dir: "./output".into(),
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_global_config_validate_zero_threads() {
        let config = GlobalConfig {
            num_threads: Some(0),
            output_dir: "./output".into(),
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            CoreError::ConfigError(msg) => assert!(msg.contains("num_threads")),
            _ => panic!("expected ConfigError"),
        }
    }

    #[test]
    fn test_global_config_validate_empty_output_dir() {
        let config = GlobalConfig {
            output_dir: String::new(),
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            CoreError::ConfigError(msg) => assert!(msg.contains("output_dir")),
            _ => panic!("expected ConfigError"),
        }
    }

    #[test]
    fn test_global_config_validate_zero_shard_max() {
        let config = GlobalConfig {
            output_dir: "./output".into(),
            shard_max_sequences: 0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            CoreError::ConfigError(msg) => assert!(msg.contains("shard_max_sequences")),
            _ => panic!("expected ConfigError"),
        }
    }
}
