use thiserror::Error;

/// core 模块统一错误类型
#[derive(Error, Debug)]
pub enum CoreError {
    /// 无效参数
    #[error("invalid parameters: {0}")]
    InvalidParams(String),

    /// 未找到指定名称的生成器
    #[error("generator not found: {0}")]
    GeneratorNotFound(String),

    /// 生成器初始化失败
    #[error("generator initialization failed: {0}")]
    GeneratorInitError(String),

    /// 生成过程出错
    #[error("generation error: {0}")]
    GenerationError(String),

    /// I/O 错误（由 std::io::Error 自动转换）
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// 序列化/反序列化错误
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// 清单文件错误
    #[error("manifest error: {0}")]
    ManifestError(String),

    /// 管道错误
    #[error("pipeline error: {0}")]
    PipelineError(String),

    /// 数据汇错误
    #[error("sink error: {0}")]
    SinkError(String),

    /// 配置错误
    #[error("config error: {0}")]
    ConfigError(String),

    /// 其他未分类错误
    #[error("{0}")]
    Other(String),
}

/// core 模块 Result 类型别名
pub type CoreResult<T> = Result<T, CoreError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_error_display() {
        let err = CoreError::GeneratorNotFound("ca".to_string());
        assert_eq!(err.to_string(), "generator not found: ca");
    }

    #[test]
    fn test_io_error_from_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let core_err: CoreError = io_err.into();
        assert!(matches!(core_err, CoreError::IoError(_)));
        assert!(core_err.to_string().contains("file not found"));
    }

    #[test]
    fn test_io_error_propagation() {
        fn fallible() -> CoreResult<()> {
            Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied").into())
        }
        let result = fallible();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CoreError::IoError(_)));
    }

    #[test]
    fn test_invalid_params_display() {
        let err = CoreError::InvalidParams("seq_length must be > 0".to_string());
        assert_eq!(
            err.to_string(),
            "invalid parameters: seq_length must be > 0"
        );
    }

    #[test]
    fn test_config_error_display() {
        let err = CoreError::ConfigError("num_threads cannot be 0".to_string());
        assert_eq!(err.to_string(), "config error: num_threads cannot be 0");
    }

    #[test]
    fn test_other_error_display() {
        let err = CoreError::Other("unknown error".to_string());
        assert_eq!(err.to_string(), "unknown error");
    }
}
