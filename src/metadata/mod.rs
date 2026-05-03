//! StructGen-rs 元数据与监控层 (metadata)
//!
//! 提供运行元数据记录、进度追踪和日志系统初始化功能。

pub mod logger;
pub mod types;
pub mod progress;
pub mod recorder;

pub use types::*;
pub use progress::*;
pub use recorder::write_metadata;
pub use logger::init_logger;
