//! 输出适配器工厂类型定义
//!
//! 提供 SinkAdapterFactory 类型别名，用于按 OutputFormat 创建适配器实例。

use crate::core::*;

use super::adapter::SinkAdapter;

/// 适配器构造函数类型：接受 OutputFormat，返回 SinkAdapter trait 对象
pub type SinkAdapterFactory = fn(OutputFormat) -> CoreResult<Box<dyn SinkAdapter>>;
