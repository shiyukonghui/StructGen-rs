//! 输出适配器工厂类型定义
//!
//! 提供 SinkAdapterFactory 类型别名，用于按 OutputFormat 创建适配器实例。

use crate::core::*;
use serde_json::Value;

use super::adapter::SinkAdapter;

/// 适配器构造函数类型
///
/// # 参数
/// - `OutputFormat`: 输出格式，决定创建哪种适配器
/// - `&Value`: 配置参数，用于传递输出器配置（如 NpyBatchConfig）
///
/// # 返回
/// 返回 SinkAdapter trait 对象
pub type SinkAdapterFactory = fn(OutputFormat, &Value) -> CoreResult<Box<dyn SinkAdapter>>;
