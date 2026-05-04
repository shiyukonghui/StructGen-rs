//! StructGen-rs 核心抽象层，定义所有公共数据类型与核心接口
//!
//! 该模块不包含任何业务逻辑实现，仅定义类型与契约。

pub mod error;
pub mod frame;
pub mod generator;
pub mod params;
pub mod registry;

pub use error::*;
pub use frame::*;
pub use generator::*;
pub use params::*;
pub use registry::*;

use serde::de::DeserializeOwned;
use serde_json::Value;
use std::collections::HashMap;

/// 从扩展字段反序列化生成器/处理器专用参数
///
/// 当 extensions 为空时返回 T 的默认值，否则将 extensions 序列化后再反序列化为 T。
/// 所有工厂函数应使用此辅助函数以避免重复的反序列化样板代码。
///
/// # Errors
/// 当反序列化失败时返回 [`CoreError::SerializationError`]
pub fn deserialize_extensions<T: Default + DeserializeOwned>(
    extensions: &HashMap<String, Value>,
) -> CoreResult<T> {
    if extensions.is_empty() {
        Ok(T::default())
    } else {
        let obj = serde_json::to_value(extensions).map_err(|e| {
            CoreError::SerializationError(format!("failed to serialize extensions: {}", e))
        })?;
        serde_json::from_value(obj).map_err(|e| {
            CoreError::SerializationError(format!("failed to deserialize params: {}", e))
        })
    }
}
