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
