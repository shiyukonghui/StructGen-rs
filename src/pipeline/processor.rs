use crate::core::{CoreResult, SequenceFrame};

/// 后处理器的抽象接口。每个处理器是一个迭代器适配器：
/// 接受帧流，返回变换后的帧流。
/// 实现者必须为 Send + Sync，保证在多线程环境中安全使用。
pub trait Processor: Send + Sync {
    /// 返回该处理器的唯一标识名称
    fn name(&self) -> &'static str;

    /// 对输入帧流应用变换，返回变换后的帧流
    ///
    /// # Arguments
    /// * `input` - 原始帧迭代器。消费此迭代器产生的所有帧。
    ///
    /// # Returns
    /// 变换后的惰性帧迭代器。消费返回的迭代器才会驱动处理。
    fn process(
        &self,
        input: Box<dyn Iterator<Item = SequenceFrame> + Send>,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>>;
}

/// 处理器构造函数类型：接受配置 JSON，返回 Processor trait 对象
pub type ProcessorFactory = fn(&serde_json::Value) -> CoreResult<Box<dyn Processor>>;
