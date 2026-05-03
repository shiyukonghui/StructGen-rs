use std::collections::HashMap;

use serde_json::Value;

use crate::core::{CoreError, CoreResult};

use super::processor::{Processor, ProcessorFactory};

/// 处理器注册表。维护名称→构造函数的映射，支持按名称查找和实例化处理器
#[derive(Default)]
pub struct ProcessorRegistry {
    factories: HashMap<&'static str, ProcessorFactory>,
}

impl ProcessorRegistry {
    /// 创建空的处理器注册表
    pub fn new() -> Self {
        ProcessorRegistry {
            factories: HashMap::new(),
        }
    }

    /// 注册一个处理器构造函数
    ///
    /// # Errors
    /// 当名称已存在时返回 [`CoreError::InvalidParams`]
    pub fn register(&mut self, name: &'static str, factory: ProcessorFactory) -> CoreResult<()> {
        if self.factories.contains_key(name) {
            return Err(CoreError::InvalidParams(format!(
                "处理器 '{}' 已经注册过了",
                name
            )));
        }
        self.factories.insert(name, factory);
        Ok(())
    }

    /// 按名称实例化一个处理器
    ///
    /// # Arguments
    /// * `name` - 处理器名称（如 "normalizer"）
    /// * `config` - 处理器配置的 JSON 值，可为 `Value::Null` 使用默认配置
    ///
    /// # Errors
    /// 当名称未注册时返回 [`CoreError::PipelineError`]
    pub fn get(&self, name: &str, config: &Value) -> CoreResult<Box<dyn Processor>> {
        let factory = self.factories.get(name).ok_or_else(|| {
            CoreError::PipelineError(format!("未找到处理器: {}", name))
        })?;
        factory(config)
    }

    /// 列出所有已注册的处理器名称
    pub fn list_names(&self) -> Vec<&'static str> {
        self.factories.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SequenceFrame;
    use serde_json::json;

    struct MockProcessor;

    impl Processor for MockProcessor {
        fn name(&self) -> &'static str {
            "mock"
        }

        fn process(
            &self,
            input: Box<dyn Iterator<Item = SequenceFrame> + Send>,
        ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
            Ok(input)
        }
    }

    fn mock_factory(_config: &Value) -> CoreResult<Box<dyn Processor>> {
        Ok(Box::new(MockProcessor))
    }

    #[test]
    fn test_registry_new_is_empty() {
        let reg = ProcessorRegistry::new();
        assert!(reg.list_names().is_empty());
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = ProcessorRegistry::new();
        reg.register("mock", mock_factory).unwrap();
        let processor = reg.get("mock", &json!(null)).unwrap();
        assert_eq!(processor.name(), "mock");
    }

    #[test]
    fn test_registry_get_not_found() {
        let reg = ProcessorRegistry::new();
        let result = reg.get("nonexistent", &json!(null));
        assert!(result.is_err());
        match result {
            Err(CoreError::PipelineError(msg)) => assert!(msg.contains("未找到处理器")),
            _ => panic!("expected PipelineError"),
        }
    }

    #[test]
    fn test_registry_duplicate_register() {
        let mut reg = ProcessorRegistry::new();
        reg.register("mock", mock_factory).unwrap();
        let result = reg.register("mock", mock_factory);
        assert!(result.is_err());
        match result {
            Err(CoreError::InvalidParams(msg)) => assert!(msg.contains("已经注册")),
            _ => panic!("expected InvalidParams"),
        }
    }

    #[test]
    fn test_registry_list_names() {
        let mut reg = ProcessorRegistry::new();
        reg.register("proc_a", mock_factory).unwrap();
        reg.register("proc_b", mock_factory).unwrap();
        let names = reg.list_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"proc_a"));
        assert!(names.contains(&"proc_b"));
    }
}
