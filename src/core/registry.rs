use std::collections::HashMap;

use serde_json::Value;

use super::error::{CoreError, CoreResult};
use super::generator::Generator;

/// 生成器构造函数类型：接受 extensions 映射，返回 Generator trait 对象
pub type GeneratorFactory = fn(&HashMap<String, Value>) -> CoreResult<Box<dyn Generator>>;

/// 生成器的全局注册表。采用名称→构造函数的静态映射。
///
/// 所有生成器在程序启动时通过 register 方法注册自身。
/// 调度器通过 instantiate 方法按名称查找构造函数并实例化生成器。
#[derive(Default)]
pub struct GeneratorRegistry {
    factories: HashMap<&'static str, GeneratorFactory>,
}

impl GeneratorRegistry {
    /// 创建空的注册表
    pub fn new() -> Self {
        GeneratorRegistry {
            factories: HashMap::new(),
        }
    }

    /// 注册一个生成器构造函数
    ///
    /// # Errors
    /// 当名称已存在时返回 [`CoreError::InvalidParams`]
    pub fn register(&mut self, name: &'static str, factory: GeneratorFactory) -> CoreResult<()> {
        if self.factories.contains_key(name) {
            return Err(CoreError::InvalidParams(format!(
                "generator '{}' is already registered",
                name
            )));
        }
        self.factories.insert(name, factory);
        Ok(())
    }

    /// 按名称实例化一个生成器
    ///
    /// # Errors
    /// 当名称未注册时返回 [`CoreError::GeneratorNotFound`]
    pub fn instantiate(
        &self,
        name: &str,
        extensions: &HashMap<String, Value>,
    ) -> CoreResult<Box<dyn Generator>> {
        let factory = self
            .factories
            .get(name)
            .ok_or_else(|| CoreError::GeneratorNotFound(name.to_string()))?;
        factory(extensions)
    }

    /// 列出所有已注册的生成器名称
    pub fn list_names(&self) -> Vec<&'static str> {
        self.factories.keys().copied().collect()
    }

    /// 判断某个名称是否已注册
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用 mock 生成器工厂
    fn mock_factory(extensions: &HashMap<String, Value>) -> CoreResult<Box<dyn Generator>> {
        let _ = extensions;
        Ok(Box::new(MockGen))
    }

    struct MockGen;

    impl Generator for MockGen {
        fn name(&self) -> &'static str {
            "mock"
        }

        fn generate_stream(
            &self,
            _seed: u64,
            _params: &super::super::params::GenParams,
        ) -> CoreResult<Box<dyn Iterator<Item = super::super::frame::SequenceFrame> + Send>> {
            Ok(Box::new(std::iter::empty()))
        }
    }

    #[test]
    fn test_registry_register_and_instantiate() {
        let mut reg = GeneratorRegistry::new();
        reg.register("mock", mock_factory).unwrap();

        assert!(reg.contains("mock"));
        let result = reg.instantiate("mock", &HashMap::new());
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "mock");
    }

    #[test]
    fn test_registry_name_not_found() {
        let reg = GeneratorRegistry::new();
        let result = reg.instantiate("nonexistent", &HashMap::new());
        assert!(result.is_err());
        match result {
            Err(CoreError::GeneratorNotFound(name)) => assert_eq!(name, "nonexistent"),
            _ => panic!("expected GeneratorNotFound error"),
        }
    }

    #[test]
    fn test_registry_duplicate_returns_error() {
        let mut reg = GeneratorRegistry::new();
        reg.register("mock", mock_factory).unwrap();
        let result = reg.register("mock", mock_factory);
        assert!(result.is_err());
        match result {
            Err(CoreError::InvalidParams(msg)) => {
                assert!(msg.contains("already registered"));
            }
            _ => panic!("expected InvalidParams error"),
        }
    }

    #[test]
    fn test_registry_list_names() {
        let mut reg = GeneratorRegistry::new();
        reg.register("ca", mock_factory).unwrap();
        reg.register("lorenz", mock_factory).unwrap();

        let names = reg.list_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"ca"));
        assert!(names.contains(&"lorenz"));
    }

    #[test]
    fn test_registry_default() {
        let reg = GeneratorRegistry::default();
        assert!(reg.list_names().is_empty());
    }

    #[test]
    fn test_registry_factory_error() {
        fn error_factory(_extensions: &HashMap<String, Value>) -> CoreResult<Box<dyn Generator>> {
            Err(CoreError::GeneratorInitError("intentional failure".into()))
        }

        let mut reg = GeneratorRegistry::new();
        reg.register("error_gen", error_factory).unwrap();
        let result = reg.instantiate("error_gen", &HashMap::new());
        assert!(result.is_err());
        let err = result.err().unwrap();
        match err {
            CoreError::GeneratorInitError(msg) => assert_eq!(msg, "intentional failure"),
            _ => panic!("expected GeneratorInitError"),
        }
    }
}
