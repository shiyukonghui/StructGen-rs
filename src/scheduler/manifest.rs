//! 清单数据结构定义与 YAML 解析
//!
//! Manifest 是用户通过 YAML 文件描述整个生成任务的顶层结构。
//! TaskSpec 是单个生成任务的详细配置。

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::core::{CoreError, CoreResult, GenParams, GlobalConfig, OutputFormat};
use crate::core::registry::GeneratorRegistry;
use crate::pipeline::ProcessorRegistry;

/// 完整清单，包含任务列表和全局配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// 任务列表
    pub tasks: Vec<TaskSpec>,
    /// 全局配置（输出目录、线程数等），可被命令行参数覆盖
    #[serde(default)]
    pub global: GlobalConfig,
}

/// 单个任务的描述
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpec {
    /// 任务名称（用于元数据和日志标识）
    pub name: String,
    /// 生成器名称，必须已在 GeneratorRegistry 中注册
    pub generator: String,
    /// 通用参数（序列长度、网格、扩展字段）
    #[serde(default)]
    pub params: GenParams,
    /// 要生成的样本数量
    pub count: usize,
    /// 基础随机种子
    pub seed: u64,
    /// 后处理管道中的处理器名称列表，按顺序应用。空列表表示不进行任何后处理
    #[serde(default)]
    pub pipeline: Vec<String>,
    /// 输出格式，若未指定则使用全局默认值
    #[serde(default)]
    pub output_format: Option<OutputFormat>,
    /// 每个分片包含的最大样本数，若未指定则自动计算
    #[serde(default)]
    pub shard_size: Option<usize>,
}

impl Manifest {
    /// 从 YAML 字符串解析清单
    ///
    /// # Errors
    /// 当 YAML 格式不正确时返回 [`CoreError::ManifestError`]
    pub fn from_yaml(yaml_str: &str) -> CoreResult<Self> {
        serde_yaml::from_str(yaml_str).map_err(|e| {
            CoreError::ManifestError(format!("YAML 解析失败: {}", e))
        })
    }

    /// 校验清单的合法性
    ///
    /// # Arguments
    /// * `gen_registry` - 生成器注册表，用于校验生成器名称是否存在
    /// * `proc_registry` - 处理器注册表，用于校验管道处理器名称是否存在
    ///
    /// # Errors
    /// 当校验不通过时返回对应错误
    pub fn validate(
        &self,
        gen_registry: &GeneratorRegistry,
        proc_registry: &ProcessorRegistry,
    ) -> CoreResult<()> {
        // 1. 校验全局配置
        self.global.validate()?;

        // 2. 校验任务列表非空
        if self.tasks.is_empty() {
            return Err(CoreError::ManifestError("任务列表不能为空".into()));
        }

        // 3. 校验任务名称唯一
        let mut names = HashSet::new();
        for task in &self.tasks {
            if !names.insert(&task.name) {
                return Err(CoreError::ManifestError(format!(
                    "任务名称重复: {}",
                    task.name
                )));
            }
        }

        // 4. 逐个校验任务
        for task in &self.tasks {
            task.validate(gen_registry, proc_registry)?;
        }

        Ok(())
    }
}

impl TaskSpec {
    /// 校验单个任务的合法性
    fn validate(
        &self,
        gen_registry: &GeneratorRegistry,
        proc_registry: &ProcessorRegistry,
    ) -> CoreResult<()> {
        // 样本数量必须 > 0
        if self.count == 0 {
            return Err(CoreError::ManifestError(format!(
                "任务 '{}' 的样本数量不能为 0",
                self.name
            )));
        }

        // 生成器名称必须在注册表中
        if !gen_registry.contains(&self.generator) {
            return Err(CoreError::ManifestError(format!(
                "任务 '{}' 引用了未注册的生成器: {}",
                self.name, self.generator
            )));
        }

        // 处理器名称必须在注册表中
        for proc_name in &self.pipeline {
            if !proc_registry.list_names().contains(&proc_name.as_str()) {
                return Err(CoreError::ManifestError(format!(
                    "任务 '{}' 引用了未注册的处理器: {}",
                    self.name, proc_name
                )));
            }
        }

        // 分片大小如果指定则 > 0
        if let Some(shard_size) = self.shard_size {
            if shard_size == 0 {
                return Err(CoreError::ManifestError(format!(
                    "任务 '{}' 的分片大小不能为 0",
                    self.name
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators;
    use crate::pipeline;
    use crate::core::params::OutputFormat;

    fn make_test_registries() -> (GeneratorRegistry, ProcessorRegistry) {
        let mut gen_reg = GeneratorRegistry::new();
        generators::register_all(&mut gen_reg).unwrap();

        let mut proc_reg = ProcessorRegistry::new();
        pipeline::register_all(&mut proc_reg).unwrap();

        (gen_reg, proc_reg)
    }

    fn make_valid_yaml() -> &'static str {
        r#"
global:
  output_dir: "./output"
  default_format: "Parquet"
  log_level: "info"
  num_threads: 4

tasks:
  - name: "rule30_ca"
    generator: "cellular_automaton"
    params:
      seq_length: 1000
      extensions:
        ca:
          rule: 30
    count: 100
    seed: 12345
    pipeline: ["normalizer"]
    output_format: "Text"

  - name: "lorenz_chaos"
    generator: "lorenz_system"
    params:
      seq_length: 2000
      extensions:
        lorenz:
          sigma: 10.0
          rho: 28.0
          beta: 2.667
    count: 200
    seed: 67890
    pipeline: ["normalizer", "token_mapper"]
    output_format: "Parquet"
"#
    }

    #[test]
    fn test_valid_yaml_parsed_successfully() {
        let manifest = Manifest::from_yaml(make_valid_yaml()).unwrap();
        assert_eq!(manifest.tasks.len(), 2);
        assert_eq!(manifest.tasks[0].name, "rule30_ca");
        assert_eq!(manifest.tasks[0].generator, "cellular_automaton");
        assert_eq!(manifest.tasks[0].count, 100);
        assert_eq!(manifest.tasks[0].seed, 12345);
        assert_eq!(manifest.tasks[0].pipeline, vec!["normalizer"]);
        assert_eq!(manifest.tasks[0].output_format, Some(OutputFormat::Text));
        assert!(manifest.tasks[0].shard_size.is_none());

        assert_eq!(manifest.tasks[1].name, "lorenz_chaos");
        assert_eq!(manifest.tasks[1].generator, "lorenz_system");
        assert_eq!(manifest.tasks[1].count, 200);
        assert_eq!(manifest.tasks[1].seed, 67890);
        assert_eq!(
            manifest.tasks[1].pipeline,
            vec!["normalizer", "token_mapper"]
        );
        assert_eq!(manifest.tasks[1].output_format, Some(OutputFormat::Parquet));

        assert_eq!(manifest.global.output_dir, "./output");
        assert_eq!(manifest.global.num_threads, Some(4));
    }

    #[test]
    fn test_valid_manifest_passes_validation() {
        let manifest = Manifest::from_yaml(make_valid_yaml()).unwrap();
        let (gen_reg, proc_reg) = make_test_registries();
        assert!(manifest.validate(&gen_reg, &proc_reg).is_ok());
    }

    #[test]
    fn test_manifest_rejects_zero_count() {
        let yaml = r#"
global:
  output_dir: "./output"
tasks:
  - name: "bad_task"
    generator: "ca"
    count: 0
    seed: 42
"#;
        let manifest = Manifest::from_yaml(yaml).unwrap();
        let (gen_reg, proc_reg) = make_test_registries();
        let result = manifest.validate(&gen_reg, &proc_reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            CoreError::ManifestError(msg) => {
                assert!(msg.contains("样本数量不能为 0"));
                assert!(msg.contains("bad_task"));
            }
            _ => panic!("expected ManifestError, got {:?}", err),
        }
    }

    #[test]
    fn test_manifest_rejects_empty_output_dir() {
        let yaml = r#"
global:
  output_dir: ""
tasks:
  - name: "test_task"
    generator: "ca"
    count: 10
    seed: 42
"#;
        let manifest = Manifest::from_yaml(yaml).unwrap();
        let (gen_reg, proc_reg) = make_test_registries();
        let result = manifest.validate(&gen_reg, &proc_reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            CoreError::ConfigError(msg) => {
                assert!(msg.contains("output_dir"));
            }
            _ => panic!("expected ConfigError, got {:?}", err),
        }
    }

    #[test]
    fn test_manifest_rejects_unknown_generator() {
        let yaml = r#"
global:
  output_dir: "./output"
tasks:
  - name: "bad_gen"
    generator: "nonexistent_generator"
    count: 10
    seed: 42
"#;
        let manifest = Manifest::from_yaml(yaml).unwrap();
        let (gen_reg, proc_reg) = make_test_registries();
        let result = manifest.validate(&gen_reg, &proc_reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            CoreError::ManifestError(msg) => {
                assert!(msg.contains("未注册的生成器"));
                assert!(msg.contains("nonexistent_generator"));
            }
            _ => panic!("expected ManifestError, got {:?}", err),
        }
    }

    #[test]
    fn test_manifest_rejects_unknown_processor() {
        let yaml = r#"
global:
  output_dir: "./output"
tasks:
  - name: "bad_proc"
    generator: "ca"
    count: 10
    seed: 42
    pipeline: ["nonexistent_processor"]
"#;
        let manifest = Manifest::from_yaml(yaml).unwrap();
        let (gen_reg, proc_reg) = make_test_registries();
        let result = manifest.validate(&gen_reg, &proc_reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            CoreError::ManifestError(msg) => {
                assert!(msg.contains("未注册的处理器"));
                assert!(msg.contains("nonexistent_processor"));
            }
            _ => panic!("expected ManifestError, got {:?}", err),
        }
    }

    #[test]
    fn test_manifest_rejects_duplicate_task_names() {
        let yaml = r#"
global:
  output_dir: "./output"
tasks:
  - name: "same_name"
    generator: "ca"
    count: 10
    seed: 1
  - name: "same_name"
    generator: "lorenz"
    count: 10
    seed: 2
"#;
        let manifest = Manifest::from_yaml(yaml).unwrap();
        let (gen_reg, proc_reg) = make_test_registries();
        let result = manifest.validate(&gen_reg, &proc_reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            CoreError::ManifestError(msg) => {
                assert!(msg.contains("任务名称重复"));
            }
            _ => panic!("expected ManifestError, got {:?}", err),
        }
    }

    #[test]
    fn test_manifest_rejects_zero_shard_size() {
        let yaml = r#"
global:
  output_dir: "./output"
tasks:
  - name: "bad_shard"
    generator: "ca"
    count: 10
    seed: 42
    shard_size: 0
"#;
        let manifest = Manifest::from_yaml(yaml).unwrap();
        let (gen_reg, proc_reg) = make_test_registries();
        let result = manifest.validate(&gen_reg, &proc_reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            CoreError::ManifestError(msg) => {
                assert!(msg.contains("分片大小不能为 0"));
            }
            _ => panic!("expected ManifestError, got {:?}", err),
        }
    }

    #[test]
    fn test_manifest_default_fields() {
        let yaml = r#"
global:
  output_dir: "./output"
tasks:
  - name: "minimal"
    generator: "ca"
    count: 5
    seed: 1
"#;
        let manifest = Manifest::from_yaml(yaml).unwrap();
        assert_eq!(manifest.tasks[0].pipeline, Vec::<String>::new());
        assert_eq!(manifest.tasks[0].output_format, None);
        assert_eq!(manifest.tasks[0].shard_size, None);
    }

    #[test]
    fn test_invalid_yaml_rejected() {
        let yaml = "this is not valid yaml: [[[";
        let result = Manifest::from_yaml(yaml);
        assert!(result.is_err());
    }
}
