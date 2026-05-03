//! 分片执行器
//!
//! 实现 execute_shard() 函数，串联完整的"生成→后处理→写出"数据流。
//! 包含容错机制，单个分片失败不影响其他分片。

use std::path::Path;
use std::panic::AssertUnwindSafe;

use serde_json::Value;

use crate::core::{CoreError, CoreResult, OutputFormat, SequenceFrame, GeneratorRegistry};
use crate::sink::{OutputConfig, OutputStats};
use crate::pipeline::ProcessorRegistry;

use super::manifest::TaskSpec;
use super::shard::{Shard, ShardResult};

/// 执行单个分片：生成→后处理→写出
///
/// # Arguments
/// * `shard` - 分片描述
/// * `task` - 所属任务配置
/// * `gen_registry` - 生成器注册表
/// * `proc_registry` - 处理器注册表
/// * `adapter_factory` - 输出适配器工厂函数
/// * `output_dir` - 输出根目录
/// * `output_config` - 输出配置
///
/// # Returns
/// ShardResult，包含输出统计或错误信息
pub fn execute_shard(
    shard: &Shard,
    task: &TaskSpec,
    gen_registry: &GeneratorRegistry,
    proc_registry: &ProcessorRegistry,
    adapter_factory: crate::sink::SinkAdapterFactory,
    output_dir: &Path,
    output_config: &OutputConfig,
) -> ShardResult {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        execute_shard_inner(shard, task, gen_registry, proc_registry, adapter_factory, output_dir, output_config)
    }));

    match result {
        Ok(Ok(shard_result)) => shard_result,
        Ok(Err(e)) => ShardResult {
            task_name: task.name.clone(),
            shard_idx: shard.shard_idx,
            seed: shard.seed,
            sample_count: shard.sample_count,
            format: task.output_format.unwrap_or(OutputFormat::Parquet),
            stats: OutputStats::default(),
            error: Some(e.to_string()),
        },
        Err(panic_payload) => {
            let msg = if let Some(s) = panic_payload.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic_payload.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "unknown panic".to_string()
            };
            ShardResult {
                task_name: task.name.clone(),
                shard_idx: shard.shard_idx,
                seed: shard.seed,
                sample_count: shard.sample_count,
                format: task.output_format.unwrap_or(OutputFormat::Parquet),
                stats: OutputStats::default(),
                error: Some(format!("panic: {}", msg)),
            }
        }
    }
}

/// execute_shard 的内部实现，可能 panic
fn execute_shard_inner(
    shard: &Shard,
    task: &TaskSpec,
    gen_registry: &GeneratorRegistry,
    proc_registry: &ProcessorRegistry,
    adapter_factory: crate::sink::SinkAdapterFactory,
    output_dir: &Path,
    output_config: &OutputConfig,
) -> CoreResult<ShardResult> {
    // 1. 实例化生成器
    let generator = gen_registry.instantiate(&task.generator, &task.params.extensions)?;

    // 2. 生成原始帧流
    let raw_stream = generator.generate_stream(shard.seed, &task.params)?;

    // 3. 构建后处理管道链
    let processed_stream = build_pipeline_chain(raw_stream, task, proc_registry)?;

    // 4. 确定输出格式
    let format = task.output_format.unwrap_or(OutputFormat::Parquet);

    // 5. 创建并初始化输出适配器
    let mut adapter = adapter_factory(format)?;
    adapter.open(output_dir, &task.name, shard.shard_idx, shard.seed, output_config)?;

    // 6. 逐帧写入
    for frame in processed_stream {
        adapter.write_frame(&frame)?;
    }

    // 7. 关闭适配器，获取输出统计
    let stats = adapter.close()?;

    Ok(ShardResult {
        task_name: task.name.clone(),
        shard_idx: shard.shard_idx,
        seed: shard.seed,
        sample_count: shard.sample_count,
        format,
        stats,
        error: None,
    })
}

/// 构建后处理管道链
///
/// 按 task.pipeline 中的名称顺序，将各处理器串联为链式迭代器。
fn build_pipeline_chain(
    mut stream: Box<dyn Iterator<Item = SequenceFrame> + Send>,
    task: &TaskSpec,
    proc_registry: &ProcessorRegistry,
) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
    for proc_name in &task.pipeline {
        let config = task
            .params
            .extensions
            .get("pipeline_config")
            .and_then(|c| c.get(proc_name))
            .cloned()
            .unwrap_or(Value::Null);

        let processor = proc_registry.get(proc_name, &config)?;
        stream = processor.process(stream)?;
    }
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::GenParams;
    use crate::generators;
    use crate::sink::SinkAdapter;
    use crate::pipeline;
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// Mock 输出适配器，将帧收集到内存中用于测试验证
    struct MockAdapter {
        frames: Mutex<Vec<SequenceFrame>>,
    }

    impl MockAdapter {
        fn new() -> Self {
            MockAdapter {
                frames: Mutex::new(Vec::new()),
            }
        }
    }

    impl SinkAdapter for MockAdapter {
        fn format(&self) -> OutputFormat {
            OutputFormat::Text
        }

        fn open(&mut self, _base_dir: &Path, _task_name: &str, _shard_id: usize, _seed: u64, _config: &OutputConfig) -> CoreResult<()> {
            Ok(())
        }

        fn write_frame(&mut self, frame: &SequenceFrame) -> CoreResult<()> {
            self.frames.lock().unwrap().push(frame.clone());
            Ok(())
        }

        fn close(&mut self) -> CoreResult<OutputStats> {
            let count = self.frames.lock().unwrap().len() as u64;
            Ok(OutputStats {
                frames_written: count,
                bytes_written: 0,
                output_path: Some(PathBuf::from("/mock/output.txt")),
                file_hash: None,
            })
        }
    }

    fn mock_adapter_factory(format: OutputFormat) -> CoreResult<Box<dyn SinkAdapter>> {
        match format {
            OutputFormat::Text => Ok(Box::new(MockAdapter::new())),
            _ => Err(CoreError::ConfigError("仅支持 Text 格式的 mock 适配器".into())),
        }
    }

    fn make_test_registries() -> (GeneratorRegistry, ProcessorRegistry) {
        let mut gen_reg = GeneratorRegistry::new();
        generators::register_all(&mut gen_reg).unwrap();

        let mut proc_reg = ProcessorRegistry::new();
        pipeline::register_all(&mut proc_reg).unwrap();

        (gen_reg, proc_reg)
    }

    #[test]
    fn test_execute_shard_end_to_end() {
        let (gen_reg, proc_reg) = make_test_registries();

        let task = TaskSpec {
            name: "e2e_test".into(),
            generator: "ca".into(),
            params: GenParams::simple(10),
            count: 10,
            seed: 42,
            pipeline: vec![],
            output_format: Some(OutputFormat::Text),
            shard_size: None,
        };

        let shard = Shard {
            task_idx: 0,
            shard_idx: 0,
            seed: 42,
            sample_count: 10,
        };

        let result = execute_shard(
            &shard,
            &task,
            &gen_reg,
            &proc_reg,
            mock_adapter_factory,
            Path::new("/mock"),
            &OutputConfig::default(),
        );

        assert!(result.error.is_none(), "不应有错误: {:?}", result.error);
        assert_eq!(result.stats.frames_written, 10);
        assert_eq!(result.task_name, "e2e_test");
    }

    #[test]
    fn test_execute_shard_with_pipeline() {
        let (gen_reg, proc_reg) = make_test_registries();

        let task = TaskSpec {
            name: "pipeline_test".into(),
            generator: "ca".into(),
            params: GenParams::simple(10),
            count: 10,
            seed: 42,
            pipeline: vec!["null".into()],
            output_format: Some(OutputFormat::Text),
            shard_size: None,
        };

        let shard = Shard {
            task_idx: 0,
            shard_idx: 0,
            seed: 42,
            sample_count: 10,
        };

        let result = execute_shard(
            &shard,
            &task,
            &gen_reg,
            &proc_reg,
            mock_adapter_factory,
            Path::new("/mock"),
            &OutputConfig::default(),
        );

        assert!(result.error.is_none());
        assert_eq!(result.stats.frames_written, 10);
    }

    #[test]
    fn test_execute_shard_unknown_generator_returns_error() {
        let (gen_reg, proc_reg) = make_test_registries();

        let task = TaskSpec {
            name: "bad_gen".into(),
            generator: "nonexistent".into(),
            params: GenParams::simple(10),
            count: 10,
            seed: 42,
            pipeline: vec![],
            output_format: Some(OutputFormat::Text),
            shard_size: None,
        };

        let shard = Shard {
            task_idx: 0,
            shard_idx: 0,
            seed: 42,
            sample_count: 10,
        };

        let result = execute_shard(
            &shard,
            &task,
            &gen_reg,
            &proc_reg,
            mock_adapter_factory,
            Path::new("/mock"),
            &OutputConfig::default(),
        );

        assert!(result.error.is_some(), "应记录错误信息");
    }

    /// 注入 panic 的生成器，验证容错隔离
    struct PanicGenerator;

    impl crate::core::Generator for PanicGenerator {
        fn name(&self) -> &'static str {
            "panic_gen"
        }

        fn generate_stream(
            &self,
            _seed: u64,
            _params: &GenParams,
        ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
            panic!("故意的 panic 用于测试容错");
        }
    }

    #[test]
    fn test_execute_shard_panicking_generator_isolated() {
        let mut gen_reg = GeneratorRegistry::new();
        gen_reg.register("panic_gen", |_: &std::collections::HashMap<String, Value>| -> CoreResult<Box<dyn crate::core::Generator>> {
            Ok(Box::new(PanicGenerator))
        }).unwrap();
        let mut proc_reg = ProcessorRegistry::new();
        pipeline::register_all(&mut proc_reg).unwrap();

        let task = TaskSpec {
            name: "panic_test".into(),
            generator: "panic_gen".into(),
            params: GenParams::simple(10),
            count: 10,
            seed: 42,
            pipeline: vec![],
            output_format: Some(OutputFormat::Text),
            shard_size: None,
        };

        let shard = Shard {
            task_idx: 0,
            shard_idx: 0,
            seed: 42,
            sample_count: 10,
        };

        let result = execute_shard(
            &shard,
            &task,
            &gen_reg,
            &proc_reg,
            mock_adapter_factory,
            Path::new("/mock"),
            &OutputConfig::default(),
        );

        assert!(result.error.is_some(), "panic 应被捕获为错误");
        assert!(result.error.unwrap().contains("panic"), "错误信息应包含 panic");
        assert_eq!(result.stats.frames_written, 0, "panic 后不应有帧被写入");
    }
}
