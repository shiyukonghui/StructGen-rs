//! StructGen-rs 任务调度层 (scheduler)
//!
//! 本模块是系统的协调中枢，负责将 YAML 清单转换为可执行的生成任务，
//! 管理确定性的种子派生，并通过 rayon 线程池并行执行所有分片任务。
//!
//! 协调生成器→后处理管道→输出适配器之间的完整数据流动。

pub mod seed;
pub mod manifest;
pub mod shard;
pub mod executor;

use std::path::Path;

use rayon::prelude::*;

use crate::core::{CoreError, CoreResult, GeneratorRegistry};
use crate::pipeline::ProcessorRegistry;
use crate::sink::{OutputConfig, SinkAdapterFactory};

use self::executor::execute_shard;
use self::manifest::Manifest;
use self::shard::{shard_tasks, ShardResult};

/// 运行一个完整的清单
///
/// 这是 scheduler 模块对外暴露的唯一入口函数。
///
/// # Arguments
/// * `manifest` - 已解析并校验的清单
/// * `gen_registry` - 生成器注册表
/// * `proc_registry` - 处理器注册表
/// * `adapter_factory` - 输出适配器工厂函数
///
/// # Returns
/// 所有分片的执行结果列表
///
/// # Errors
/// 当分片任务列表为空时返回 ManifestError
pub fn run_manifest(
    manifest: &Manifest,
    gen_registry: &GeneratorRegistry,
    proc_registry: &ProcessorRegistry,
    adapter_factory: SinkAdapterFactory,
) -> CoreResult<Vec<ShardResult>> {
    // 1. 切分任务为分片
    let shards = shard_tasks(manifest);
    if shards.is_empty() {
        return Err(CoreError::ManifestError("没有可执行的分片任务".into()));
    }

    // 2. 配置输出
    let output_dir = Path::new(&manifest.global.output_dir);
    let output_config = OutputConfig::default();

    // 3. 并行执行所有分片
    let results: Vec<ShardResult> = shards
        .par_iter()
        .map(|shard| {
            let task = &manifest.tasks[shard.task_idx];
            execute_shard(
                shard,
                task,
                gen_registry,
                proc_registry,
                adapter_factory,
                output_dir,
                &output_config,
            )
        })
        .collect();

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{FrameData, GenParams, OutputFormat, SequenceFrame};
    use crate::sink::SinkAdapter;
    use crate::generators;
    use crate::pipeline;
    use serde_json::Value;
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// 测试用 Mock 适配器
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

        fn close(&mut self) -> CoreResult<crate::sink::OutputStats> {
            let count = self.frames.lock().unwrap().len() as u64;
            Ok(crate::sink::OutputStats {
                frames_written: count,
                bytes_written: 0,
                output_path: Some(PathBuf::from("/mock/result.txt")),
                file_hash: None,
            })
        }
    }

    fn mock_adapter_factory(format: OutputFormat, _config: &Value) -> CoreResult<Box<dyn SinkAdapter>> {
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
    fn test_run_manifest_single_shard() {
        let (gen_reg, proc_reg) = make_test_registries();

        let manifest = Manifest {
            tasks: vec![manifest::TaskSpec {
                name: "single_task".into(),
                generator: "ca".into(),
                params: GenParams::simple(5),
                count: 5,
                seed: 42,
                pipeline: vec![],
                output_format: Some(OutputFormat::Text),
                shard_size: Some(5),
            }],
            global: crate::core::GlobalConfig {
                output_dir: "./test_output".into(),
                ..Default::default()
            },
        };

        let results = run_manifest(&manifest, &gen_reg, &proc_reg, mock_adapter_factory).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].error.is_none());
        // 5 个样本 × 每个 5 帧 = 25 帧
        assert_eq!(results[0].stats.frames_written, 25);
        assert_eq!(results[0].task_name, "single_task");
    }

    #[test]
    fn test_run_manifest_multi_shard() {
        let (gen_reg, proc_reg) = make_test_registries();

        let manifest = Manifest {
            tasks: vec![manifest::TaskSpec {
                name: "multi_shard".into(),
                generator: "ca".into(),
                params: GenParams::simple(5),
                count: 15,
                seed: 100,
                pipeline: vec![],
                output_format: Some(OutputFormat::Text),
                shard_size: Some(5),
            }],
            global: crate::core::GlobalConfig {
                output_dir: "./test_output".into(),
                ..Default::default()
            },
        };

        let results = run_manifest(&manifest, &gen_reg, &proc_reg, mock_adapter_factory).unwrap();

        assert_eq!(results.len(), 3);

        let total_frames: u64 = results.iter().map(|r| r.stats.frames_written).sum();
        // 3 个分片 × 5 个样本 × 5 帧 = 75 帧
        assert_eq!(total_frames, 75);

        for result in &results {
            assert!(result.error.is_none(), "分片不应有错误");
        }
    }

    #[test]
    fn test_run_manifest_deterministic() {
        let (gen_reg, proc_reg) = make_test_registries();

        let manifest = Manifest {
            tasks: vec![manifest::TaskSpec {
                name: "det_test".into(),
                generator: "ca".into(),
                params: GenParams::simple(5),
                count: 10,
                seed: 12345,
                pipeline: vec![],
                output_format: Some(OutputFormat::Text),
                shard_size: Some(5),
            }],
            global: crate::core::GlobalConfig {
                output_dir: "./test_output".into(),
                ..Default::default()
            },
        };

        let results1 = run_manifest(&manifest, &gen_reg, &proc_reg, mock_adapter_factory).unwrap();
        let results2 = run_manifest(&manifest, &gen_reg, &proc_reg, mock_adapter_factory).unwrap();

        assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.stats.frames_written, r2.stats.frames_written);
            assert_eq!(r1.sample_count, r2.sample_count);
            assert_eq!(r1.seed, r2.seed);
            assert_eq!(r1.task_name, r2.task_name);
            assert_eq!(r1.error.is_some(), r2.error.is_some());
        }
    }

    #[test]
    fn test_run_manifest_fault_tolerance() {
        use std::collections::HashMap;

        // 创建一个包含一个会失败的生成器的注册表
        let mut gen_reg = GeneratorRegistry::new();

        // 注册一个正常的生成器
        gen_reg.register("good_gen", |_: &HashMap<String, Value>| -> CoreResult<Box<dyn crate::core::Generator>> {
            struct GoodGen;
            impl crate::core::Generator for GoodGen {
                fn name(&self) -> &'static str { "good_gen" }
                fn generate_stream(&self, _seed: u64, params: &GenParams) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
                    let len = if params.seq_length == 0 { 5 } else { params.seq_length };
                    Ok(Box::new((0..len).map(|i| SequenceFrame::new(i as u64, FrameData::new()))))
                }
            }
            Ok(Box::new(GoodGen))
        }).unwrap();

        // 注册一个会失败的生成器
        gen_reg.register("bad_gen", |_: &HashMap<String, Value>| -> CoreResult<Box<dyn crate::core::Generator>> {
            Err(CoreError::GeneratorInitError("故意的初始化失败".into()))
        }).unwrap();

        let mut proc_reg = ProcessorRegistry::new();
        pipeline::register_all(&mut proc_reg).unwrap();

        let manifest = Manifest {
            tasks: vec![
                manifest::TaskSpec {
                    name: "good_task".into(),
                    generator: "good_gen".into(),
                    params: GenParams::simple(5),
                    count: 5,
                    seed: 1,
                    pipeline: vec![],
                    output_format: Some(OutputFormat::Text),
                    shard_size: Some(5),
                },
                manifest::TaskSpec {
                    name: "bad_task".into(),
                    generator: "bad_gen".into(),
                    params: GenParams::simple(5),
                    count: 5,
                    seed: 2,
                    pipeline: vec![],
                    output_format: Some(OutputFormat::Text),
                    shard_size: Some(5),
                },
            ],
            global: crate::core::GlobalConfig {
                output_dir: "./test_output".into(),
                ..Default::default()
            },
        };

        let results = run_manifest(&manifest, &gen_reg, &proc_reg, mock_adapter_factory).unwrap();

        assert_eq!(results.len(), 2);

        // 好任务的分片应成功
        let good_result = results.iter().find(|r| r.task_name == "good_task").unwrap();
        assert!(good_result.error.is_none());
        // 5 个样本 × 每个 5 帧 = 25 帧
        assert_eq!(good_result.stats.frames_written, 25);

        // 坏任务的分片应记录错误
        let bad_result = results.iter().find(|r| r.task_name == "bad_task").unwrap();
        assert!(bad_result.error.is_some());
    }
}
