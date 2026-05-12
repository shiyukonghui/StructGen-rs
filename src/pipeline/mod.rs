//! StructGen-rs 后处理管道层，定义数据清洗与变换的处理器接口和内置处理器
//!
//! 该模块提供可级联的帧序列处理器，包括标准化、去重、差分编码、令牌映射等。

pub mod processor;
pub mod registry;
pub mod null_proc;
pub mod normalizer;
pub mod dedup;
pub mod diff_encoder;
pub mod token_mapper;
pub mod clip_stitcher;
pub mod patch_tokenizer;
pub mod sequence_stitcher;
pub mod batch_collector;

// 重导出公开类型
pub use processor::{Processor, ProcessorFactory};
pub use registry::ProcessorRegistry;
pub use null_proc::{NullProcessor, create_null_processor};
pub use normalizer::{Normalizer, NormalizerConfig, NormalizeMethod, create_normalizer};
pub use dedup::{DedupFilter, DedupConfig, create_dedup};
pub use diff_encoder::{DiffEncoder, DiffEncoderConfig, create_diff_encoder};
pub use token_mapper::{TokenMapper, TokenMapperConfig, create_token_mapper};
pub use clip_stitcher::{ClipStitcher, ClipStitcherConfig, create_clip_stitcher};
pub use patch_tokenizer::{PatchTokenizer, PatchTokenizerConfig, create_patch_tokenizer};
pub use sequence_stitcher::{SequenceStitcher, SequenceStitcherConfig, create_sequence_stitcher};
pub use batch_collector::{BatchCollector, BatchCollectorConfig, BatchData, BatchSample, create_batch_collector};

/// 向注册表中注册所有内置处理器
///
/// # 注册的处理器
/// - "null" — 透传处理器
/// - "normalizer" — 标准化器
/// - "dedup" — 去重过滤器
/// - "diff_encoder" — 差分编码器
/// - "token_mapper" — 令牌映射器
/// - "clip_stitcher" — 序列截断拼接器
/// - "patch_tokenizer" — Patch tokenization 处理器
/// - "sequence_stitcher" — 序列串联处理器
/// - "batch_collector" — 批量收集处理器
pub fn register_all(registry: &mut ProcessorRegistry) -> crate::core::CoreResult<()> {
    registry.register("null", create_null_processor)?;
    registry.register("normalizer", create_normalizer)?;
    registry.register("dedup", create_dedup)?;
    registry.register("diff_encoder", create_diff_encoder)?;
    registry.register("token_mapper", create_token_mapper)?;
    registry.register("clip_stitcher", create_clip_stitcher)?;
    registry.register("patch_tokenizer", create_patch_tokenizer)?;
    registry.register("sequence_stitcher", create_sequence_stitcher)?;
    registry.register("batch_collector", create_batch_collector)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{FrameData, FrameState, SequenceFrame};
    use serde_json::json;

    /// 构造测试用混合帧序列
    fn make_test_frames() -> Vec<SequenceFrame> {
        let make_frame = |step: u64, values: Vec<FrameState>| -> SequenceFrame {
            SequenceFrame::new(step, FrameData { values })
        };

        vec![
            make_frame(0, vec![FrameState::Float(0.0), FrameState::Bool(false)]),   // 全零帧
            make_frame(1, vec![FrameState::Float(0.25), FrameState::Bool(true)]),
            make_frame(2, vec![FrameState::Float(0.25), FrameState::Bool(true)]),  // 连续重复
            make_frame(3, vec![FrameState::Float(0.5), FrameState::Bool(false)]),
            make_frame(4, vec![FrameState::Float(1.0), FrameState::Bool(true)]),
            make_frame(5, vec![FrameState::Float(1.0), FrameState::Bool(true)]),  // 连续重复
        ]
    }

    #[test]
    fn test_register_all_builtins() {
        let mut registry = ProcessorRegistry::new();
        register_all(&mut registry).unwrap();

        let names = registry.list_names();
        assert!(names.contains(&"null"));
        assert!(names.contains(&"normalizer"));
        assert!(names.contains(&"dedup"));
        assert!(names.contains(&"diff_encoder"));
        assert!(names.contains(&"token_mapper"));
        assert!(names.contains(&"clip_stitcher"));
    }

    #[test]
    fn test_pipeline_chain_normalize_dedup_diff_token_map() {
        // 构建四个处理器链：normalize → dedup → diff → token_map
        let frames = make_test_frames();
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());

        // 1. 标准化
        let normalizer_config = json!({
            "max_val": 255,
            "method": "Linear"
        });
        let normalizer = create_normalizer(&normalizer_config).unwrap();
        let stream = normalizer.process(input).unwrap();

        // 2. 去重（移除全零帧和连续重复帧）
        let dedup_config = json!({
            "remove_consecutive_duplicates": true,
            "remove_all_zeros": true,
            "min_entropy": 0.0
        });
        let dedup = create_dedup(&dedup_config).unwrap();
        let stream = dedup.process(stream).unwrap();

        // 3. 差分编码
        let diff_config = json!({
            "prepend_zero_frame": false
        });
        let diff = create_diff_encoder(&diff_config).unwrap();
        let stream = diff.process(stream).unwrap();

        // 4. 令牌映射
        let token_config = json!({
            "start_codepoint": 0x4E00,
            "insert_newline": false
        });
        let token_mapper = create_token_mapper(&token_config).unwrap();
        let result: Vec<SequenceFrame> = token_mapper.process(stream).unwrap().collect();

        // 验证：应该有一系列输出帧
        // 原始6帧 → 去重移除全零帧(step0)和连续重复帧(step2, step5) → 3帧
        // 差分编码保持3帧 → 令牌映射
        assert!(!result.is_empty(), "链式处理后应有输出");

        // 验证所有输出码点在 Unicode 安全范围内
        for frame in &result {
            for state in &frame.state.values {
                match state {
                    FrameState::Integer(v) => {
                        assert!(*v >= 0, "码点不能为负: {}", v);
                        assert!(
                            *v as u64 <= 0x10_FFFF,
                            "码点超出 Unicode 范围: {}",
                            v
                        );
                    }
                    _ => {
                        // 标准化后所有值应已是 Integer，差分后也是 Integer
                        // 令牌映射后也全是 Integer
                    }
                }
            }
        }
    }

    #[test]
    fn test_pipeline_chain_empty_input() {
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(std::iter::empty());

        let normalizer = create_normalizer(&json!(null)).unwrap();
        let stream = normalizer.process(input).unwrap();
        let dedup = create_dedup(&json!(null)).unwrap();
        let stream = dedup.process(stream).unwrap();
        let diff = create_diff_encoder(&json!(null)).unwrap();
        let stream = diff.process(stream).unwrap();
        let token_mapper = create_token_mapper(&json!(null)).unwrap();
        let result: Vec<SequenceFrame> = token_mapper.process(stream).unwrap().collect();

        assert!(result.is_empty());
    }

    #[test]
    fn test_processors_instantiable_via_registry() {
        let mut registry = ProcessorRegistry::new();
        register_all(&mut registry).unwrap();

        // 验证所有处理器均可通过注册表实例化
        for name in registry.list_names() {
            let config = if name == "patch_tokenizer" {
                json!({
                    "patch": 2,
                    "num_colors": 10,
                    "rows": 4,
                    "cols": 4,
                    "n_groups": 1
                })
            } else if name == "batch_collector" {
                json!({
                    "batch_size": 2,
                    "num_frames": 3
                })
            } else if name == "sequence_stitcher" {
                json!({
                    "frames_per_sequence": 5,
                    "add_sequence_start": true,
                    "add_sequence_end": true,
                    "start_token": 10000,
                    "end_token": 10001
                })
            } else {
                json!(null)
            };
            let processor = registry.get(name, &config).unwrap();
            assert_eq!(processor.name(), name);
        }
    }
}
