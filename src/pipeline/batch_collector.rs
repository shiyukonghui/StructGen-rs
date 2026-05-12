//! 批量收集处理器
//!
//! 将流式帧收集为批量数据，用于生成训练数据批次。
//! 输出格式兼容 Python 训练流程的 (B, T, H, W, C) 张量格式。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::{CoreError, CoreResult, FrameData, FrameState, SequenceFrame};

use super::processor::Processor;

/// 批量收集处理器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCollectorConfig {
    /// 批量大小（样本数量）
    pub batch_size: usize,
    /// 每个样本的帧数
    pub num_frames: usize,
}

/// 单个样本数据
#[derive(Debug, Clone)]
pub struct BatchSample {
    /// 样本中的所有帧
    pub frames: Vec<SequenceFrame>,
}

impl BatchSample {
    /// 创建空样本
    pub fn new() -> Self {
        BatchSample { frames: Vec::new() }
    }

    /// 添加帧到样本
    pub fn push(&mut self, frame: SequenceFrame) {
        self.frames.push(frame);
    }

    /// 检查样本是否已满
    pub fn is_complete(&self, num_frames: usize) -> bool {
        self.frames.len() >= num_frames
    }
}

impl Default for BatchSample {
    fn default() -> Self {
        Self::new()
    }
}

/// 批量数据结构
#[derive(Debug, Clone)]
pub struct BatchData {
    /// 批量样本数据
    pub samples: Vec<BatchSample>,
}

impl BatchData {
    /// 创建空批量
    pub fn new() -> Self {
        BatchData { samples: Vec::new() }
    }

    /// 添加样本到批量
    pub fn push(&mut self, sample: BatchSample) {
        self.samples.push(sample);
    }

    /// 检查批量是否已满
    pub fn is_complete(&self, batch_size: usize) -> bool {
        self.samples.len() >= batch_size
    }

    /// 将批量数据序列化为 SequenceFrame
    /// 使用 label 字段标记为 "batch_data"，state.values 包含所有帧的数据
    pub fn to_sequence_frame(&self) -> SequenceFrame {
        // 计算总 token 数
        let total_tokens: usize = self.samples.iter()
            .map(|s| s.frames.iter().map(|f| f.state.values.len()).sum::<usize>())
            .sum();

        // 收集所有帧的数据
        let mut all_values = Vec::with_capacity(total_tokens);
        for sample in &self.samples {
            for frame in &sample.frames {
                all_values.extend(frame.state.values.clone());
            }
        }

        SequenceFrame {
            step_index: 0,
            state: FrameData { values: all_values },
            label: Some("batch_data".to_string()),
            sample_id: None,
        }
    }
}

impl Default for BatchData {
    fn default() -> Self {
        Self::new()
    }
}

/// 批量收集处理器
///
/// 将流式帧收集为批量数据，用于生成训练数据批次。
pub struct BatchCollector {
    config: BatchCollectorConfig,
}

impl BatchCollector {
    /// 根据配置创建 BatchCollector
    pub fn new(config: BatchCollectorConfig) -> CoreResult<Self> {
        if config.batch_size == 0 {
            return Err(CoreError::InvalidParams(
                "batch_collector: batch_size must be >= 1".into(),
            ));
        }
        if config.num_frames == 0 {
            return Err(CoreError::InvalidParams(
                "batch_collector: num_frames must be >= 1".into(),
            ));
        }
        Ok(BatchCollector { config })
    }
}

/// 批量收集迭代器适配器
struct BatchCollectIter<I: Iterator<Item = SequenceFrame>> {
    inner: I,
    batch_size: usize,
    num_frames: usize,
    /// 当前正在构建的批量
    current_batch: BatchData,
    /// 当前正在构建的样本
    current_sample: BatchSample,
    /// 是否已耗尽输入
    exhausted: bool,
}

impl<I: Iterator<Item = SequenceFrame>> BatchCollectIter<I> {
    fn new(inner: I, batch_size: usize, num_frames: usize) -> Self {
        BatchCollectIter {
            inner,
            batch_size,
            num_frames,
            current_batch: BatchData::new(),
            current_sample: BatchSample::new(),
            exhausted: false,
        }
    }

    /// 尝试完成当前样本并添加到批量
    fn finalize_sample(&mut self) {
        if self.current_sample.is_complete(self.num_frames) {
            let sample = std::mem::replace(&mut self.current_sample, BatchSample::new());
            self.current_batch.push(sample);
        }
    }

    /// 尝试完成当前批量并返回
    fn finalize_batch(&mut self) -> Option<SequenceFrame> {
        if self.current_batch.is_complete(self.batch_size) {
            let batch = std::mem::replace(&mut self.current_batch, BatchData::new());
            return Some(batch.to_sequence_frame());
        }
        None
    }
}

impl<I: Iterator<Item = SequenceFrame>> Iterator for BatchCollectIter<I> {
    type Item = SequenceFrame;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        loop {
            match self.inner.next() {
                Some(frame) => {
                    // 添加帧到当前样本
                    self.current_sample.push(frame);

                    // 检查样本是否完成
                    self.finalize_sample();

                    // 检查批量是否完成
                    if let Some(batch_frame) = self.finalize_batch() {
                        return Some(batch_frame);
                    }
                }
                None => {
                    // 输入已耗尽，处理剩余数据
                    self.exhausted = true;

                    // 如果当前样本有数据，添加到批量
                    if !self.current_sample.frames.is_empty() {
                        let sample = std::mem::replace(&mut self.current_sample, BatchSample::new());
                        self.current_batch.push(sample);
                    }

                    // 如果批量有数据，返回
                    if !self.current_batch.samples.is_empty() {
                        let batch = std::mem::replace(&mut self.current_batch, BatchData::new());
                        return Some(batch.to_sequence_frame());
                    }

                    return None;
                }
            }
        }
    }
}

impl Processor for BatchCollector {
    fn name(&self) -> &'static str {
        "batch_collector"
    }

    fn process(
        &self,
        input: Box<dyn Iterator<Item = SequenceFrame> + Send>,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let iter = BatchCollectIter::new(input, self.config.batch_size, self.config.num_frames);
        Ok(Box::new(iter))
    }
}

/// 工厂函数：从 JSON 配置创建 BatchCollector
pub fn create_batch_collector(config: &Value) -> CoreResult<Box<dyn Processor>> {
    let config: BatchCollectorConfig = if config.is_null() {
        return Err(CoreError::InvalidParams(
            "batch_collector requires configuration (batch_size, num_frames)".into(),
        ));
    } else {
        serde_json::from_value(config.clone()).map_err(|e| {
            CoreError::SerializationError(format!("batch_collector 配置解析失败: {}", e))
        })?
    };
    let collector = BatchCollector::new(config)?;
    Ok(Box::new(collector))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::registry::ProcessorRegistry;
    use crate::pipeline::register_all;

    fn make_frame(step: u64, values: Vec<i64>) -> SequenceFrame {
        let values: Vec<FrameState> = values.into_iter().map(FrameState::Integer).collect();
        SequenceFrame::new(step, FrameData { values })
    }

    #[test]
    fn test_batch_collector_basic() {
        // batch_size=2, num_frames=3 → 每批 2 个样本，每个样本 3 帧
        let frames: Vec<SequenceFrame> = (0..6)
            .map(|i| make_frame(i, vec![i as i64]))
            .collect();

        let config = BatchCollectorConfig {
            batch_size: 2,
            num_frames: 3,
        };
        let collector = BatchCollector::new(config).unwrap();
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> = Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = collector.process(input).unwrap().collect();

        // 应该输出 1 个批量
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].label, Some("batch_data".to_string()));
        // 6 帧，每帧 1 个值
        assert_eq!(output[0].state.values.len(), 6);
    }

    #[test]
    fn test_batch_collector_multiple_batches() {
        // batch_size=2, num_frames=2 → 每批 2 个样本，每个样本 2 帧
        // 8 帧 → 2 批
        let frames: Vec<SequenceFrame> = (0..8)
            .map(|i| make_frame(i, vec![i as i64]))
            .collect();

        let config = BatchCollectorConfig {
            batch_size: 2,
            num_frames: 2,
        };
        let collector = BatchCollector::new(config).unwrap();
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> = Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = collector.process(input).unwrap().collect();

        // 应该输出 2 个批量
        assert_eq!(output.len(), 2);
        for batch in &output {
            assert_eq!(batch.label, Some("batch_data".to_string()));
            // 每批 4 帧，每帧 1 个值
            assert_eq!(batch.state.values.len(), 4);
        }
    }

    #[test]
    fn test_batch_collector_partial_batch() {
        // batch_size=3, num_frames=2 → 每批 3 个样本，每个样本 2 帧
        // 4 帧 → 1 批（不完整）
        let frames: Vec<SequenceFrame> = (0..4)
            .map(|i| make_frame(i, vec![i as i64]))
            .collect();

        let config = BatchCollectorConfig {
            batch_size: 3,
            num_frames: 2,
        };
        let collector = BatchCollector::new(config).unwrap();
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> = Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = collector.process(input).unwrap().collect();

        // 应该输出 1 个批量（2 个样本）
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].state.values.len(), 4);
    }

    #[test]
    fn test_batch_collector_empty_input() {
        let config = BatchCollectorConfig {
            batch_size: 2,
            num_frames: 3,
        };
        let collector = BatchCollector::new(config).unwrap();
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> = Box::new(std::iter::empty());
        let output: Vec<SequenceFrame> = collector.process(input).unwrap().collect();

        assert!(output.is_empty());
    }

    #[test]
    fn test_batch_collector_invalid_batch_size() {
        let config = BatchCollectorConfig {
            batch_size: 0,
            num_frames: 3,
        };
        let result = BatchCollector::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_collector_invalid_num_frames() {
        let config = BatchCollectorConfig {
            batch_size: 2,
            num_frames: 0,
        };
        let result = BatchCollector::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_collector_via_registry() {
        let mut registry = ProcessorRegistry::new();
        register_all(&mut registry).unwrap();

        let config = serde_json::json!({
            "batch_size": 2,
            "num_frames": 3
        });

        let processor = registry.get("batch_collector", &config);
        assert!(processor.is_ok(), "Should be able to instantiate batch_collector via registry");
        assert_eq!(processor.unwrap().name(), "batch_collector");
    }

    #[test]
    fn test_batch_sample_is_complete() {
        let mut sample = BatchSample::new();
        assert!(!sample.is_complete(3));

        sample.push(make_frame(0, vec![0]));
        sample.push(make_frame(1, vec![1]));
        assert!(!sample.is_complete(3));

        sample.push(make_frame(2, vec![2]));
        assert!(sample.is_complete(3));
    }

    #[test]
    fn test_batch_data_is_complete() {
        let mut batch = BatchData::new();
        assert!(!batch.is_complete(2));

        batch.push(BatchSample::new());
        assert!(!batch.is_complete(2));

        batch.push(BatchSample::new());
        assert!(batch.is_complete(2));
    }
}
