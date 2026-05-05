use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::{CoreError, CoreResult, FrameData, FrameState, SequenceFrame};

use super::processor::Processor;

/// 分隔符帧的标签
const SEPARATOR_LABEL: &str = "__SEPARATOR__";

/// 序列截断/拼接器配置
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClipStitcherConfig {
    /// 最大序列长度（帧数），超出部分切分为新子序列
    /// None 表示不截断
    #[serde(default)]
    pub max_len: Option<usize>,
    /// 是否在子序列之间插入分隔标记帧
    #[serde(default)]
    pub insert_separator: bool,
}

/// 序列截断与拼接器：将过长的序列切分为固定长度子序列
///
/// 按 max_len 切分序列，超出部分作为新子序列。
/// 如果 insert_separator，在子序列间插入分隔帧
/// （step_index 递增，label="__SEPARATOR__"）。
pub struct ClipStitcher {
    config: ClipStitcherConfig,
}

/// 截断拼接迭代器适配器
struct ClipStitchIter<I: Iterator<Item = SequenceFrame>> {
    inner: I,
    config: ClipStitcherConfig,
    /// 待输出的帧缓冲区
    buffer: Vec<SequenceFrame>,
    /// 全局步索引计数器（分隔帧使用）
    global_step: u64,
    /// 是否有待输出的分隔帧
    pending_separator: bool,
    /// 当前批次是否填满了 max_len
    current_batch_full: bool,
}

impl<I: Iterator<Item = SequenceFrame>> Iterator for ClipStitchIter<I> {
    type Item = SequenceFrame;

    fn next(&mut self) -> Option<Self::Item> {
        // 如果有待输出的分隔符，优先输出并重置批次状态
        if self.pending_separator {
            self.pending_separator = false;
            self.current_batch_full = false;
            let step = self.global_step;
            self.global_step += 1;
            return Some(SequenceFrame {
                step_index: step,
                state: FrameData { values: vec![] },
                label: Some(SEPARATOR_LABEL.to_string()),
                sample_id: None,
            });
        }

        let max_len = match self.config.max_len {
            Some(len) => len,
            None => {
                if !self.buffer.is_empty() {
                    return Some(self.buffer.remove(0));
                }
                return self.inner.next();
            }
        };

        // 填充阶段：如果批次未满，从输入继续填充缓冲区
        if self.buffer.is_empty() || !self.current_batch_full {
            while self.buffer.len() < max_len {
                match self.inner.next() {
                    Some(frame) => self.buffer.push(frame),
                    None => break,
                }
            }
            self.current_batch_full = self.buffer.len() >= max_len;
        }

        if self.buffer.is_empty() {
            return None;
        }

        // 输出阶段：输出第一帧
        let first = self.buffer.remove(0);

        // 批次输出完毕且为满批次且需要分隔符时，探测是否还有更多输入
        if self.buffer.is_empty()
            && self.current_batch_full
            && self.config.insert_separator
        {
            if let Some(preload) = self.inner.next() {
                self.buffer.push(preload);
                self.pending_separator = true;
            }
        }

        Some(first)
    }
}

impl ClipStitcher {
    /// 根据配置创建截断拼接器
    pub fn new(config: ClipStitcherConfig) -> Self {
        ClipStitcher { config }
    }
}

impl Processor for ClipStitcher {
    fn name(&self) -> &'static str {
        "clip_stitcher"
    }

    fn process(
        &self,
        input: Box<dyn Iterator<Item = SequenceFrame> + Send>,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let iter = ClipStitchIter {
            inner: input,
            config: self.config.clone(),
            buffer: Vec::new(),
            global_step: 0,
            pending_separator: false,
            current_batch_full: false,
        };
        Ok(Box::new(iter))
    }
}

/// 工厂函数：从 JSON 配置创建序列截断拼接器
pub fn create_clip_stitcher(config: &Value) -> CoreResult<Box<dyn Processor>> {
    let config: ClipStitcherConfig = if config.is_null() {
        ClipStitcherConfig::default()
    } else {
        serde_json::from_value(config.clone()).map_err(|e| {
            CoreError::SerializationError(format!("序列截断拼接器配置解析失败: {}", e))
        })?
    };
    Ok(Box::new(ClipStitcher::new(config)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(step: u64, val: i64) -> SequenceFrame {
        SequenceFrame::new(
            step,
            FrameData {
                values: vec![FrameState::Integer(val)],
            },
        )
    }

    #[test]
    fn test_clip_stitcher_truncates_sequence() {
        let frames: Vec<SequenceFrame> = (0..7).map(|i| make_frame(i, i as i64)).collect();
        let config = ClipStitcherConfig {
            max_len: Some(3),
            insert_separator: false,
        };
        let stitcher = ClipStitcher::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = stitcher.process(input).unwrap().collect();

        // 7 帧以 max_len=3 截断：应输出全部 7 帧（不丢弃帧）
        assert_eq!(output.len(), 7);
        for (i, frame) in output.iter().enumerate() {
            assert_eq!(frame.step_index, i as u64);
        }
    }

    #[test]
    fn test_clip_stitcher_insert_separator() {
        let frames: Vec<SequenceFrame> = (0..6).map(|i| make_frame(i, i as i64)).collect();
        let config = ClipStitcherConfig {
            max_len: Some(3),
            insert_separator: true,
        };
        let stitcher = ClipStitcher::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = stitcher.process(input).unwrap().collect();

        // 6 帧 + 1 个分隔符 = 7 帧
        assert_eq!(output.len(), 7);

        // 检查分隔符帧
        let separator_frames: Vec<&SequenceFrame> = output
            .iter()
            .filter(|f| f.label.as_deref() == Some(SEPARATOR_LABEL))
            .collect();
        assert_eq!(separator_frames.len(), 1);

        // 分隔符帧应该有 step_index 和空 values
        let sep = separator_frames[0];
        assert!(sep.state.values.is_empty());
    }

    #[test]
    fn test_clip_stitcher_no_max_len() {
        let frames: Vec<SequenceFrame> = (0..5).map(|i| make_frame(i, i as i64)).collect();
        let config = ClipStitcherConfig {
            max_len: None,
            insert_separator: false,
        };
        let stitcher = ClipStitcher::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = stitcher.process(input).unwrap().collect();

        // max_len=None 时应输出全部帧
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_clip_stitcher_empty_input() {
        let config = ClipStitcherConfig::default();
        let stitcher = ClipStitcher::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(std::iter::empty());
        let output: Vec<SequenceFrame> = stitcher.process(input).unwrap().collect();
        assert!(output.is_empty());
    }

    #[test]
    fn test_clip_stitcher_max_len_one() {
        let frames: Vec<SequenceFrame> = (0..3).map(|i| make_frame(i, i as i64)).collect();
        let config = ClipStitcherConfig {
            max_len: Some(1),
            insert_separator: true,
        };
        let stitcher = ClipStitcher::new(config);
        let input: Box<dyn Iterator<Item = SequenceFrame> + Send> =
            Box::new(frames.into_iter());
        let output: Vec<SequenceFrame> = stitcher.process(input).unwrap().collect();

        // 3 帧 + 2 个分隔符 = 5 帧
        assert_eq!(output.len(), 5);

        let separator_count = output
            .iter()
            .filter(|f| f.label.as_deref() == Some(SEPARATOR_LABEL))
            .count();
        assert_eq!(separator_count, 2);
    }
}
