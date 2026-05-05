//! 线程桥接：生成器迭代器 → UI 线程
//!
//! 使用滑动窗口缓冲，只保留最近 N 帧数据，防止长时间运行时内存无限增长。

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::core::frame::SequenceFrame;

/// CA 维度枚举
#[derive(Debug, Clone)]
pub enum CaDimension {
    OneD { width: usize },
    TwoD { rows: usize, cols: usize, d_state: u8 },
    ThreeD { depth: usize, rows: usize, cols: usize, d_state: u8 },
}

/// 生成器与 UI 线程之间的帧缓冲
///
/// 内部使用 `VecDeque` 实现滑动窗口：当缓冲帧数超过 `max_frames` 时，
/// 自动丢弃最旧的帧，并通过 `base_step` 记录已丢弃的偏移量，
/// 使渲染器能正确映射绝对步数到缓冲索引。
pub struct CaFrameBuffer {
    /// 累积的帧数据（滑动窗口）
    pub frames: Arc<Mutex<VecDeque<SequenceFrame>>>,
    /// 网格维度元数据
    pub dimension: CaDimension,
    /// 迭代器是否已耗尽
    pub generator_exhausted: Arc<AtomicBool>,
    /// 已从缓冲头部移除的帧数（用于步数→索引映射）
    base_step: Arc<AtomicU64>,
    /// 已生成的总帧数（不受滑动窗口裁剪影响）
    total_generated: Arc<AtomicU64>,
    /// 缓冲区最大帧数
    max_frames: usize,
}

/// 1D 时空图最大显示行数（与 render_1d 保持一致）
const MAX_DISPLAY_ROWS_1D: usize = 2000;

impl CaFrameBuffer {
    /// 创建新的帧缓冲
    pub fn new(dimension: CaDimension) -> Self {
        let max_frames = match &dimension {
            // 1D 时空图需要保留足够行数用于显示
            CaDimension::OneD { .. } => MAX_DISPLAY_ROWS_1D + 500,
            // 2D/3D 仅需少量帧供播放器追赶
            CaDimension::TwoD { .. } => 200,
            CaDimension::ThreeD { .. } => 200,
        };
        CaFrameBuffer {
            frames: Arc::new(Mutex::new(VecDeque::new())),
            dimension,
            generator_exhausted: Arc::new(AtomicBool::new(false)),
            base_step: Arc::new(AtomicU64::new(0)),
            total_generated: Arc::new(AtomicU64::new(0)),
            max_frames,
        }
    }

    /// 启动后台线程消费生成器迭代器
    pub fn spawn_generator_thread(
        &self,
        iterator: Box<dyn Iterator<Item = SequenceFrame> + Send>,
    ) {
        let frames = Arc::clone(&self.frames);
        let exhausted = Arc::clone(&self.generator_exhausted);
        let base_step = Arc::clone(&self.base_step);
        let total_generated = Arc::clone(&self.total_generated);
        let max_frames = self.max_frames;

        std::thread::spawn(move || {
            for frame in iterator {
                let mut buf = frames.lock().unwrap();
                buf.push_back(frame);
                total_generated.fetch_add(1, Ordering::Relaxed);
                // 滑动窗口裁剪：超出容量时移除最旧帧
                while buf.len() > max_frames {
                    buf.pop_front();
                    base_step.fetch_add(1, Ordering::Relaxed);
                }
            }
            exhausted.store(true, Ordering::Relaxed);
        });
    }

    /// 获取已生成的总帧数（不受滑动窗口裁剪影响）
    pub fn frame_count(&self) -> usize {
        self.total_generated.load(Ordering::Relaxed) as usize
    }

    /// 获取缓冲区首帧对应的绝对步数
    ///
    /// 渲染器通过 `buffer_index = current_step - base_step` 映射绝对步数到缓冲索引。
    pub fn base_step(&self) -> usize {
        self.base_step.load(Ordering::Relaxed) as usize
    }
}
