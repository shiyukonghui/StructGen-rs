//! 线程桥接：生成器迭代器 → UI 线程

use std::sync::atomic::{AtomicBool, Ordering};
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
pub struct CaFrameBuffer {
    /// 累积的帧数据
    pub frames: Arc<Mutex<Vec<SequenceFrame>>>,
    /// 网格维度元数据
    pub dimension: CaDimension,
    /// 迭代器是否已耗尽
    pub generator_exhausted: Arc<AtomicBool>,
}

impl CaFrameBuffer {
    /// 创建新的帧缓冲
    pub fn new(dimension: CaDimension) -> Self {
        CaFrameBuffer {
            frames: Arc::new(Mutex::new(Vec::new())),
            dimension,
            generator_exhausted: Arc::new(AtomicBool::new(false)),
        }
    }

    /// 启动后台线程消费生成器迭代器
    pub fn spawn_generator_thread(
        &self,
        iterator: Box<dyn Iterator<Item = SequenceFrame> + Send>,
    ) {
        let frames = Arc::clone(&self.frames);
        let exhausted = Arc::clone(&self.generator_exhausted);

        std::thread::spawn(move || {
            for frame in iterator {
                let mut buf = frames.lock().unwrap();
                buf.push(frame);
            }
            exhausted.store(true, Ordering::Relaxed);
        });
    }

    /// 获取当前已缓冲的帧数
    pub fn frame_count(&self) -> usize {
        self.frames.lock().unwrap().len()
    }
}
