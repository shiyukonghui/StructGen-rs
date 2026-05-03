//! 进度追踪模块，提供线程安全的生成进度追踪

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// 进度快照，包含当前生成进度的完整信息
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    /// 已完成样本数
    pub completed_samples: u64,
    /// 总样本数
    pub total_samples: u64,
    /// 完成百分比 (0.0 ~ 100.0)
    pub percent: f64,
    /// 已生成的总帧数
    pub total_frames: u64,
    /// 已耗时（秒）
    pub elapsed_secs: f64,
    /// 预计剩余时间（秒）
    pub eta_secs: f64,
}

/// 线程安全的进度追踪器，可跨线程共享
#[derive(Clone)]
pub struct ProgressTracker {
    total_samples: u64,
    completed_samples: Arc<AtomicU64>,
    total_frames: Arc<AtomicU64>,
    start_instant: Instant,
}

impl ProgressTracker {
    /// 创建一个新的进度追踪器
    ///
    /// # 参数
    /// * `total_samples` - 总样本数
    pub fn new(total_samples: usize) -> Self {
        Self {
            total_samples: total_samples as u64,
            completed_samples: Arc::new(AtomicU64::new(0)),
            total_frames: Arc::new(AtomicU64::new(0)),
            start_instant: Instant::now(),
        }
    }

    /// 报告已完成的样本和帧数，线程安全
    ///
    /// # 参数
    /// * `samples` - 本次完成的样本数
    /// * `frames` - 本次生成的帧数
    pub fn report_completed(&self, samples: usize, frames: u64) {
        self.completed_samples
            .fetch_add(samples as u64, Ordering::Relaxed);
        self.total_frames
            .fetch_add(frames, Ordering::Relaxed);
    }

    /// 获取当前进度快照
    pub fn progress(&self) -> ProgressInfo {
        let completed = self.completed_samples.load(Ordering::Relaxed);
        let total = self.total_samples;

        let percent = if total == 0 {
            0.0
        } else {
            completed as f64 / total as f64 * 100.0
        };

        let elapsed_secs = self.start_instant.elapsed().as_secs_f64();

        let eta_secs = if completed == 0 {
            0.0
        } else {
            let remaining = total.saturating_sub(completed) as f64;
            let rate = completed as f64 / elapsed_secs; // 样本/秒
            let eta = remaining / rate;
            eta.max(0.0)
        };

        ProgressInfo {
            completed_samples: completed,
            total_samples: total,
            percent,
            total_frames: self.total_frames.load(Ordering::Relaxed),
            elapsed_secs,
            eta_secs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_tracker_initial_state() {
        let tracker = ProgressTracker::new(100);
        let info = tracker.progress();

        assert_eq!(info.total_samples, 100);
        assert_eq!(info.completed_samples, 0);
        assert_eq!(info.total_frames, 0);
        assert_eq!(info.percent, 0.0);
        assert_eq!(info.eta_secs, 0.0);
    }

    #[test]
    fn test_progress_tracker_basic() {
        let tracker = ProgressTracker::new(100);
        tracker.report_completed(50, 500);

        let info = tracker.progress();
        assert_eq!(info.completed_samples, 50);
        assert_eq!(info.total_samples, 100);
        assert_eq!(info.total_frames, 500);
        assert!(info.percent > 45.0 && info.percent < 55.0);
        assert!(info.elapsed_secs >= 0.0);
    }

    #[test]
    fn test_progress_tracker_eta_non_negative() {
        let tracker = ProgressTracker::new(100);
        tracker.report_completed(10, 100);

        let info1 = tracker.progress();
        assert!(info1.eta_secs >= 0.0);

        tracker.report_completed(20, 200);
        let info2 = tracker.progress();
        assert!(info2.eta_secs >= 0.0);
    }

    #[test]
    fn test_progress_tracker_zero_samples() {
        let tracker = ProgressTracker::new(0);
        let info = tracker.progress();

        assert_eq!(info.total_samples, 0);
        assert_eq!(info.percent, 0.0);
        assert_eq!(info.eta_secs, 0.0);
        assert_eq!(info.completed_samples, 0);

        tracker.report_completed(5, 50);
        let info2 = tracker.progress();
        assert_eq!(info2.completed_samples, 5);
        assert_eq!(info2.percent, 0.0); // total=0 时分母为0，percent 应为 0
    }

    #[test]
    fn test_progress_tracker_complete() {
        let tracker = ProgressTracker::new(100);
        tracker.report_completed(100, 1000);

        let info = tracker.progress();
        assert_eq!(info.completed_samples, 100);
        assert_eq!(info.total_samples, 100);
        assert_eq!(info.total_frames, 1000);
        assert!(
            (info.percent - 100.0).abs() < 0.001,
            "percent should be ~100, got {}",
            info.percent
        );
        assert!(
            info.eta_secs <= 0.001,
            "eta should be ~0 when complete, got {}",
            info.eta_secs
        );
    }

    #[test]
    fn test_progress_tracker_thread_safety() {
        let tracker = ProgressTracker::new(1000);

        std::thread::scope(|s| {
            for _ in 0..10 {
                s.spawn(|| {
                    for _ in 0..100 {
                        tracker.report_completed(1, 10);
                    }
                });
            }
        });

        let info = tracker.progress();
        assert_eq!(info.completed_samples, 1000);
        assert_eq!(info.total_frames, 10000);
    }
}
