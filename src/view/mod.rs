//! 元胞自动机实时可视化模块
//!
//! 基于 egui/eframe 实现 1D/2D/3D 元胞自动机的实时渲染展示。
//! 通过 CLI 子命令 `structgen-rs view` 启动。

#[cfg(feature = "view")]
mod app;
#[cfg(feature = "view")]
mod color;
#[cfg(feature = "view")]
mod frame_buffer;
#[cfg(feature = "view")]
mod render_1d;
#[cfg(feature = "view")]
mod render_2d;
#[cfg(feature = "view")]
mod render_3d;

#[cfg(feature = "view")]
pub use app::launch_viewer;

/// View 子命令参数（始终可用，不依赖 view feature）
#[derive(Debug, Clone)]
pub struct ViewArgs {
    pub generator: String,
    pub rule: Option<String>,
    pub seed: u64,
    pub steps: usize,
    pub width: usize,
    pub rows: usize,
    pub cols: usize,
    pub depth: usize,
    pub speed: u64,
    pub boundary: String,
    pub init: String,
}

#[cfg(not(feature = "view"))]
pub fn launch_viewer(_args: ViewArgs) -> i32 {
    eprintln!("Error: view feature is not enabled. Rebuild with --features view");
    2
}
