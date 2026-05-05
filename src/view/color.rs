//! 多状态元胞自动机配色方案

use eframe::egui::Color32;

/// 元胞自动机配色映射表
pub struct CaColorMap {
    palette: Vec<Color32>,
}

impl CaColorMap {
    /// 根据状态数创建配色方案
    ///
    /// - d_state=2: 二值黑白
    /// - d_state=4: WireWorld 专用配色
    /// - 其他: HSV 色相均匀旋转
    pub fn new(d_state: u8) -> Self {
        let palette = match d_state {
            2 => vec![Color32::BLACK, Color32::WHITE],
            4 => vec![
                Color32::BLACK,                // 0: 空
                Color32::from_rgb(0, 100, 255), // 1: 电子头 (蓝)
                Color32::from_rgb(255, 60, 30), // 2: 电子尾 (红)
                Color32::from_rgb(255, 200, 0), // 3: 铜 (黄)
            ],
            _ => {
                let count = d_state as usize;
                let mut pal = Vec::with_capacity(count);
                for i in 0..count {
                    if i == 0 {
                        pal.push(Color32::BLACK);
                    } else {
                        let hue = (i as f32 / count as f32) * 360.0;
                        let (r, g, b) = hsv_to_rgb(hue, 0.9, 1.0);
                        pal.push(Color32::from_rgb(r, g, b));
                    }
                }
                pal
            }
        };
        CaColorMap { palette }
    }

    /// 获取指定状态值对应的颜色
    pub fn color_for_state(&self, state: u8) -> Color32 {
        let idx = state as usize;
        if idx < self.palette.len() {
            self.palette[idx]
        } else {
            Color32::MAGENTA // 越界标记为洋红
        }
    }
}

/// HSV → RGB 转换 (h: 0-360, s: 0-1, v: 0-1) → (r, g, b) 各 0-255
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = match h {
        h if h < 60.0 => (c, x, 0.0),
        h if h < 120.0 => (x, c, 0.0),
        h if h < 180.0 => (0.0, c, x),
        h if h < 240.0 => (0.0, x, c),
        h if h < 300.0 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    )
}
