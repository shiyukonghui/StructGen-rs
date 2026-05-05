//! 2D 元胞自动机网格动画渲染

use eframe::egui::{Color32, ColorImage, Context, Pos2, Rect, TextureHandle, TextureOptions, Vec2};

use crate::core::frame::FrameState;
use crate::view::color::CaColorMap;
use crate::view::frame_buffer::CaFrameBuffer;

/// 2D 网格动画渲染器
pub struct Render2D {
    rows: usize,
    cols: usize,
    texture: Option<TextureHandle>,
}

impl Render2D {
    pub fn new(rows: usize, cols: usize) -> Self {
        Render2D {
            rows,
            cols,
            texture: None,
        }
    }

    /// 渲染 2D 网格（单帧）
    pub fn render(
        &mut self,
        ui: &mut eframe::egui::Ui,
        ctx: &Context,
        buffer: &CaFrameBuffer,
        current_step: usize,
        color_map: &CaColorMap,
    ) {
        let frames = buffer.frames.lock().unwrap();
        let available_rect = ui.available_rect_before_wrap();
        let available = available_rect.size();
        if available.x <= 0.0 || available.y <= 0.0 {
            return;
        }

        // 获取当前帧（current_step 是已推进的帧数，显示最后一帧）
        let frame_idx = current_step.saturating_sub(1);
        let frame = match frames.get(frame_idx) {
            Some(f) => f,
            None => {
                ui.label("Waiting for frames...");
                return;
            }
        };

        // 构建 ColorImage
        let img_w = self.cols;
        let img_h = self.rows;
        let mut pixels = vec![Color32::BLACK; img_w * img_h];

        for (idx, state) in frame.state.values.iter().enumerate() {
            if idx >= img_w * img_h {
                break;
            }
            let color = match state {
                FrameState::Integer(v) => color_map.color_for_state(*v as u8),
                FrameState::Bool(b) => {
                    if *b {
                        Color32::WHITE
                    } else {
                        Color32::BLACK
                    }
                }
                _ => Color32::BLACK,
            };
            // values 按 row-major 排列: values[row * cols + col]
            // ColorImage 按 (x, y) 即 (col, row) 排列: pixels[y * width + x]
            let row = idx / img_w;
            let col = idx % img_w;
            pixels[row * img_w + col] = color;
        }

        let color_image = ColorImage {
            size: [img_w, img_h],
            source_size: eframe::egui::Vec2::new(img_w as f32, img_h as f32),
            pixels,
        };

        let tex = upload_texture_2d(ctx, "ca_2d_grid", color_image, &mut self.texture);

        // 保持宽高比缩放
        let scale_x = available.x / img_w as f32;
        let scale_y = available.y / img_h as f32;
        let scale = scale_x.min(scale_y);
        let display_w = img_w as f32 * scale;
        let display_h = img_h as f32 * scale;

        let offset_x = (available.x - display_w) * 0.5;
        let offset_y = (available.y - display_h) * 0.5;

        let rect = Rect::from_min_size(
            available_rect.min + Vec2::new(offset_x, offset_y),
            Vec2::new(display_w, display_h),
        );

        let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
        ui.painter().image(tex.id(), rect, uv, Color32::WHITE);
        drop(frames);
    }
}

fn upload_texture_2d(
    ctx: &Context,
    id: &str,
    image: ColorImage,
    existing: &mut Option<TextureHandle>,
) -> TextureHandle {
    match existing {
        Some(tex) => {
            tex.set(image, TextureOptions::NEAREST);
            tex.clone()
        }
        None => {
            let handle = ctx.load_texture(id, image, TextureOptions::NEAREST);
            *existing = Some(handle.clone());
            handle
        }
    }
}
