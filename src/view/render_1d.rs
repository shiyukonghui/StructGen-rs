//! 1D 元胞自动机时空图渲染

use eframe::egui::{Color32, ColorImage, Context, Pos2, Rect, TextureHandle, TextureOptions, Vec2};

use crate::core::frame::FrameState;
use crate::view::color::CaColorMap;
use crate::view::frame_buffer::CaFrameBuffer;

/// 1D 时空图渲染器
pub struct Render1D {
    width: usize,
    max_rows: usize,
    /// 时空图纹理
    texture: Option<TextureHandle>,
}

/// 时空图最大显示行数（超出后滚动）
const MAX_DISPLAY_ROWS: usize = 2000;

impl Render1D {
    pub fn new(width: usize) -> Self {
        Render1D {
            width,
            max_rows: MAX_DISPLAY_ROWS,
            texture: None,
        }
    }

    /// 渲染 1D 时空图
    pub fn render(
        &mut self,
        ui: &mut eframe::egui::Ui,
        ctx: &Context,
        buffer: &CaFrameBuffer,
        current_step: usize,
        color_map: &CaColorMap,
    ) {
        let frames = buffer.frames.lock().unwrap();
        let available = ui.available_size();
        if available.x <= 0.0 || available.y <= 0.0 {
            return;
        }

        // 确定需要渲染的行范围
        let total_frames = frames.len();
        let steps_to_show = current_step.min(total_frames);
        if steps_to_show == 0 {
            ui.label("Waiting for frames...");
            return;
        }

        // 计算实际显示行数（受窗口高度限制）
        let display_rows = steps_to_show.min(self.max_rows);
        let start_step = steps_to_show.saturating_sub(display_rows);

        // 构建 ColorImage
        let img_w = self.width;
        let img_h = display_rows;
        let mut pixels = vec![Color32::from_rgb(20, 20, 30); img_w * img_h];

        for (row_idx, step) in (start_step..steps_to_show).enumerate() {
            if step >= total_frames {
                break;
            }
            let frame = &frames[step];
            for (col, state) in frame.state.values.iter().enumerate() {
                if col >= img_w {
                    break;
                }
                let color = match state {
                    FrameState::Bool(b) => {
                        if *b {
                            Color32::WHITE
                        } else {
                            Color32::BLACK
                        }
                    }
                    FrameState::Integer(v) => color_map.color_for_state(*v as u8),
                    _ => Color32::BLACK,
                };
                pixels[row_idx * img_w + col] = color;
            }
        }

        let color_image = ColorImage {
            size: [img_w, img_h],
            source_size: eframe::egui::Vec2::new(img_w as f32, img_h as f32),
            pixels,
        };

        // 上传/更新纹理
        let tex = upload_texture(ctx, "ca_1d_spacetime", color_image, &mut self.texture);

        // 计算显示区域，保持宽高比
        let scale_x = available.x / img_w as f32;
        let scale_y = available.y / img_h as f32;
        let scale = scale_x.min(scale_y);
        let display_w = img_w as f32 * scale;
        let display_h = img_h as f32 * scale;

        let offset_x = (available.x - display_w) * 0.5;
        let offset_y = (available.y - display_h) * 0.5;

        let rect = Rect::from_min_size(
            ui.painter().clip_rect().min + Vec2::new(offset_x, offset_y),
            Vec2::new(display_w, display_h),
        );

        let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
        ui.painter().image(tex.id(), rect, uv, Color32::WHITE);
        drop(frames);
    }
}

/// 上传或更新纹理
fn upload_texture(
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
