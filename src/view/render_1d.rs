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
        let available_rect = ui.available_rect_before_wrap();
        let available = available_rect.size();
        if available.x <= 0.0 || available.y <= 0.0 {
            return;
        }

        // 将绝对步数映射到缓冲索引
        let base = buffer.base_step();
        let buf_len = frames.len();
        if buf_len == 0 {
            ui.label("Waiting for frames...");
            return;
        }

        // 计算可显示的缓冲区范围
        // - current_step <= base：旧帧已被裁剪，直接显示缓冲区中最新的帧
        // - current_step > base：正常动画模式，显示到 current_step 对应位置
        let show_end = if current_step <= base {
            buf_len
        } else {
            (current_step - base).min(buf_len)
        };
        if show_end == 0 {
            ui.label("Waiting for frames...");
            return;
        }

        // 显示最近 max_rows 帧（滚动窗口）
        let display_rows = show_end.min(self.max_rows);
        let show_start = show_end.saturating_sub(display_rows);

        // 构建 ColorImage
        let img_w = self.width;
        let img_h = display_rows;
        let mut pixels = vec![Color32::from_rgb(20, 20, 30); img_w * img_h];

        for (row_idx, buf_idx) in (show_start..show_end).enumerate() {
            let frame = &frames[buf_idx];
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

        // 时空图水平拉伸填满可用宽度，仅垂直方向保持像素比例
        // （128px 宽 × 2000 行的宽高比极窄，保持比例会导致有效显示区域过小）
        let display_w = available.x;
        let display_h = available.y;

        let rect = Rect::from_min_size(
            available_rect.min,
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
