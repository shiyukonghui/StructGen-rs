//! 2D 元胞自动机网格动画渲染

use eframe::egui::{Color32, ColorImage, Context, Pos2, Rect, TextureHandle, TextureOptions, Vec2};

use crate::core::frame::FrameState;
use crate::view::color::CaColorMap;
use crate::view::frame_buffer::CaFrameBuffer;

/// 2D 网格动画渲染器
pub struct Render2D {
    rows: usize,
    cols: usize,
    n_groups: u8,
    current_group: u8,
    texture: Option<TextureHandle>,
}

impl Render2D {
    pub fn new(rows: usize, cols: usize, n_groups: u8) -> Self {
        Render2D {
            rows,
            cols,
            n_groups,
            current_group: 0,
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
        // 多通道时显示分组选择滑块
        if self.n_groups > 1 {
            ui.horizontal(|ui| {
                ui.label("Group:");
                let mut g = self.current_group as usize;
                ui.add(
                    eframe::egui::Slider::new(&mut g, 0..=(self.n_groups - 1) as usize)
                        .text(""),
                );
                self.current_group = g as u8;
                ui.label(format!("({}/{})", g + 1, self.n_groups));
            });
        }

        let frames = buffer.frames.lock().unwrap();
        let available_rect = ui.available_rect_before_wrap();
        let available = available_rect.size();
        if available.x <= 0.0 || available.y <= 0.0 {
            return;
        }

        // 获取当前帧（current_step 是绝对步数，需映射到缓冲索引）
        // 当 current_step <= base（旧帧被裁剪），跳到缓冲区最新帧
        let base = buffer.base_step();
        let buf_len = frames.len();
        if buf_len == 0 {
            ui.label("Waiting for frames...");
            return;
        }
        let frame_idx = if current_step <= base {
            buf_len - 1
        } else {
            (current_step - base).saturating_sub(1).min(buf_len - 1)
        };
        let frame = &frames[frame_idx];

        // 构建 ColorImage
        let img_w = self.cols;
        let img_h = self.rows;
        let mut pixels = vec![Color32::BLACK; img_w * img_h];

        let n_groups_usize = self.n_groups as usize;
        let current_group_usize = self.current_group as usize;
        let slice_len = img_w * img_h;

        for i in 0..slice_len {
            // NCA2D 帧布局: values[(r*cols + c) * n_groups + g]
            // CA2D 帧布局: values[r*cols + c] (n_groups == 1)
            let src_idx = if n_groups_usize > 1 {
                i * n_groups_usize + current_group_usize
            } else {
                i
            };
            let color = if src_idx < frame.state.values.len() {
                match &frame.state.values[src_idx] {
                    FrameState::Integer(v) => color_map.color_for_state(*v as u8),
                    FrameState::Bool(b) => {
                        if *b {
                            Color32::WHITE
                        } else {
                            Color32::BLACK
                        }
                    }
                    _ => Color32::BLACK,
                }
            } else {
                Color32::BLACK
            };
            let row = i / img_w;
            let col = i % img_w;
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
