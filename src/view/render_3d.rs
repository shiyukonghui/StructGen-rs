//! 3D 元胞自动机切片视图渲染

use eframe::egui::{Color32, ColorImage, Context, Pos2, Rect, TextureHandle, TextureOptions, Vec2};

use crate::core::frame::FrameState;
use crate::view::color::CaColorMap;
use crate::view::frame_buffer::CaFrameBuffer;

/// 3D 切片视图渲染器
pub struct Render3D {
    depth: usize,
    rows: usize,
    cols: usize,
    /// 当前显示的深度层
    depth_layer: usize,
    texture: Option<TextureHandle>,
}

impl Render3D {
    pub fn new(depth: usize, rows: usize, cols: usize) -> Self {
        Render3D {
            depth,
            rows,
            cols,
            depth_layer: depth / 2, // 默认显示中间层
            texture: None,
        }
    }

    /// 渲染 3D 切片视图
    pub fn render(
        &mut self,
        ui: &mut eframe::egui::Ui,
        ctx: &Context,
        buffer: &CaFrameBuffer,
        current_step: usize,
        color_map: &CaColorMap,
    ) {
        // 底部控制栏：深度层选择
        let bottom_bar_height = 40.0;
        let available_rect = ui.available_rect_before_wrap();
        let available = available_rect.size();
        if available.x <= 0.0 || available.y <= bottom_bar_height {
            return;
        }

        // 深度层滑块
        ui.horizontal(|ui| {
            ui.label("Depth Layer:");
            let mut layer = self.depth_layer;
            ui.add(
                eframe::egui::Slider::new(&mut layer, 0..=self.depth.saturating_sub(1))
                    .text(""),
            );
            self.depth_layer = layer;
            ui.label(format!("({}/{})", self.depth_layer + 1, self.depth));
        });

        let frames = buffer.frames.lock().unwrap();

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
        let frame = match frames.get(frame_idx) {
            Some(f) => f,
            None => {
                ui.label("Waiting for frames...");
                return;
            }
        };

        // 提取当前深度层的切片
        let layer = self.depth_layer;
        let img_w = self.cols;
        let img_h = self.rows;
        let slice_offset = layer * (self.rows * self.cols);
        let slice_len = self.rows * self.cols;

        let mut pixels = vec![Color32::BLACK; img_w * img_h];

        for i in 0..slice_len {
            let src_idx = slice_offset + i;
            let color = if src_idx < frame.state.values.len() {
                match &frame.state.values[src_idx] {
                    FrameState::Integer(v) => color_map.color_for_state(*v as u8),
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

        let tex = upload_texture_3d(ctx, "ca_3d_slice", color_image, &mut self.texture);

        // 缩放渲染
        let paint_available_rect = ui.available_rect_before_wrap();
        let paint_available = paint_available_rect.size();
        if paint_available.x <= 0.0 || paint_available.y <= 0.0 {
            drop(frames);
            return;
        }

        let scale_x = paint_available.x / img_w as f32;
        let scale_y = paint_available.y / img_h as f32;
        let scale = scale_x.min(scale_y);
        let display_w = img_w as f32 * scale;
        let display_h = img_h as f32 * scale;

        let offset_x = (paint_available.x - display_w) * 0.5;
        let offset_y = (paint_available.y - display_h) * 0.5;

        let rect = Rect::from_min_size(
            paint_available_rect.min + Vec2::new(offset_x, offset_y),
            Vec2::new(display_w, display_h),
        );

        let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
        ui.painter().image(tex.id(), rect, uv, Color32::WHITE);
        drop(frames);
    }
}

fn upload_texture_3d(
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
