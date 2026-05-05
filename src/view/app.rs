//! CaViewApp — eframe::App 实现

use std::time::Instant;

use eframe::egui;

use crate::core::params::{GenParams, GridSize, GridSize3D};
use crate::core::registry::GeneratorRegistry;
use crate::generators::register_all as register_all_generators;
use crate::view::color::CaColorMap;
use crate::view::frame_buffer::{CaDimension, CaFrameBuffer};
use crate::view::render_1d::Render1D;
use crate::view::render_2d::Render2D;
use crate::view::render_3d::Render3D;
use crate::view::ViewArgs;

/// 元胞自动机可视化应用
struct CaViewApp {
    frame_buffer: CaFrameBuffer,
    current_step: usize,
    speed_ms: u64,
    last_step_time: Instant,
    color_map: CaColorMap,
    generator_name: String,
    seq_length: usize,
    /// 渲染器（按维度选择）
    renderer: Renderer,
}

/// 维度相关渲染器
enum Renderer {
    OneD(Render1D),
    TwoD(Render2D),
    ThreeD(Render3D),
}

impl CaViewApp {
    fn new(
        frame_buffer: CaFrameBuffer,
        color_map: CaColorMap,
        generator_name: String,
        seq_length: usize,
        speed_ms: u64,
    ) -> Self {
        let renderer = match &frame_buffer.dimension {
            CaDimension::OneD { width } => Renderer::OneD(Render1D::new(*width)),
            CaDimension::TwoD { rows, cols, .. } => Renderer::TwoD(Render2D::new(*rows, *cols)),
            CaDimension::ThreeD { depth, rows, cols, .. } => {
                Renderer::ThreeD(Render3D::new(*depth, *rows, *cols))
            }
        };
        CaViewApp {
            frame_buffer,
            current_step: 0,
            speed_ms,
            last_step_time: Instant::now(),
            color_map,
            generator_name,
            seq_length,
            renderer,
        }
    }
}

impl eframe::App for CaViewApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // 请求持续重绘以推进动画
        ui.ctx().request_repaint();

        // 推进步数
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_step_time).as_millis() as u64;
        let total_available = self.frame_buffer.frame_count();

        if elapsed >= self.speed_ms && total_available > self.current_step {
            // 所有维度：按 speed_ms 节奏逐帧推进（不跳帧）
            let steps_behind = total_available - self.current_step;
            let steps_to_advance = ((elapsed / self.speed_ms) as usize).min(steps_behind);
            self.current_step += steps_to_advance;
            self.last_step_time = now;
        }

        // 主面板（使用 show_inside，egui 0.34 API）
        egui::CentralPanel::default().show_inside(ui, |ui| {
            // 信息栏
            let exhausted =
                self.frame_buffer
                    .generator_exhausted
                    .load(std::sync::atomic::Ordering::Relaxed);
            let status = if exhausted { "done" } else { "running" };
            let step_info = if self.seq_length > 0 {
                format!(
                    "Step: {} / {}  [{}]",
                    self.current_step, self.seq_length, status
                )
            } else {
                format!("Step: {}  [{}]", self.current_step, status)
            };

            ui.horizontal(|ui| {
                ui.strong(&self.generator_name);
                ui.separator();
                ui.label(&step_info);
                ui.separator();
                let dim_info = match &self.frame_buffer.dimension {
                    CaDimension::OneD { width } => format!("1D (w={})", width),
                    CaDimension::TwoD { rows, cols, d_state } => {
                        format!("2D ({}×{}, d={})", rows, cols, d_state)
                    }
                    CaDimension::ThreeD { depth, rows, cols, d_state } => {
                        format!("3D ({}×{}×{}, d={})", depth, rows, cols, d_state)
                    }
                };
                ui.label(&dim_info);
            });
            ui.separator();

            // 渲染区域
            let ctx = ui.ctx().clone();
            match &mut self.renderer {
                Renderer::OneD(r) => {
                    r.render(ui, &ctx, &self.frame_buffer, self.current_step, &self.color_map);
                }
                Renderer::TwoD(r) => {
                    r.render(ui, &ctx, &self.frame_buffer, self.current_step, &self.color_map);
                }
                Renderer::ThreeD(r) => {
                    r.render(ui, &ctx, &self.frame_buffer, self.current_step, &self.color_map);
                }
            }
        });
    }
}

/// 启动可视化窗口
pub fn launch_viewer(args: ViewArgs) -> i32 {
    // 1. 初始化注册表
    let mut registry = GeneratorRegistry::new();
    if let Err(e) = register_all_generators(&mut registry) {
        eprintln!("Error: failed to register generators: {}", e);
        return 2;
    }

    // 2. 构建 GenParams
    let gen_params = match build_gen_params(&args) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: {}", e);
            return 2;
        }
    };

    // 3. 实例化生成器
    let generator = match registry.instantiate(&args.generator, &gen_params.extensions) {
        Ok(g) => g,
        Err(e) => {
            eprintln!(
                "Error: failed to instantiate generator '{}': {}",
                args.generator, e
            );
            return 2;
        }
    };

    // 4. 确定维度和 d_state
    let dimension = match infer_dimension(&args) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error: {}", e);
            return 2;
        }
    };

    let d_state = infer_d_state(&args);

    // 5. 生成流
    let iterator = match generator.generate_stream(args.seed, &gen_params) {
        Ok(iter) => iter,
        Err(e) => {
            eprintln!("Error: failed to generate stream: {}", e);
            return 2;
        }
    };

    // 6. 创建帧缓冲并启动后台线程
    let frame_buffer = CaFrameBuffer::new(dimension);
    frame_buffer.spawn_generator_thread(iterator);

    let color_map = CaColorMap::new(d_state);
    let generator_name = args.generator.clone();
    let seq_length = args.steps;
    let speed_ms = args.speed;

    // 7. 创建 App 并运行
    let app = CaViewApp::new(
        frame_buffer,
        color_map,
        generator_name,
        seq_length,
        speed_ms,
    );

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 700.0])
            .with_min_inner_size([400.0, 300.0])
            .with_title("StructGen CA Viewer"),
        ..Default::default()
    };

    if let Err(e) = eframe::run_native(
        "StructGen CA Viewer",
        native_options,
        Box::new(move |_cc| Ok(Box::new(app))),
    ) {
        eprintln!("Error: eframe failed: {}", e);
        return 2;
    }

    0
}

/// 从 ViewArgs 构建 GenParams
fn build_gen_params(args: &ViewArgs) -> Result<GenParams, String> {
    let mut extensions = std::collections::HashMap::new();

    // 规则参数
    if let Some(rule) = &args.rule {
        // 尝试解析为数字（1D Wolfram rule）
        if let Ok(rule_num) = rule.parse::<u64>() {
            extensions.insert("rule".into(), serde_json::Value::Number(rule_num.into()));
        } else {
            // 作为预设名称传入
            extensions.insert("preset".into(), serde_json::Value::String(rule.clone()));
        }
    }

    // 边界条件
    extensions.insert(
        "boundary".into(),
        serde_json::Value::String(args.boundary.clone()),
    );

    // 初始化模式
    extensions.insert(
        "init_mode".into(),
        serde_json::Value::String(args.init.clone()),
    );

    // 维度特定参数
    let grid_size;
    let grid_size_3d;

    match args.generator.as_str() {
        "ca" | "cellular_automaton" => {
            extensions.insert(
                "width".into(),
                serde_json::Value::Number(args.width.into()),
            );
            grid_size = None;
            grid_size_3d = None;
        }
        "ca2d" | "cellular_automaton_2d" => {
            grid_size = Some(GridSize {
                rows: args.rows,
                cols: args.cols,
            });
            grid_size_3d = None;
        }
        "ca3d" | "cellular_automaton_3d" => {
            grid_size = None;
            grid_size_3d = Some(GridSize3D {
                depth: args.depth,
                rows: args.rows,
                cols: args.cols,
            });
        }
        _ => {
            grid_size = None;
            grid_size_3d = None;
        }
    }

    Ok(GenParams {
        seq_length: args.steps,
        grid_size,
        grid_size_3d,
        extensions,
    })
}

/// 从 ViewArgs 推断 CA 维度
fn infer_dimension(args: &ViewArgs) -> Result<CaDimension, String> {
    match args.generator.as_str() {
        "ca" | "cellular_automaton" => Ok(CaDimension::OneD { width: args.width }),
        "ca2d" | "cellular_automaton_2d" => Ok(CaDimension::TwoD {
            rows: args.rows,
            cols: args.cols,
            d_state: infer_d_state(args),
        }),
        "ca3d" | "cellular_automaton_3d" => Ok(CaDimension::ThreeD {
            depth: args.depth,
            rows: args.rows,
            cols: args.cols,
            d_state: infer_d_state(args),
        }),
        _ => Err(format!(
            "unknown generator '{}' for CA view (expected: ca, ca2d, ca3d)",
            args.generator
        )),
    }
}

/// 推断状态数（从规则参数推断，默认为2）
fn infer_d_state(args: &ViewArgs) -> u8 {
    if let Some(rule) = &args.rule {
        match rule.as_str() {
            "wireworld" => 4,
            "cyclic" | "cyclic_3d" => 16,
            _ => 2,
        }
    } else {
        2
    }
}
