# 查看器 NCA2D 支持升级规划

## 需求场景

当前查看器（Viewer）仅支持 `ca`/`ca2d`/`ca3d` 三种生成器的可视化，而 `nca2d`（神经元胞自动机 2D）生成器已在生成器注册表中注册，但查看器无法使用它——`infer_dimension` 会返回 "unknown generator" 错误，`build_gen_params` 缺少 NCA2D 参数分支。

NCA2D 本质产生 **2D 网格数据**，与 CA2D 的数据结构基本一致（`rows × cols` 离散状态网格），但有两个关键差异：
1. **状态数默认为 10**（CA2D 默认 2），需要正确的颜色映射
2. **支持多通道（n_groups）**，每帧数据为 `rows × cols × n_groups` 个整数值，需要分组可视化

## 架构与技术方案

### 整体思路

在现有 2D 渲染管线基础上扩展 NCA2D 支持，遵循以下原则：
- NCA2D 复用 `CaDimension::TwoD` 和 `Render2D`，不引入新的维度类型
- 为 `CaDimension::TwoD` 添加 `n_groups` 字段，使 Render2D 能识别多通道数据
- 为 `Render2D` 添加分组选择器（类似 3D 的深度层选择器），当 `n_groups > 1` 时可切换显示不同通道
- CLI 参数扩展，添加 NCA2D 专有参数

### 数据流路径

```
CLI 参数 (ViewCliArgs)
    ↓ From<ViewArgs>
ViewArgs (view/mod.rs)
    ↓ build_gen_params()
GenParams (含 NCA2D extensions)
    ↓ registry.instantiate()
NeuralCA2D 生成器实例
    ↓ generate_stream()
Iterator<SequenceFrame>  [values: rows × cols × n_groups]
    ↓ 后台线程 → CaFrameBuffer
CaFrameBuffer { dimension: TwoD { rows, cols, d_state, n_groups } }
    ↓ 渲染
Render2D (分组选择器 → 选取第 g 组 → ColorImage → 纹理)
```

## 受影响的文件

### 1. `src/view/mod.rs` — ViewArgs 结构体扩展
- **修改类型**：结构体字段新增
- **影响函数**：`ViewArgs` 结构体定义
- **新增字段**：`d_state`、`n_groups`、`temperature`、`hidden_dim`、`conv_features`、`identity_bias`

### 2. `src/main.rs` — CLI 参数扩展
- **修改类型**：CLI 参数定义 + 转换逻辑
- **影响函数**：`ViewCliArgs` 结构体、`From<ViewCliArgs> for ViewArgs`
- **新增参数**：与 ViewArgs 对应的 NCA2D 专有 CLI 参数
- **修改帮助文本**：generator 描述更新为 `ca | ca2d | ca3d | nca2d`

### 3. `src/view/frame_buffer.rs` — CaDimension 扩展
- **修改类型**：枚举变体字段新增
- **影响函数**：`CaDimension::TwoD` 变体、`CaFrameBuffer::new()` 中的 match 分支
- **新增字段**：`CaDimension::TwoD` 添加 `n_groups: u8`

### 4. `src/view/app.rs` — 核心逻辑修改
- **修改类型**：函数逻辑扩展
- **影响函数**：
  - `infer_dimension()` — 新增 nca2d 分支
  - `build_gen_params()` — 新增 nca2d 分支，传入所有 NCA2D 参数
  - `infer_d_state()` — 新增 NCA2D 默认状态数处理
  - `CaViewApp::new()` — 传递 n_groups 给 CaDimension::TwoD

### 5. `src/view/render_2d.rs` — 多通道可视化
- **修改类型**：渲染逻辑扩展
- **影响函数**：`Render2D` 结构体、`Render2D::render()`、`Render2D::new()`
- **新增功能**：
  - `Render2D` 添加 `n_groups: u8`、`current_group: u8` 字段
  - 当 `n_groups > 1` 时显示分组选择滑块
  - 渲染时根据 `current_group` 从帧数据中提取对应通道

## 实现细节

### 1. ViewArgs 扩展

```rust
// src/view/mod.rs
pub struct ViewArgs {
    // ... 现有字段 ...
    /// NCA2D: 离散状态数（默认 10）
    pub d_state: u8,
    /// NCA2D: 通道/组数（默认 1）
    pub n_groups: u8,
    /// NCA2D: 采样温度（默认 1.0）
    pub temperature: f64,
    /// NCA2D: 隐藏层维度（默认 16）
    pub hidden_dim: usize,
    /// NCA2D: 卷积特征数（默认 4）
    pub conv_features: usize,
    /// NCA2D: 恒等偏置（默认 0.0）
    pub identity_bias: f64,
}
```

### 2. ViewCliArgs 扩展

```rust
// src/main.rs
pub struct ViewCliArgs {
    // ... 现有字段 ...
    /// NCA2D: 离散状态数
    #[arg(long, default_value = "10")]
    pub d_state: u8,
    /// NCA2D: 通道/组数
    #[arg(long, default_value = "1")]
    pub n_groups: u8,
    /// NCA2D: 采样温度
    #[arg(long, default_value = "1.0")]
    pub temperature: f64,
    /// NCA2D: 隐藏层维度
    #[arg(long, default_value = "16")]
    pub hidden_dim: usize,
    /// NCA2D: 卷积特征数
    #[arg(long, default_value = "4")]
    pub conv_features: usize,
    /// NCA2D: 恒等偏置
    #[arg(long, default_value = "0.0")]
    pub identity_bias: f64,
}
```

### 3. CaDimension::TwoD 扩展

```rust
// src/view/frame_buffer.rs
pub enum CaDimension {
    OneD { width: usize },
    TwoD { rows: usize, cols: usize, d_state: u8, n_groups: u8 },  // 新增 n_groups
    ThreeD { depth: usize, rows: usize, cols: usize, d_state: u8 },
}
```

### 4. infer_dimension 扩展

```rust
// src/view/app.rs
fn infer_dimension(args: &ViewArgs) -> Result<CaDimension, String> {
    match args.generator.as_str() {
        "ca" | "cellular_automaton" => Ok(CaDimension::OneD { width: args.width }),
        "ca2d" | "cellular_automaton_2d" => Ok(CaDimension::TwoD {
            rows: args.rows,
            cols: args.cols,
            d_state: infer_d_state(args),
            n_groups: 1,  // CA2D 固定为单通道
        }),
        "ca3d" | "cellular_automaton_3d" => Ok(CaDimension::ThreeD { ... }),
        "nca2d" | "neural_cellular_automaton_2d" => Ok(CaDimension::TwoD {
            rows: args.rows,
            cols: args.cols,
            d_state: args.d_state,
            n_groups: args.n_groups,
        }),
        _ => Err(format!(
            "unknown generator '{}' for CA view (expected: ca, ca2d, ca3d, nca2d)",
            args.generator
        )),
    }
}
```

### 5. build_gen_params NCA2D 分支

```rust
"nca2d" | "neural_cellular_automaton_2d" => {
    extensions.insert("rows", serde_json::Value::Number(args.rows.into()));
    extensions.insert("cols", serde_json::Value::Number(args.cols.into()));
    extensions.insert("d_state", serde_json::json!(args.d_state));
    extensions.insert("n_groups", serde_json::json!(args.n_groups));
    extensions.insert("temperature", serde_json::json!(args.temperature));
    extensions.insert("hidden_dim", serde_json::Value::Number(args.hidden_dim.into()));
    extensions.insert("conv_features", serde_json::Value::Number(args.conv_features.into()));
    extensions.insert("identity_bias", serde_json::json!(args.identity_bias));
    grid_size = Some(GridSize { rows: args.rows, cols: args.cols });
    grid_size_3d = None;
}
```

### 6. Render2D 多通道支持

```rust
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
            rows, cols, n_groups,
            current_group: 0,
            texture: None,
        }
    }

    pub fn render(&mut self, ui, ctx, buffer, current_step, color_map) {
        // 当 n_groups > 1 时，显示分组选择滑块
        if self.n_groups > 1 {
            ui.horizontal(|ui| {
                ui.label("Group:");
                let mut g = self.current_group as usize;
                ui.add(egui::Slider::new(&mut g, 0..=(self.n_groups - 1) as usize).text(""));
                self.current_group = g as u8;
                ui.label(format!("({}/{})", g + 1, self.n_groups));
            });
        }

        // 渲染逻辑：从帧数据中提取 current_group 对应的值
        // NCA2D 布局: values[(r * cols + c) * n_groups + g]
        for i in 0..slice_len {
            let src_idx = i * (self.n_groups as usize) + (self.current_group as usize);
            let color = if src_idx < frame.state.values.len() {
                match &frame.state.values[src_idx] {
                    FrameState::Integer(v) => color_map.color_for_state(*v as u8),
                    _ => Color32::BLACK,
                }
            } else {
                Color32::BLACK
            };
            // ...
        }
    }
}
```

### 7. infer_d_state 更新

NCA2D 的 d_state 不从规则名推断，而是直接使用 `args.d_state`。当生成器为 NCA2D 时，跳过规则名匹配逻辑：

```rust
fn infer_d_state(args: &ViewArgs) -> u8 {
    // NCA2D 使用显式 d_state 参数
    if args.generator == "nca2d" || args.generator == "neural_cellular_automaton_2d" {
        return args.d_state;
    }
    // CA 系列从规则名推断
    if let Some(rule) = &args.rule {
        match rule.as_str() {
            "wireworld" => 4,
            "cyclic" | "cyclic_3d" => 14,
            _ => 2,
        }
    } else {
        2
    }
}
```

## 边界条件与异常处理

1. **n_groups = 0**：ViewCliArgs 中 `n_groups` 的默认值为 1，且 clap 不会产生 0 值（因为 u8 的默认值是 1）。但若用户手动传入 `--n-groups 0`，nca2d_factory 会返回校验错误，无需在查看器层额外处理。

2. **d_state = 0 或 1**：同样由 nca2d_factory 校验拒绝，查看器不重复校验。

3. **temperature < 0**：由 nca2d_factory 校验拒绝。

4. **n_groups > 1 时的帧数据大小**：当前 Render2D 假设帧数据大小为 `rows × cols`。NCA2D 的帧数据大小为 `rows × cols × n_groups`。当 n_groups = 1 时行为不变；n_groups > 1 时按 `current_group` 选取对应通道。

5. **NCA2D 参数对 CA 系列生成器无效**：新增的 CLI 参数（d_state, n_groups 等）仅当选择 nca2d 生成器时才有意义。对 ca/ca2d/ca3d 无副作用，因为这些参数不会被写入它们的 extensions。

6. **CaDimension::TwoD 新增 n_groups 字段的影响**：所有构造 `CaDimension::TwoD` 的地方都需要添加 `n_groups: 1`（CA2D 场景）。受影响位置：
   - `infer_dimension()` 中的 `ca2d` 分支
   - 任何测试代码（如有）

## 预期结果

1. 用户可通过 `structgen-rs view --generator nca2d` 启动 NCA2D 实时可视化
2. 支持所有 NCA2D 专有参数的 CLI 传入
3. NCA2D 的默认 10 状态颜色映射正确（HSV 色轮分布）
4. 多通道 NCA2D（n_groups > 1）可通过分组选择器切换查看
5. 现有 ca/ca2d/ca3d 查看功能完全不受影响
