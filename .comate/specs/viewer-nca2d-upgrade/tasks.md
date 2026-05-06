# 查看器 NCA2D 支持升级任务计划

- [x] Task 1: 扩展 ViewArgs 和 ViewCliArgs 结构体，添加 NCA2D 专有参数
    - 1.1: 在 `src/view/mod.rs` 的 `ViewArgs` 中新增字段：`d_state: u8`、`n_groups: u8`、`temperature: f64`、`hidden_dim: usize`、`conv_features: usize`、`identity_bias: f64`
    - 1.2: 在 `src/main.rs` 的 `ViewCliArgs` 中新增对应的 CLI 参数定义（含默认值和帮助文本）
    - 1.3: 更新 `From<ViewCliArgs> for ViewArgs` 转换实现，映射新增字段
    - 1.4: 更新 `ViewCliArgs` 的 generator 帮助文本为 `ca | ca2d | ca3d | nca2d`

- [x] Task 2: 扩展 CaDimension::TwoD 添加 n_groups 字段
    - 2.1: 在 `src/view/frame_buffer.rs` 的 `CaDimension::TwoD` 变体中添加 `n_groups: u8` 字段
    - 2.2: 确认 `CaFrameBuffer::new()` 的 match 分支无需修改（仅读取 `dimension` 引用）

- [x] Task 3: 扩展 app.rs 核心逻辑支持 NCA2D
    - 3.1: 修改 `infer_dimension()` — 添加 `"nca2d" | "neural_cellular_automaton_2d"` 分支，返回 `CaDimension::TwoD { rows, cols, d_state, n_groups }`
    - 3.2: 修改 `infer_dimension()` 中 `ca2d` 分支，补充 `n_groups: 1`
    - 3.3: 修改 `build_gen_params()` — 添加 NCA2D 分支，传入 rows/cols/d_state/n_groups/temperature/hidden_dim/conv_features/identity_bias 到 extensions，并设置 grid_size
    - 3.4: 修改 `infer_d_state()` — 当生成器为 NCA2D 时直接返回 `args.d_state`，跳过规则名推断

- [x] Task 4: 扩展 Render2D 支持多通道 NCA2D 可视化
    - 4.1: 在 `Render2D` 结构体中添加 `n_groups: u8` 和 `current_group: u8` 字段
    - 4.2: 修改 `Render2D::new()` 签名，接收 `n_groups` 参数
    - 4.3: 当 `n_groups > 1` 时，在渲染区域上方添加分组选择滑块（参考 Render3D 的深度层选择器）
    - 4.4: 修改像素构建逻辑 — 当 `n_groups > 1` 时，按 `values[i * n_groups + current_group]` 提取对应通道数据；`n_groups == 1` 时保持原有逻辑
    - 4.5: 更新 `CaViewApp::new()` 中 `Renderer::TwoD` 的构造调用，传入 `n_groups`

- [x] Task 5: 编译验证与测试
    - 5.1: 运行 `cargo build --features view` 确认编译通过
    - 5.2: 运行 `cargo test` 确认现有测试通过
    - 5.3: 运行 `cargo test --features view` 确认 view 功能测试通过
