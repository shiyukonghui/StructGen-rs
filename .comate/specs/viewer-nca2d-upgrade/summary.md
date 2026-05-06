# 查看器 NCA2D 支持升级 — 完成总结

## 变更概述

为 StructGen-rs 的实时查看器（Viewer）添加了 NCA2D（2D 神经元胞自动机）生成器的可视化支持。此前查看器仅支持 ca/ca2d/ca3d 三种生成器，NCA2D 会因 "unknown generator" 错误无法启动。

## 修改文件清单

### 1. `src/view/mod.rs`
- `ViewArgs` 结构体新增 6 个字段：`d_state`、`n_groups`、`temperature`、`hidden_dim`、`conv_features`、`identity_bias`

### 2. `src/main.rs`
- `ViewCliArgs` 结构体新增对应的 6 个 CLI 参数（含默认值和帮助文本）
- `From<ViewCliArgs> for ViewArgs` 转换实现映射新增字段
- generator 帮助文本更新为 `ca | ca2d | ca3d | nca2d`

### 3. `src/view/frame_buffer.rs`
- `CaDimension::TwoD` 变体新增 `n_groups: u8` 字段

### 4. `src/view/app.rs`
- `infer_dimension()`: 新增 `nca2d`/`neural_cellular_automaton_2d` 分支，返回 `CaDimension::TwoD` 并传入 `d_state`/`n_groups`；`ca2d` 分支补充 `n_groups: 1`
- `build_gen_params()`: 新增 NCA2D 分支，传入 rows/cols/d_state/n_groups/temperature/hidden_dim/conv_features/identity_bias
- `infer_d_state()`: NCA2D 直接使用 `args.d_state`，跳过规则名推断
- `CaViewApp::new()`: 传递 `n_groups` 给 `Render2D::new()`
- 信息栏显示：当 `n_groups > 1` 时额外显示组数

### 5. `src/view/render_2d.rs`
- `Render2D` 结构体新增 `n_groups: u8` 和 `current_group: u8` 字段
- `Render2D::new()` 签名新增 `n_groups` 参数
- 当 `n_groups > 1` 时，渲染区域上方显示分组选择滑块
- 像素构建逻辑：`n_groups > 1` 时按 `values[i * n_groups + current_group]` 提取对应通道数据；`n_groups == 1` 时保持原有逐像素逻辑

## 验证结果

- `cargo build --features view` 编译通过
- `cargo test` 全部 451 个测试通过
- `cargo test --features view` 全部 451 个测试通过

## 使用方式

```bash
# NCA2D 默认参数可视化
structgen-rs view --generator nca2d --seed 42 --steps 200

# 自定义参数
structgen-rs view --generator nca2d --rows 32 --cols 32 --d-state 5 --n-groups 2 --temperature 0.5 --steps 500

# 使用完整名称
structgen-rs view --generator neural_cellular_automaton_2d --d-state 8 --hidden-dim 32
```
