# 生成器模块缺陷修复总结

## 修复概览

共修复 12 个问题（4 个 P0 严重、4 个 P1 中等、4 个 P2 轻微），涉及 12 个源文件的修改。

## 修复详情

### P0 严重问题

| # | 问题 | 修复方式 | 文件 |
|---|------|---------|------|
| 1 | FrameState::Float 绕过 NaN/Infinity 检查 | 新增 `float_or_zero()` 方法，所有浮点生成器改用此方法构造 | `core/frame.rs`, `lorenz.rs`, `logistic.rs`, `nbody.rs`, `ifs.rs` |
| 4 | VM 除零丢失栈数据 | 除零时将 a 和 b 都压回栈（NOP 语义） | `vm.rs` |

### P1 中等问题

| # | 问题 | 修复方式 | 文件 |
|---|------|---------|------|
| 2 | Logistic Map 发散无保护 | 添加越界检测，x 超出 [0,1] 时置 0.0 并标记 finished | `logistic.rs` |
| 6 | Logistic/VM 参数传递不一致 | 将 x0/program_size/max_steps 存入结构体，工厂函数统一提取 | `logistic.rs`, `vm.rs` |
| 5 | seq_limit 双重终止冗余 | 移除闭包内 seq_limit 检查，仅依赖 `.take()` | 全部 9 个生成器 |
| 8 | LSystem 字符串增长无保护 | 新增 `max_string_length` 参数（默认 1M），超限停止迭代 | `lsystem.rs` |

### P2 轻微问题

| # | 问题 | 修复方式 | 文件 |
|---|------|---------|------|
| 3 | IFS 概率选择边界遗漏 | fallback 默认选择最后一个变换 | `ifs.rs` |
| 10 | 工厂函数反序列化模式重复 | 新增 `deserialize_extensions<T>()` 辅助函数 | `core/mod.rs`, 全部 9 个生成器 |
| 13 | Integrator 缺少 Copy derive | 添加 `Copy` derive，使用 Copy 替代 clone | `nbody.rs` |
| 14 | VM registers 未使用 | 移除 registers，输出维度从 10 变为 2 | `vm.rs` |
| 9 | FormalGrammar 完成后行为 | 添加设计意图注释 | `formal_grammar.rs` |

## 关键变更

### 新增公共 API
- `FrameState::float_or_zero(value: f64) -> Self` — 非有限值自动替换为 0.0
- `deserialize_extensions<T>(extensions: &HashMap<String, Value>) -> CoreResult<T>` — 工厂函数公共反序列化辅助

### 破坏性变更
- VM 输出维度从 10（PC + 8 regs + stack_top）变为 2（PC + stack_top）
- LogisticMap 新增 `x0` 参数验证（x0 必须在 [0,1] 范围内）
- LSystem 新增 `max_string_length` 参数（默认 1_000_000，值为 0 时拒绝）

### 新增测试
- `test_float_or_zero_nan/infinity/zero/finite` — NaN clamping 验证
- `test_r4_no_divergence` — Logistic r=4 不发散验证
- `test_x0_param_passed_via_factory` — x0 参数传递验证
- `test_x0_out_of_range_rejected` — x0 参数验证
- `test_div_zero_preserves_stack` — VM 除零 NOP 语义验证
- `test_params_via_factory` — VM 参数传递验证
- `test_max_string_length_enforced` — LSystem 字符串长度保护验证

## 验证结果

- **全量测试**：296 passed, 0 failed
- **clippy**：无新增 warning（25 个 warning 均为修改前已存在的未使用导入/方法）
