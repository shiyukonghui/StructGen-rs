# 生成器模块缺陷修复规格说明

## 背景

基于代码审查，发现生成器模块存在 14 个问题（4 个严重、5 个中等、5 个轻微）。本文档聚焦于需要代码修改的 13 个问题（#12 模偏差为文档问题，不影响正确性），按优先级分批修复。

## 修复范围

### P0 - 严重问题（正确性风险）

#### 问题 #1：FrameState::Float 绕过 NaN/Infinity 检查

**现状**：`FrameState::Float(f64)` 是公开变体，`FrameState::float()` 有 NaN/Infinity 检查但所有生成器都直接使用 `FrameState::Float(x)` 构造，绕过了检查。

**风险**：Lorenz/Logistic/IFS/NBody 等浮点生成器可能产生非有限值，导致序列化失败或下游处理器 panic。

**修复方案**：
1. 将 `FrameState::Float` 变体的可见性降为 `pub(crate)`，外部通过 `FrameState::float()` 构造
2. 所有生成器中使用 `FrameState::float(x)?` 替代 `FrameState::Float(x)`
3. 在 `generate_stream` 闭包内对非有限值进行 clamping 处理（将 NaN/Infinity 替换为 0.0），而非返回错误中断流

**影响文件**：
- `src/core/frame.rs` — 将 `Float(f64)` 改为 `pub(crate)` 变体，新增 `float_or_zero` 便捷方法
- `src/generators/lorenz.rs` — 使用 `FrameState::float()` 构造
- `src/generators/logistic.rs` — 使用 `FrameState::float()` 构造
- `src/generators/nbody.rs` — 使用 `FrameState::float()` 构造
- `src/generators/ifs.rs` — 使用 `FrameState::float()` 构造

#### 问题 #4：VM 除零处理逻辑错误

**现状**：`vm.rs:193-206`，当除数为 0 时弹出 `a` 和 `b`，但只压回 `b`（0），`a` 丢失。

**修复方案**：除零时视为 NOP —— 将 `a` 和 `b` 都压回栈，PC 前进一步。语义清晰：除零 = 跳过此指令。

**影响文件**：
- `src/generators/vm.rs` — 修改 `OpCode::Div` 分支逻辑

### P1 - 中等问题

#### 问题 #2：Logistic Map 发散无保护

**现状**：`x = r * x * (1.0 - x)` 在 r 接近 4 时可能因浮点误差导致 x 越出 [0,1]，产生 NaN/Infinity。文档注释提到"越界时输出 0.0 并停止产生帧"但代码未实现。

**修复方案**：每次迭代后将 x clamping 到 [0.0, 1.0]；若 clamping 前检测到 x 不在 [0,1] 范围，将 x 置为 0.0 并标记结束（`finished = true`），符合文档描述的行为。

**影响文件**：
- `src/generators/logistic.rs` — 在迭代后添加越界检测与 clamping

#### 问题 #6：Logistic/VM 参数传递不一致

**现状**：`logistic_factory` 从 `extensions` 读取 `r` 存入结构体，但 `x0` 只能从 `params.extensions["logistic"]` 读取。而调度器调用链中，`extensions` 仅传给工厂函数，`generate_stream` 收到的 `params.extensions` 来自 YAML manifest 的另一个字段。导致 `x0` 永远为 None。VM 的 `VMParams` 同理。

**修复方案**：将所有生成器特有参数统一在工厂函数中提取并存入结构体，`generate_stream` 不再从 `params.extensions` 读取。具体：
- LogisticMap：在结构体中增加 `x0: Option<f64>` 字段，工厂函数中读取
- AlgorithmVM：在结构体中增加 `program_size` 和 `max_steps` 字段，工厂函数中读取

**影响文件**：
- `src/generators/logistic.rs` — 结构体增加 x0，generate_stream 使用结构体字段
- `src/generators/vm.rs` — 结构体增加 program_size/max_steps，generate_stream 使用结构体字段

#### 问题 #5：seq_limit 双重终止冗余

**现状**：每个生成器的闭包内检查 `step_counter >= seq_limit` + 闭包外 `.take(seq_limit)` 功能重复。

**修复方案**：移除闭包内的 `step_counter >= seq_limit` 检查，仅依赖 `.take(seq_limit)` 控制有限流。对于 `seq_limit == 0`（无限流）不添加 `.take()`。这样闭包逻辑更简洁，且 `.take()` 是标准库优化的适配器。

**注意**：LSystem 和 FormalGrammar 有自然终止条件（`finished` 标志），它们的 `step_counter` 检查需保留自然终止逻辑。

**影响文件**：
- `src/generators/ca.rs` — 移除闭包内 seq_limit 检查
- `src/generators/lorenz.rs` — 同上
- `src/generators/logistic.rs` — 同上
- `src/generators/nbody.rs` — 同上
- `src/generators/lsystem.rs` — 同上（保留 finished 检查）
- `src/generators/vm.rs` — 同上（保留 halted/max_steps 检查）
- `src/generators/boolean_network.rs` — 同上
- `src/generators/ifs.rs` — 同上
- `src/generators/formal_grammar.rs` — 同上（保留 finished 检查）

#### 问题 #8：LSystem 字符串增长无保护

**现状**：指数增长的 L-System 规则可导致字符串迅速膨胀耗尽内存。

**修复方案**：在 `LSystemParams` 中新增 `max_string_length` 参数（默认 1_000_000），闭包内在每次替换后检查字符串长度，超限时停止迭代（`finished = true`）。

**影响文件**：
- `src/generators/lsystem.rs` — 新增参数和长度检查

### P2 - 轻微问题

#### 问题 #3：IFS 概率选择边界遗漏

**现状**：浮点累积误差可能导致最后一个变换永远不被选中。

**修复方案**：循环结束后 fallback 到最后一个变换索引。

**影响文件**：
- `src/generators/ifs.rs` — 添加 fallback 逻辑

#### 问题 #10：工厂函数反序列化模式重复

**现状**：9 个工厂函数中 `if extensions.is_empty() { Default } else { serialize + deserialize }` 模式重复。

**修复方案**：在 `core/` 中新增 `deserialize_extensions<T: Default + DeserializeOwned>()` 辅助函数，所有工厂函数改用此辅助函数。

**影响文件**：
- `src/core/mod.rs` — 新增 pub 函数 `deserialize_extensions`
- 所有 9 个生成器文件 — 替换反序列化逻辑

#### 问题 #13：Integrator enum 应使用 Copy

**现状**：`Integrator` 使用 `Clone`，但两个变体均无数据，可以使用 `Copy`。

**修复方案**：将 `#[derive(Clone)]` 改为 `#[derive(Copy, Clone)]`，闭包中使用 Copy 代替 clone。

**影响文件**：
- `src/generators/nbody.rs` — 修改 derive 和使用方式

#### 问题 #14：VM registers 未使用

**现状**：`registers: [i64; 8]` 声明但从未修改，每帧输出全零值。

**修复方案**：移除 registers 相关代码。当前指令集无 Store/Load 指令，registers 是死代码。每帧输出维度从 10（PC + 8 regs + stack_top）变为 2（PC + stack_top）。

**影响文件**：
- `src/generators/vm.rs` — 移除 registers，调整输出维度

#### 问题 #9：FormalGrammar 完成后持续输出相同帧

**现状**：推导完成后持续输出完全相同的帧直到 seq_limit。这是有意为之的设计（保持流活跃），但语义上有歧义。

**修复方案**：保持当前行为，但在代码中添加注释说明设计意图：推导完成后重复输出最终结果以维持时间序列的连续性。这是行为决策而非 bug，不需要修改逻辑。

**影响文件**：
- `src/generators/formal_grammar.rs` — 添加注释

## 不修复项

- **#11 SeedRng Xorshift64 统计质量**：当前场景不需要高统计质量，保持简单 PRNG 即可。添加文档注释说明限制。
- **#12 SeedRng 模偏差**：对当前用途影响极小，不修改。

## 测试要求

每个修复必须包含对应的测试验证：
- P0 修复：必须有针对性测试（如 NaN clamping 测试、除零恢复测试）
- P1/P2 修复：至少验证不破坏现有测试
- 全部修复完成后运行 `cargo test` 确保零回归

## 预期结果

- 所有生成器产出的 `FrameState::Float` 值保证有限
- Logistic Map 在 r=4 边界条件下不会发散
- VM 除零不再丢失栈数据
- 参数传递路径统一一致
- 代码重复减少、风格统一
