# 生成器模块缺陷修复任务计划

- [x] Task 1: 修复 FrameState::Float 绕过 NaN/Infinity 检查 (#1)
    - 1.1: 将 `FrameState::Float` 变体可见性降为 `pub(crate)`，新增 `float_or_zero()` 便捷方法
    - 1.2: 修改 lorenz.rs 使用 `FrameState::float_or_zero()`
    - 1.3: 修改 logistic.rs 使用 `FrameState::float_or_zero()`
    - 1.4: 修改 nbody.rs 使用 `FrameState::float_or_zero()`
    - 1.5: 修改 ifs.rs 使用 `FrameState::float_or_zero()`
    - 1.6: 添加 NaN/Infinity clamping 的单元测试

- [x] Task 2: 修复 VM 除零处理逻辑错误 (#4)
    - 2.1: 修改 `OpCode::Div` 分支，除零时将 a 和 b 都压回栈（NOP 语义）
    - 2.2: 添加除零恢复测试

- [x] Task 3: 修复 Logistic Map 发散无保护 (#2)
    - 3.1: 在迭代后添加越界检测：x 超出 [0,1] 时置为 0.0 并标记 `finished = true`
    - 3.2: 添加 r=4 边界条件下长时间运行不发散的测试
    - 3.3: 添加越界终止行为的测试

- [x] Task 4: 修复 Logistic/VM 参数传递不一致 (#6)
    - 4.1: LogisticMap 结构体增加 `x0: Option<f64>` 字段，工厂函数中提取并存入
    - 4.2: AlgorithmVM 结构体增加 `program_size` 和 `max_steps` 字段，工厂函数中提取并存入
    - 4.3: 移除 generate_stream 中对 params.extensions 的参数读取
    - 4.4: 更新相关测试以通过 extensions 传递参数

- [x] Task 5: 消除 seq_limit 双重终止冗余 (#5)
    - 5.1: ca.rs — 移除闭包内 seq_limit 检查，依赖 .take()
    - 5.2: lorenz.rs — 同上
    - 5.3: logistic.rs — 同上（保留 finished 检查）
    - 5.4: nbody.rs — 同上
    - 5.5: boolean_network.rs — 同上
    - 5.6: ifs.rs — 同上
    - 5.7: lsystem.rs — 同上（保留 finished 检查）
    - 5.8: vm.rs — 同上（保留 halted/max_steps 检查）
    - 5.9: formal_grammar.rs — 同上（保留 finished 检查）

- [x] Task 6: LSystem 字符串增长保护 (#8)
    - 6.1: LSystemParams 新增 `max_string_length` 字段（默认 1_000_000）
    - 6.2: 闭包内每次替换后检查字符串长度，超限时 finished = true
    - 6.3: 添加字符串长度溢出终止的测试

- [x] Task 7: IFS 概率选择边界遗漏修复 (#3)
    - 7.1: 循环后添加 fallback 到最后一个变换索引
    - 7.2: 添加概率选择边界条件的测试

- [x] Task 8: 提取工厂函数公共反序列化逻辑 (#10)
    - 8.1: 在 core/mod.rs 新增 `deserialize_extensions<T>()` 辅助函数
    - 8.2: 替换所有 9 个工厂函数中的重复反序列化代码

- [x] Task 9: Integrator enum 添加 Copy derive (#13)
    - 9.1: 将 `#[derive(Clone)]` 改为 `#[derive(Copy, Clone)]`
    - 9.2: 闭包中使用 Copy 代替 clone

- [x] Task 10: 移除 VM 未使用的 registers (#14)
    - 10.1: 移除 registers 相关声明和输出代码
    - 10.2: 每帧输出维度从 10 变为 2（PC + stack_top）
    - 10.3: 更新相关测试的维度断言

- [x] Task 11: FormalGrammar 完成后行为注释 (#9)
    - 11.1: 在 `finished` 分支添加设计意图注释

- [x] Task 12: 运行全量测试确认零回归
    - 12.1: 执行 `cargo test` 并确认全部通过
    - 12.2: 执行 `cargo clippy` 确认无新警告
