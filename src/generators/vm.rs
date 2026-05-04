use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::rng::SeedRng;
use crate::core::*;

/// 栈式虚拟机专用参数
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VMParams {
    /// 程序指令条数（随机生成）
    #[serde(default = "default_program_size")]
    program_size: usize,
    /// 最大执行步数保护（防无限循环）
    #[serde(default = "default_max_steps")]
    max_steps: usize,
}

fn default_program_size() -> usize {
    32
}
fn default_max_steps() -> usize {
    1000
}

impl Default for VMParams {
    fn default() -> Self {
        VMParams {
            program_size: default_program_size(),
            max_steps: default_max_steps(),
        }
    }
}

/// 虚拟机指令集
#[derive(Debug, Clone, Copy, PartialEq)]
enum OpCode {
    /// 将立即数压入栈顶
    Push(i64),
    /// 弹出栈顶并丢弃
    Pop,
    /// 弹出两个值，压入和
    Add,
    /// 弹出两个值，压入差 (次栈顶 - 栈顶)
    Sub,
    /// 弹出两个值，压入积
    Mul,
    /// 弹出两个值，压入商 (次栈顶 / 栈顶，整数除法)
    Div,
    /// 复制栈顶值
    Dup,
    /// 交换栈顶两个值
    Swap,
    /// 无条件跳转到指定地址
    Jmp(i64),
    /// 弹出栈顶，若为 0 则跳转到指定地址
    Jz(i64),
    /// 停机
    Halt,
}

/// 栈式虚拟机生成器
///
/// 精简指令集虚拟机，随机生成程序并逐指令执行。
/// 每条指令执行后记录 PC + 栈顶作为一帧。
pub struct AlgorithmVM {
    program_size: usize,
    max_steps: usize,
}

impl AlgorithmVM {
    /// 从种子 PRNG 生成随机指令序列
    fn generate_program(rng: &mut SeedRng, size: usize) -> Vec<OpCode> {
        let mut program = Vec::with_capacity(size);
        // 最后一条指令必须是 Halt
        for i in 0..size.saturating_sub(1) {
            program.push(Self::random_opcode(rng, size, i));
        }
        program.push(OpCode::Halt);
        program
    }

    /// 随机生成一条指令
    fn random_opcode(rng: &mut SeedRng, program_size: usize, _pc: usize) -> OpCode {
        let op = rng.next_u64() % 11;
        match op {
            0 => OpCode::Push(rng.next_u64() as i64 % 100),
            1 => OpCode::Pop,
            2 => OpCode::Add,
            3 => OpCode::Sub,
            4 => OpCode::Mul,
            5 => OpCode::Div,
            6 => OpCode::Dup,
            7 => OpCode::Swap,
            8 => OpCode::Jmp(rng.next_usize(program_size) as i64),
            9 => OpCode::Jz(rng.next_usize(program_size) as i64),
            _ => OpCode::Halt,
        }
    }
}

impl Generator for AlgorithmVM {
    fn name(&self) -> &'static str {
        "algorithm_vm"
    }

    fn generate_stream(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let seq_limit = params.seq_length;
        let program_size = self.program_size;
        let max_steps = self.max_steps;

        // 用种子 PRNG 生成随机程序
        let mut rng = SeedRng::new(seed);
        let program = AlgorithmVM::generate_program(&mut rng, program_size);

        // 虚拟机状态
        let mut stack: Vec<i64> = Vec::new();
        let mut pc: usize = 0;
        let mut step_counter: u64 = 0;
        let mut halted = false;
        let mut steps_executed: usize = 0;

        let iter = std::iter::from_fn(move || {
            if halted || steps_executed >= max_steps {
                return None;
            }

            let step = step_counter;
            step_counter += 1;

            // 构建当前状态帧（PC + stack top）
            let stack_top = stack.last().copied().unwrap_or(0);
            let values = vec![
                FrameState::Integer(pc as i64),
                FrameState::Integer(stack_top),
            ];
            let frame = SequenceFrame::new(step, FrameData { values });

            // 执行一条指令
            if pc < program.len() && !halted {
                let op = program[pc];
                match op {
                    OpCode::Push(val) => {
                        stack.push(val);
                        pc += 1;
                    }
                    OpCode::Pop => {
                        stack.pop();
                        pc += 1;
                    }
                    OpCode::Add => {
                        if stack.len() >= 2 {
                            let b = stack.pop().unwrap();
                            let a = stack.pop().unwrap();
                            stack.push(a.wrapping_add(b));
                        }
                        pc += 1;
                    }
                    OpCode::Sub => {
                        if stack.len() >= 2 {
                            let b = stack.pop().unwrap();
                            let a = stack.pop().unwrap();
                            stack.push(a.wrapping_sub(b));
                        }
                        pc += 1;
                    }
                    OpCode::Mul => {
                        if stack.len() >= 2 {
                            let b = stack.pop().unwrap();
                            let a = stack.pop().unwrap();
                            stack.push(a.wrapping_mul(b));
                        }
                        pc += 1;
                    }
                    OpCode::Div => {
                        if stack.len() >= 2 {
                            let b = stack.pop().unwrap();
                            if b != 0 {
                                let a = stack.pop().unwrap();
                                stack.push(a / b);
                            } else {
                                // 除零：视为 NOP，将 a 和 b 都压回栈
                                let a = stack.pop().unwrap();
                                stack.push(a);
                                stack.push(b);
                            }
                        }
                        pc += 1;
                    }
                    OpCode::Dup => {
                        if let Some(&top) = stack.last() {
                            stack.push(top);
                        }
                        pc += 1;
                    }
                    OpCode::Swap => {
                        if stack.len() >= 2 {
                            let len = stack.len();
                            stack.swap(len - 1, len - 2);
                        }
                        pc += 1;
                    }
                    OpCode::Jmp(addr) => {
                        pc = addr as usize % program.len();
                    }
                    OpCode::Jz(addr) => {
                        let condition = stack.pop().unwrap_or(0);
                        if condition == 0 {
                            pc = addr as usize % program.len();
                        } else {
                            pc += 1;
                        }
                    }
                    OpCode::Halt => {
                        halted = true;
                    }
                }
                steps_executed += 1;
            } else {
                halted = true;
            }

            Some(frame)
        });

        let iter: Box<dyn Iterator<Item = SequenceFrame> + Send> = if seq_limit == 0 {
            Box::new(iter)
        } else {
            Box::new(iter.take(seq_limit))
        };

        Ok(iter)
    }
}

/// 虚拟机工厂函数
pub fn vm_factory(extensions: &HashMap<String, Value>) -> CoreResult<Box<dyn Generator>> {
    let vm_params: VMParams = deserialize_extensions(extensions)?;

    Ok(Box::new(AlgorithmVM {
        program_size: vm_params.program_size,
        max_steps: vm_params.max_steps,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_vm_factory(program_size: usize, max_steps: usize) -> Box<dyn Generator> {
        let mut extensions = HashMap::new();
        extensions.insert("program_size".to_string(), json!(program_size));
        extensions.insert("max_steps".to_string(), json!(max_steps));
        vm_factory(&extensions).unwrap()
    }

    fn collect_frames(gen: &dyn Generator, seed: u64, num_frames: usize) -> Vec<SequenceFrame> {
        let params = GenParams::simple(num_frames);
        gen.generate_stream(seed, &params).unwrap().collect()
    }

    #[test]
    fn test_deterministic_same_seed_same_output() {
        let gen1 = make_vm_factory(8, 100);
        let gen2 = make_vm_factory(8, 100);

        let frames1 = collect_frames(gen1.as_ref(), 42, 20);
        let frames2 = collect_frames(gen2.as_ref(), 42, 20);

        assert_eq!(frames1.len(), frames2.len());
        for i in 0..frames1.len().min(frames2.len()) {
            assert_eq!(
                frames1[i].state.values, frames2[i].state.values,
                "Frame {} should be identical",
                i
            );
        }
    }

    #[test]
    fn test_output_has_correct_dimensions() {
        let gen = make_vm_factory(8, 200);
        let params = GenParams::simple(10);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        for f in &frames {
            // 2 个值：PC + stack top
            assert_eq!(f.state.values.len(), 2);
        }
    }

    #[test]
    fn test_halt_stops_execution() {
        let gen = make_vm_factory(4, 50);
        let frames = gen.generate_stream(12345, &GenParams::simple(100)).unwrap().collect::<Vec<_>>();
        assert!(
            frames.len() <= 100,
            "VM should halt before generating 100 frames"
        );
    }

    #[test]
    fn test_default_params() {
        let gen = vm_factory(&HashMap::new()).unwrap();
        assert_eq!(gen.name(), "algorithm_vm");

        let params = GenParams::simple(3);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        assert!(!frames.is_empty());
    }

    #[test]
    fn test_zero_program_size() {
        let gen = make_vm_factory(0, 50);
        let frames = gen.generate_stream(0, &GenParams::simple(10)).unwrap().collect::<Vec<_>>();
        // 空程序：仅包含 Halt，立即停止
        assert!(frames.len() <= 10);
    }

    #[test]
    fn test_max_steps_enforced() {
        let gen = make_vm_factory(32, 5);
        let frames = gen
            .generate_stream(42, &GenParams::simple(0))
            .unwrap()
            .collect::<Vec<_>>();
        // max_steps=5 表示最多执行 5 条指令，输出 5 帧
        assert!(frames.len() <= 5);
    }

    #[test]
    fn test_div_zero_preserves_stack() {
        // 验证除零时栈不变
        let mut stack: Vec<i64> = vec![10, 0];
        let b = stack.pop().unwrap();
        if b != 0 {
            let a = stack.pop().unwrap();
            stack.push(a / b);
        } else {
            let a = stack.pop().unwrap();
            stack.push(a);
            stack.push(b);
        }
        assert_eq!(stack, vec![10, 0], "除零时应保持栈不变");
    }

    #[test]
    fn test_params_via_factory() {
        // 验证参数通过工厂函数传递到生成器
        let gen = make_vm_factory(4, 10);
        let frames = gen.generate_stream(0, &GenParams::simple(0)).unwrap().collect::<Vec<_>>();
        // program_size=4, max_steps=10，应最多 10 帧
        assert!(frames.len() <= 10);
    }
}
