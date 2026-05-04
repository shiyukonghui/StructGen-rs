use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use crate::core::*;

/// 上下文无关 L 系统专用参数
#[derive(Debug, Clone, Deserialize)]
struct LSystemParams {
    /// 公理（起始符号串）
    #[serde(default = "default_axiom")]
    axiom: String,
    /// 产生式规则：字符 -> 替换字符串
    #[serde(default)]
    rules: HashMap<char, String>,
    /// 迭代次数
    #[serde(default = "default_iterations")]
    iterations: usize,
    /// 最大字符串长度保护（超出时停止迭代）
    #[serde(default = "default_max_string_length")]
    max_string_length: usize,
}

fn default_axiom() -> String {
    "F".to_string()
}
fn default_iterations() -> usize {
    5
}
fn default_max_string_length() -> usize {
    1_000_000
}

impl Default for LSystemParams {
    fn default() -> Self {
        LSystemParams {
            axiom: default_axiom(),
            rules: HashMap::new(),
            iterations: default_iterations(),
            max_string_length: default_max_string_length(),
        }
    }
}

/// L 系统生成器
///
/// 上下文无关重写系统：从公理开始，每步将符号串中匹配的字符替换为对应产生式。
/// 每次迭代后输出完整符号串，编码为 FrameState::Integer 序列（每个字符映射为其 ASCII/Unicode 码点）。
pub struct LSystem {
    axiom: String,
    rules: HashMap<char, String>,
    iterations: usize,
    max_string_length: usize,
}

impl Generator for LSystem {
    fn name(&self) -> &'static str {
        "lsystem"
    }

    fn generate_stream(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        // L 系统本身是确定性的（无随机性），但符合接口要求保留 seed
        let _ = seed;

        let max_iterations = self.iterations;
        let axiom = self.axiom.clone();
        let rules = self.rules.clone();
        let max_string_length = self.max_string_length;
        let seq_limit = params.seq_length;

        let mut step_counter: u64 = 0;
        let mut current = axiom;
        let mut rules_applied: usize = 0;
        let mut finished = false;

        let iter = std::iter::from_fn(move || {
            if finished {
                return None;
            }

            let step = step_counter;
            step_counter += 1;

            // 输出当前符号串
            let values: Vec<FrameState> = current
                .chars()
                .map(|c| FrameState::Integer(c as i64))
                .collect();
            let frame = SequenceFrame::new(step, FrameData { values });

            // 执行替换（除非已达到最大迭代次数或字符串长度限制）
            if rules_applied < max_iterations && current.len() < max_string_length {
                let mut next = String::new();
                for ch in current.chars() {
                    if let Some(replacement) = rules.get(&ch) {
                        next.push_str(replacement);
                    } else {
                        next.push(ch);
                    }
                }
                // 替换后超长也停止后续迭代
                if next.len() > max_string_length {
                    finished = true;
                }
                current = next;
                rules_applied += 1;
            } else {
                finished = true;
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

/// L 系统工厂函数
pub fn lsystem_factory(extensions: &HashMap<String, Value>) -> CoreResult<Box<dyn Generator>> {
    let lsystem_params: LSystemParams = deserialize_extensions(extensions)?;

    if lsystem_params.axiom.is_empty() {
        return Err(CoreError::InvalidParams(
            "LSystem axiom cannot be empty".into(),
        ));
    }

    if lsystem_params.max_string_length == 0 {
        return Err(CoreError::InvalidParams(
            "LSystem max_string_length must be greater than 0".into(),
        ));
    }

    Ok(Box::new(LSystem {
        axiom: lsystem_params.axiom,
        rules: lsystem_params.rules,
        iterations: lsystem_params.iterations,
        max_string_length: lsystem_params.max_string_length,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// 经典的 Koch 曲线 L 系统：公理 "F", 规则 F -> "F+F-F-F+F", 角度 90°
    fn koch_lsystem() -> (String, HashMap<char, String>) {
        let axiom = "F".to_string();
        let mut rules = HashMap::new();
        rules.insert('F', "F+F-F-F+F".to_string());
        (axiom, rules)
    }

    #[test]
    fn test_deterministic_same_params_same_output() {
        let (axiom, rules) = koch_lsystem();

        let mut ext1 = HashMap::new();
        ext1.insert("axiom".to_string(), json!(axiom));
        ext1.insert("rules".to_string(), json!(rules));
        ext1.insert("iterations".to_string(), json!(3));

        let mut ext2 = HashMap::new();
        ext2.insert("axiom".to_string(), json!("F"));
        ext2.insert("rules".to_string(), json!({"F": "F+F-F-F+F"}));
        ext2.insert("iterations".to_string(), json!(3));

        let gen1 = lsystem_factory(&ext1).unwrap();
        let gen2 = lsystem_factory(&ext2).unwrap();

        let params = GenParams::simple(4);
        let frames1 = gen1.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        let frames2 = gen2.generate_stream(0, &params).unwrap().collect::<Vec<_>>();

        for i in 0..4 {
            assert_eq!(
                frames1[i].state.values, frames2[i].state.values,
                "Frame {} should be identical",
                i
            );
        }
    }

    #[test]
    fn test_koch_curve_iterations() {
        let (axiom, rules) = koch_lsystem();

        let mut extensions = HashMap::new();
        extensions.insert("axiom".to_string(), json!(axiom));
        extensions.insert("rules".to_string(), json!(rules));
        extensions.insert("iterations".to_string(), json!(3));

        let gen = lsystem_factory(&extensions).unwrap();
        let params = GenParams::simple(4); // axiom + 3 次迭代 = 4 帧
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();

        assert_eq!(frames.len(), 4);

        // 第 0 帧：公理 "F"
        assert_eq!(frames[0].state.values, vec![FrameState::Integer('F' as i64)]);

        // 第 1 帧：一次迭代 "F+F-F-F+F" = 9 个字符
        assert_eq!(frames[1].state.values.len(), 9);
        let expected: Vec<FrameState> = "F+F-F-F+F"
            .chars()
            .map(|c| FrameState::Integer(c as i64))
            .collect();
        assert_eq!(frames[1].state.values, expected);

        // 第 2 帧：9 chars * 5 Fs + 4 operators = 49
        assert_eq!(frames[2].state.values.len(), 49);
    }

    #[test]
    fn test_empty_axiom_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("axiom".to_string(), json!(""));
        let result = lsystem_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_default_params() {
        let gen = lsystem_factory(&HashMap::new()).unwrap();
        assert_eq!(gen.name(), "lsystem");

        let params = GenParams::simple(6);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        // 默认 axiom="F", no rules, iterations=5 -> 6 帧输出 (axiom + 5 iterations)
        assert_eq!(frames.len(), 6);
    }

    #[test]
    fn test_unbounded_stream() {
        let (axiom, rules) = koch_lsystem();
        let mut extensions = HashMap::new();
        extensions.insert("axiom".to_string(), json!(axiom));
        extensions.insert("rules".to_string(), json!(rules));
        extensions.insert("iterations".to_string(), json!(3));

        let gen = lsystem_factory(&extensions).unwrap();
        let params = GenParams::simple(0);
        let frames = gen
            .generate_stream(0, &params)
            .unwrap()
            .collect::<Vec<_>>();
        // 应产生 4 帧（axiom + 3 iterations），然后停止
        assert_eq!(frames.len(), 4);
    }

    #[test]
    fn test_max_string_length_enforced() {
        let (axiom, rules) = koch_lsystem();
        let mut extensions = HashMap::new();
        extensions.insert("axiom".to_string(), json!(axiom));
        extensions.insert("rules".to_string(), json!(rules));
        extensions.insert("iterations".to_string(), json!(100)); // 很高迭代次数
        extensions.insert("max_string_length".to_string(), json!(50)); // 但限制字符串长度

        let gen = lsystem_factory(&extensions).unwrap();
        let params = GenParams::simple(0);
        let frames = gen
            .generate_stream(0, &params)
            .unwrap()
            .collect::<Vec<_>>();

        // 迭代应在字符串长度超限时提前停止
        assert!(frames.len() < 100, "应在字符串长度超限时停止");
        // 所有帧的字符串长度不应远超限制
        for f in &frames {
            assert!(f.state.values.len() <= 300,
                "字符串长度应受限制，实际: {}", f.state.values.len());
        }
    }
}
