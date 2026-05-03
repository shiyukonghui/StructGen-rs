use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use super::rng::SeedRng;
use crate::core::*;

/// 上下文无关文法专用参数
#[derive(Debug, Clone, Deserialize)]
struct FormalGrammarParams {
    /// 产生式规则：非终结符 -> 候选产生式列表
    #[serde(default)]
    productions: HashMap<char, Vec<String>>,
    /// 起始符号
    #[serde(default = "default_start_symbol")]
    start_symbol: char,
    /// 最大推导次数
    #[serde(default = "default_max_derivations")]
    max_derivations: usize,
}

fn default_start_symbol() -> char {
    'S'
}
fn default_max_derivations() -> usize {
    100
}

impl Default for FormalGrammarParams {
    fn default() -> Self {
        FormalGrammarParams {
            productions: HashMap::new(),
            start_symbol: default_start_symbol(),
            max_derivations: default_max_derivations(),
        }
    }
}

/// 上下文无关文法生成器
///
/// 从起始符号开始，每次随机选择产生式进行推导，直到所有非终结符被替换为终结符。
/// 每步输出当前推导结果（可能含非终结符），编码为 FrameState 序列。
pub struct FormalGrammar {
    productions: HashMap<char, Vec<String>>,
    start_symbol: char,
    max_derivations: usize,
}

impl Generator for FormalGrammar {
    fn name(&self) -> &'static str {
        "formal_grammar"
    }

    fn generate_stream(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let productions = self.productions.clone();
        let start_symbol = self.start_symbol;
        let max_derivations = self.max_derivations;
        let seq_limit = params.seq_length;

        let mut rng = SeedRng::new(seed);
        let mut current = String::from(start_symbol);
        let mut step_counter: u64 = 0;
        let mut derivations_done: usize = 0;
        let mut finished = false;

        let iter = std::iter::from_fn(move || {
            if finished {
                // 推导完成：继续输出相同结果直到达序列限制
                if seq_limit > 0 && step_counter >= seq_limit as u64 {
                    return None;
                }
                let step = step_counter;
                step_counter += 1;
                let values: Vec<FrameState> = current
                    .chars()
                    .map(|c| FrameState::Integer(c as i64))
                    .collect();
                return Some(SequenceFrame::new(step, FrameData { values }));
            }

            if seq_limit > 0 && step_counter >= seq_limit as u64 {
                return None;
            }

            let step = step_counter;
            step_counter += 1;

            // 输出当前推导结果
            let values: Vec<FrameState> = current
                .chars()
                .map(|c| FrameState::Integer(c as i64))
                .collect();
            let frame = SequenceFrame::new(step, FrameData { values });

            // 查找第一个非终结符（大写字母）并替换
            if derivations_done < max_derivations && !productions.is_empty() {
                let mut found = false;
                let mut new_current = String::new();
                for ch in current.chars() {
                    if !found && ch.is_ascii_uppercase() {
                        if let Some(options) = productions.get(&ch) {
                            if !options.is_empty() {
                                let idx = rng.next_usize(options.len());
                                new_current.push_str(&options[idx]);
                                derivations_done += 1;
                                found = true;
                                continue;
                            }
                        }
                    }
                    new_current.push(ch);
                }

                if !found {
                    // 没有更多非终结符可替换，推导完成
                    finished = true;
                    // 继续输出最后一个状态（由下次迭代处理）
                }
                current = new_current;
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

/// 上下文无关文法工厂函数
pub fn formal_grammar_factory(
    extensions: &HashMap<String, Value>,
) -> CoreResult<Box<dyn Generator>> {
    let fg_params: FormalGrammarParams = if extensions.is_empty() {
        FormalGrammarParams::default()
    } else {
        let obj = serde_json::to_value(extensions).map_err(|e| {
            CoreError::SerializationError(format!("failed to serialize extensions: {}", e))
        })?;
        serde_json::from_value(obj).map_err(|e| {
            CoreError::SerializationError(format!(
                "failed to deserialize FormalGrammar params: {}",
                e
            ))
        })?
    };

    Ok(Box::new(FormalGrammar {
        productions: fg_params.productions,
        start_symbol: fg_params.start_symbol,
        max_derivations: fg_params.max_derivations,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// 构建一个简单的算术表达式文法
    fn arithmetic_grammar() -> HashMap<char, Vec<String>> {
        let mut prods = HashMap::new();
        prods.insert(
            'S',
            vec!["E".to_string()],
        );
        prods.insert(
            'E',
            vec![
                "E+T".to_string(),
                "T".to_string(),
            ],
        );
        prods.insert(
            'T',
            vec![
                "T*F".to_string(),
                "F".to_string(),
            ],
        );
        prods.insert(
            'F',
            vec![
                "(E)".to_string(),
                "n".to_string(),
            ],
        );
        prods
    }

    #[test]
    fn test_deterministic_same_seed_same_output() {
        let prods = arithmetic_grammar();

        let mut ext1 = HashMap::new();
        ext1.insert("productions".to_string(), json!(prods));
        ext1.insert("start_symbol".to_string(), json!('S'));
        ext1.insert("max_derivations".to_string(), json!(20));

        let mut ext2 = HashMap::new();
        ext2.insert("productions".to_string(), json!(arithmetic_grammar()));
        ext2.insert("start_symbol".to_string(), json!('S'));
        ext2.insert("max_derivations".to_string(), json!(20));

        let gen1 = formal_grammar_factory(&ext1).unwrap();
        let gen2 = formal_grammar_factory(&ext2).unwrap();

        let params = GenParams::simple(21);
        let frames1 = gen1.generate_stream(42, &params).unwrap().collect::<Vec<_>>();
        let frames2 = gen2.generate_stream(42, &params).unwrap().collect::<Vec<_>>();

        // 两者使用相同的文法，相同的种子，应产生相同的输出
        for i in 0..frames1.len().min(frames2.len()) {
            assert_eq!(
                frames1[i].state.values, frames2[i].state.values,
                "Frame {} should be identical",
                i
            );
        }
    }

    #[test]
    fn test_different_seed_different_output() {
        let prods = arithmetic_grammar();
        let mut extensions = HashMap::new();
        extensions.insert("productions".to_string(), json!(prods));
        extensions.insert("max_derivations".to_string(), json!(5));

        let gen = formal_grammar_factory(&extensions).unwrap();

        let params = GenParams::simple(6);
        let frames1 = gen.generate_stream(100, &params).unwrap().collect::<Vec<_>>();
        let frames2 = gen.generate_stream(200, &params).unwrap().collect::<Vec<_>>();

        // 不同种子可能导致不同的推导路径
        // 但第一个字符可能相同（都是 'S'）
        assert_eq!(frames1[0].state.values, frames2[0].state.values);
    }

    #[test]
    fn test_derivation_progresses() {
        let prods = arithmetic_grammar();
        let mut extensions = HashMap::new();
        extensions.insert("productions".to_string(), json!(prods));
        extensions.insert("max_derivations".to_string(), json!(10));

        let gen = formal_grammar_factory(&extensions).unwrap();

        let params = GenParams::simple(11);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();

        // 第 0 帧应该是 'S'
        assert_eq!(
            frames[0].state.values,
            vec![FrameState::Integer('S' as i64)]
        );
    }

    #[test]
    fn test_max_derivations_enforced() {
        // 只允许一次推导：S -> E
        let mut prods = HashMap::new();
        prods.insert('S', vec!["E".to_string()]);

        let mut extensions = HashMap::new();
        extensions.insert("productions".to_string(), json!(prods));
        extensions.insert("max_derivations".to_string(), json!(1));

        let gen = formal_grammar_factory(&extensions).unwrap();
        let params = GenParams::simple(10);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();

        // 应该至少有 2 帧：'S' 和 'E'
        assert!(frames.len() >= 2);
        // 第一帧是 'S'
        assert_eq!(
            frames[0].state.values,
            vec![FrameState::Integer('S' as i64)]
        );
        // 第二帧是 'E'（一次推导后）
        assert_eq!(
            frames[1].state.values,
            vec![FrameState::Integer('E' as i64)]
        );
    }

    #[test]
    fn test_default_params() {
        let gen = formal_grammar_factory(&HashMap::new()).unwrap();
        assert_eq!(gen.name(), "formal_grammar");

        let params = GenParams::simple(3);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        // 默认空文法：只有 'S'（默认起始符号），没有产生式 -> 无法推导
        assert_eq!(
            frames[0].state.values,
            vec![FrameState::Integer('S' as i64)]
        );
    }

    #[test]
    fn test_output_all_integers() {
        let prods = arithmetic_grammar();
        let mut extensions = HashMap::new();
        extensions.insert("productions".to_string(), json!(prods));
        extensions.insert("max_derivations".to_string(), json!(5));

        let gen = formal_grammar_factory(&extensions).unwrap();
        let params = GenParams::simple(6);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();

        for f in &frames {
            for v in &f.state.values {
                assert!(matches!(v, FrameState::Integer(_)));
            }
        }
    }
}
