use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use super::rng::SeedRng;
use crate::core::*;

/// 逻辑斯蒂映射专用参数
#[derive(Debug, Clone, Deserialize)]
struct LogisticParams {
    /// 生长率参数 r
    #[serde(default = "default_r")]
    r: f64,
    /// 初始值 x0（如不指定则由种子 PRNG 生成）
    #[serde(default)]
    x0: Option<f64>,
}

fn default_r() -> f64 {
    3.9
}

impl Default for LogisticParams {
    fn default() -> Self {
        LogisticParams {
            r: default_r(),
            x0: None,
        }
    }
}

/// 逻辑斯蒂映射生成器
///
/// 离散映射：x_{n+1} = r * x_n * (1 - x_n)
/// 每步输出当前 x 值；当 x 越出 [0, 1] 时置为 0.0 并停止产生帧
pub struct LogisticMap {
    r: f64,
    x0: Option<f64>,
}

impl Generator for LogisticMap {
    fn name(&self) -> &'static str {
        "logistic_map"
    }

    fn generate_stream(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let r = self.r;
        let seq_limit = params.seq_length;

        // 获取初始值 x0（优先使用结构体中的值，否则由种子 PRNG 生成）
        let mut rng = SeedRng::new(seed);
        let mut x = self.x0.unwrap_or_else(|| rng.next_f64_range(0.01, 0.99));
        let mut step_counter: u64 = 0;
        let mut finished = false;

        let iter = std::iter::from_fn(move || {
            if finished {
                return None;
            }

            let step = step_counter;
            step_counter += 1;

            let values = vec![FrameState::float_or_zero(x)];
            let frame = SequenceFrame::new(step, FrameData { values });

            // 迭代映射
            x = r * x * (1.0 - x);

            // 越界检测：x 超出 [0, 1] 时置为 0.0 并标记结束
            if !(0.0..=1.0).contains(&x) {
                x = 0.0;
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

/// 逻辑斯蒂映射工厂函数
pub fn logistic_factory(extensions: &HashMap<String, Value>) -> CoreResult<Box<dyn Generator>> {
    let logistic_params: LogisticParams = deserialize_extensions(extensions)?;

    // r 必须在 [0, 4] 范围内
    if logistic_params.r < 0.0 || logistic_params.r > 4.0 {
        return Err(CoreError::InvalidParams(format!(
            "Logistic r must be in [0, 4], got {}",
            logistic_params.r
        )));
    }

    // 验证 x0 在合法范围内
    if let Some(x0) = logistic_params.x0 {
        if !(0.0..=1.0).contains(&x0) {
            return Err(CoreError::InvalidParams(format!(
                "Logistic x0 must be in [0, 1], got {}",
                x0
            )));
        }
    }

    Ok(Box::new(LogisticMap {
        r: logistic_params.r,
        x0: logistic_params.x0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn collect_frames(gen: &dyn Generator, seed: u64, num_frames: usize) -> Vec<SequenceFrame> {
        let params = GenParams::simple(num_frames);
        gen.generate_stream(seed, &params).unwrap().collect()
    }

    #[test]
    fn test_deterministic_same_seed_same_output() {
        let gen1 = logistic_factory(&HashMap::new()).unwrap();
        let gen2 = logistic_factory(&HashMap::new()).unwrap();

        let frames1 = collect_frames(gen1.as_ref(), 42, 20);
        let frames2 = collect_frames(gen2.as_ref(), 42, 20);

        assert_eq!(frames1.len(), 20);
        for i in 0..20 {
            assert_eq!(
                frames1[i].state.values, frames2[i].state.values,
                "Frame {} should be identical",
                i
            );
        }
    }

    #[test]
    fn test_different_seed_different_output() {
        let gen = logistic_factory(&HashMap::new()).unwrap();
        let frames1 = collect_frames(gen.as_ref(), 100, 5);
        let frames2 = collect_frames(gen.as_ref(), 200, 5);
        assert!(frames1[0].state.values != frames2[0].state.values);
    }

    #[test]
    fn test_r_out_of_range_negative_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("r".to_string(), json!(-0.5));
        let result = logistic_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_r_out_of_range_over_4_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("r".to_string(), json!(4.5));
        let result = logistic_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_r_boundary_values_accepted() {
        let mut ext0 = HashMap::new();
        ext0.insert("r".to_string(), json!(0.0));
        assert!(logistic_factory(&ext0).is_ok());

        let mut ext4 = HashMap::new();
        ext4.insert("r".to_string(), json!(4.0));
        assert!(logistic_factory(&ext4).is_ok());
    }

    #[test]
    fn test_default_params() {
        let gen = logistic_factory(&HashMap::new()).unwrap();
        assert_eq!(gen.name(), "logistic_map");
        let params = GenParams::simple(3);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 3);
        for f in &frames {
            assert_eq!(f.state.values.len(), 1);
            match &f.state.values[0] {
                FrameState::Float(v) => assert!(v.is_finite()),
                _ => panic!("Expected Float"),
            }
        }
    }

    #[test]
    fn test_unbounded_stream() {
        let gen = logistic_factory(&HashMap::new()).unwrap();
        let params = GenParams::simple(0);
        let frames = gen
            .generate_stream(42, &params)
            .unwrap()
            .take(50)
            .collect::<Vec<_>>();
        assert_eq!(frames.len(), 50);
    }

    #[test]
    fn test_r4_no_divergence() {
        // r=4 边界条件下长时间运行不应产生 NaN/Infinity
        let mut extensions = HashMap::new();
        extensions.insert("r".to_string(), json!(4.0));
        let gen = logistic_factory(&extensions).unwrap();
        let params = GenParams::simple(1000);
        let frames = gen.generate_stream(42, &params).unwrap().collect::<Vec<_>>();
        for f in &frames {
            for v in &f.state.values {
                match v {
                    FrameState::Float(val) => assert!(val.is_finite() && *val >= 0.0 && *val <= 1.0),
                    _ => panic!("Expected Float"),
                }
            }
        }
    }

    #[test]
    fn test_x0_param_passed_via_factory() {
        // 验证 x0 通过工厂函数传递到生成器
        let mut extensions = HashMap::new();
        extensions.insert("r".to_string(), json!(3.0));
        extensions.insert("x0".to_string(), json!(0.5));
        let gen = logistic_factory(&extensions).unwrap();
        let params = GenParams::simple(1);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        // 第一帧应该是 x0=0.5
        match frames[0].state.values[0] {
            FrameState::Float(v) => assert!((v - 0.5).abs() < 1e-10),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_x0_out_of_range_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("x0".to_string(), json!(1.5));
        let result = logistic_factory(&extensions);
        assert!(result.is_err());
    }
}
