use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use super::rng::SeedRng;
use crate::core::*;

/// IFS（迭代函数系统）专用参数
#[derive(Debug, Clone, Deserialize)]
struct IFSParams {
    /// 仿射变换数量
    #[serde(default = "default_num_transforms")]
    num_transforms: usize,
}

fn default_num_transforms() -> usize {
    4
}

impl Default for IFSParams {
    fn default() -> Self {
        IFSParams {
            num_transforms: default_num_transforms(),
        }
    }
}

/// 单一仿射变换参数
#[derive(Debug, Clone)]
struct AffineTransform {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    probability: f64,
}

/// 随机迭代函数系统生成器
///
/// 多组随机仿射变换 + 概率，每次根据概率选择变换应用，输出当前点坐标 (x, y) 为一帧。
pub struct Ifs {
    num_transforms: usize,
}

impl Generator for Ifs {
    fn name(&self) -> &'static str {
        "ifs"
    }

    fn generate_stream(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let num_transforms = self.num_transforms;
        let seq_limit = params.seq_length;

        let mut rng = SeedRng::new(seed);

        // 生成随机仿射变换参数
        let mut transforms: Vec<AffineTransform> = Vec::with_capacity(num_transforms);
        let mut total_prob = 0.0f64;
        for _ in 0..num_transforms {
            let prob = rng.next_f64_range(0.1, 1.0);
            total_prob += prob;
            transforms.push(AffineTransform {
                a: rng.next_f64_range(-1.0, 1.0),
                b: rng.next_f64_range(-1.0, 1.0),
                c: rng.next_f64_range(-1.0, 1.0),
                d: rng.next_f64_range(-1.0, 1.0),
                e: rng.next_f64_range(-0.5, 0.5),
                f: rng.next_f64_range(-0.5, 0.5),
                probability: prob,
            });
        }

        // 归一化概率
        for t in &mut transforms {
            t.probability /= total_prob;
        }

        // 初始点
        let mut x: f64 = 0.0;
        let mut y: f64 = 0.0;
        let mut step_counter: u64 = 0;

        let iter = std::iter::from_fn(move || {
            let step = step_counter;
            step_counter += 1;

            // 输出当前点坐标
            let values = vec![FrameState::float_or_zero(x), FrameState::float_or_zero(y)];
            let frame = SequenceFrame::new(step, FrameData { values });

            // 根据概率选择一个变换
            let r = rng.next_f64();
            let mut cumulative = 0.0;
            let mut chosen = transforms.len() - 1; // 默认选择最后一个，避免浮点累积误差遗漏
            for (i, t) in transforms.iter().enumerate() {
                cumulative += t.probability;
                if r <= cumulative {
                    chosen = i;
                    break;
                }
            }

            // 应用变换
            let t = &transforms[chosen];
            let nx = t.a * x + t.b * y + t.e;
            let ny = t.c * x + t.d * y + t.f;
            x = nx;
            y = ny;

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

/// IFS 工厂函数
pub fn ifs_factory(extensions: &HashMap<String, Value>) -> CoreResult<Box<dyn Generator>> {
    let ifs_params: IFSParams = deserialize_extensions(extensions)?;

    if ifs_params.num_transforms == 0 {
        return Err(CoreError::InvalidParams(
            "IFS num_transforms must be greater than 0".into(),
        ));
    }

    Ok(Box::new(Ifs {
        num_transforms: ifs_params.num_transforms,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_frames(gen: &dyn Generator, seed: u64, num_frames: usize) -> Vec<SequenceFrame> {
        let params = GenParams::simple(num_frames);
        gen.generate_stream(seed, &params).unwrap().collect()
    }

    #[test]
    fn test_deterministic_same_seed_same_output() {
        let gen1 = ifs_factory(&HashMap::new()).unwrap();
        let gen2 = ifs_factory(&HashMap::new()).unwrap();

        let frames1 = collect_frames(gen1.as_ref(), 42, 50);
        let frames2 = collect_frames(gen2.as_ref(), 42, 50);

        assert_eq!(frames1.len(), 50);
        for i in 0..50 {
            assert_eq!(
                frames1[i].state.values, frames2[i].state.values,
                "Frame {} should be identical",
                i
            );
        }
    }

    #[test]
    fn test_different_seed_different_output() {
        let gen = ifs_factory(&HashMap::new()).unwrap();

        let frames1 = collect_frames(gen.as_ref(), 100, 10);
        let frames2 = collect_frames(gen.as_ref(), 200, 10);

        // 帧 0 始终是初始点 (0,0)，从帧 1 开始因不同变换参数而分叉
        assert!(frames1[1].state.values != frames2[1].state.values);
    }

    #[test]
    fn test_output_dimensions() {
        let gen = ifs_factory(&HashMap::new()).unwrap();
        let params = GenParams::simple(5);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();

        for f in &frames {
            assert_eq!(f.state.values.len(), 2); // (x, y)
            for v in &f.state.values {
                match v {
                    FrameState::Float(val) => assert!(val.is_finite()),
                    _ => panic!("Expected Float values in IFS output"),
                }
            }
        }
    }

    #[test]
    fn test_custom_num_transforms() {
        let mut extensions = HashMap::new();
        extensions.insert(
            "num_transforms".to_string(),
            Value::Number(serde_json::Number::from(6)),
        );

        let gen = ifs_factory(&extensions).unwrap();
        let params = GenParams::simple(3);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 3);
    }

    #[test]
    fn test_zero_transforms_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert(
            "num_transforms".to_string(),
            Value::Number(serde_json::Number::from(0)),
        );
        let result = ifs_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_default_params() {
        let gen = ifs_factory(&HashMap::new()).unwrap();
        assert_eq!(gen.name(), "ifs");

        let params = GenParams::simple(2);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 2);
    }

    #[test]
    fn test_unbounded_stream() {
        let gen = ifs_factory(&HashMap::new()).unwrap();
        let params = GenParams::simple(0);
        let frames = gen
            .generate_stream(42, &params)
            .unwrap()
            .take(100)
            .collect::<Vec<_>>();
        assert_eq!(frames.len(), 100);
    }
}
