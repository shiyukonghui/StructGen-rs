use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use super::rng::SeedRng;
use crate::core::*;

/// 洛伦兹系统专用参数
#[derive(Debug, Clone, Deserialize)]
struct LorenzParams {
    /// σ 参数（普朗特数）
    #[serde(default = "default_sigma")]
    sigma: f64,
    /// ρ 参数（瑞利数）
    #[serde(default = "default_rho")]
    rho: f64,
    /// β 参数
    #[serde(default = "default_beta")]
    beta: f64,
    /// 积分步长
    #[serde(default = "default_dt")]
    dt: f64,
}

fn default_sigma() -> f64 {
    10.0
}
fn default_rho() -> f64 {
    28.0
}
fn default_beta() -> f64 {
    8.0 / 3.0
}
fn default_dt() -> f64 {
    0.01
}

impl Default for LorenzParams {
    fn default() -> Self {
        LorenzParams {
            sigma: default_sigma(),
            rho: default_rho(),
            beta: default_beta(),
            dt: default_dt(),
        }
    }
}

/// 洛伦兹系统生成器
///
/// 三维连续动力系统：
/// - dx/dt = σ(y - x)
/// - dy/dt = x(ρ - z) - y
/// - dz/dt = xy - βz
///
/// 使用 RK4 积分，每步输出 (x, y, z) 为一帧。
pub struct LorenzSystem {
    sigma: f64,
    rho: f64,
    beta: f64,
    dt: f64,
}

impl LorenzSystem {
    /// 洛伦兹系统状态导数
    fn derivatives(x: f64, y: f64, z: f64, sigma: f64, rho: f64, beta: f64) -> (f64, f64, f64) {
        let dx = sigma * (y - x);
        let dy = x * (rho - z) - y;
        let dz = x * y - beta * z;
        (dx, dy, dz)
    }

    /// RK4 单步积分
    fn rk4_step(
        x: f64,
        y: f64,
        z: f64,
        dt: f64,
        sigma: f64,
        rho: f64,
        beta: f64,
    ) -> (f64, f64, f64) {
        let (k1x, k1y, k1z) = Self::derivatives(x, y, z, sigma, rho, beta);

        let x2 = x + 0.5 * dt * k1x;
        let y2 = y + 0.5 * dt * k1y;
        let z2 = z + 0.5 * dt * k1z;
        let (k2x, k2y, k2z) = Self::derivatives(x2, y2, z2, sigma, rho, beta);

        let x3 = x + 0.5 * dt * k2x;
        let y3 = y + 0.5 * dt * k2y;
        let z3 = z + 0.5 * dt * k2z;
        let (k3x, k3y, k3z) = Self::derivatives(x3, y3, z3, sigma, rho, beta);

        let x4 = x + dt * k3x;
        let y4 = y + dt * k3y;
        let z4 = z + dt * k3z;
        let (k4x, k4y, k4z) = Self::derivatives(x4, y4, z4, sigma, rho, beta);

        let nx = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        let ny = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
        let nz = z + (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);

        (nx, ny, nz)
    }
}

impl Generator for LorenzSystem {
    fn name(&self) -> &'static str {
        "lorenz_system"
    }

    fn generate_stream(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let sigma = self.sigma;
        let rho = self.rho;
        let beta = self.beta;
        let dt = self.dt;
        let seq_limit = params.seq_length;

        // 用种子 PRNG 在吸引子附近生成初始值 [-15, 15]
        let mut rng = SeedRng::new(seed);
        let mut x = rng.next_f64_range(-15.0, 15.0);
        let mut y = rng.next_f64_range(-15.0, 15.0);
        let mut z = rng.next_f64_range(-15.0, 15.0);
        let mut step_counter: u64 = 0;

        let iter = std::iter::from_fn(move || {
            if seq_limit > 0 && step_counter >= seq_limit as u64 {
                return None;
            }

            let step = step_counter;
            step_counter += 1;

            // 输出当前状态
            let values = vec![
                FrameState::Float(x),
                FrameState::Float(y),
                FrameState::Float(z),
            ];
            let frame = SequenceFrame::new(step, FrameData { values });

            // 执行 RK4 步进
            let (nx, ny, nz) = LorenzSystem::rk4_step(x, y, z, dt, sigma, rho, beta);
            x = nx;
            y = ny;
            z = nz;

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

/// 洛伦兹系统工厂函数
pub fn lorenz_factory(extensions: &HashMap<String, Value>) -> CoreResult<Box<dyn Generator>> {
    let lorenz_params: LorenzParams = if extensions.is_empty() {
        LorenzParams::default()
    } else {
        let obj = serde_json::to_value(extensions).map_err(|e| {
            CoreError::SerializationError(format!("failed to serialize extensions: {}", e))
        })?;
        serde_json::from_value(obj).map_err(|e| {
            CoreError::SerializationError(format!("failed to deserialize Lorenz params: {}", e))
        })?
    };

    if lorenz_params.dt <= 0.0 {
        return Err(CoreError::InvalidParams(
            "Lorenz dt must be positive".into(),
        ));
    }

    Ok(Box::new(LorenzSystem {
        sigma: lorenz_params.sigma,
        rho: lorenz_params.rho,
        beta: lorenz_params.beta,
        dt: lorenz_params.dt,
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
        let gen1 = lorenz_factory(&HashMap::new()).unwrap();
        let gen2 = lorenz_factory(&HashMap::new()).unwrap();

        let frames1 = collect_frames(gen1.as_ref(), 42, 50);
        let frames2 = collect_frames(gen2.as_ref(), 42, 50);

        assert_eq!(frames1.len(), 50);
        assert_eq!(frames2.len(), 50);
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
        let gen = lorenz_factory(&HashMap::new()).unwrap();

        let frames1 = collect_frames(gen.as_ref(), 100, 10);
        let frames2 = collect_frames(gen.as_ref(), 200, 10);

        assert!(frames1[0].state.values != frames2[0].state.values);
    }

    #[test]
    fn test_output_dimensions() {
        let gen = lorenz_factory(&HashMap::new()).unwrap();
        let params = GenParams::simple(5);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        for f in &frames {
            assert_eq!(f.state.values.len(), 3); // x, y, z
            // 验证都是有限值
            for v in &f.state.values {
                match v {
                    FrameState::Float(val) => assert!(val.is_finite()),
                    _ => panic!("Expected Float values in Lorenz output"),
                }
            }
        }
    }

    #[test]
    fn test_negative_dt_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("dt".to_string(), json!(-0.01));
        let result = lorenz_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_dt_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("dt".to_string(), json!(0.0));
        let result = lorenz_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_params() {
        let mut extensions = HashMap::new();
        extensions.insert("sigma".to_string(), json!(12.0));
        extensions.insert("rho".to_string(), json!(30.0));
        extensions.insert("beta".to_string(), json!(2.5));
        extensions.insert("dt".to_string(), json!(0.005));

        let gen = lorenz_factory(&extensions).unwrap();
        assert_eq!(gen.name(), "lorenz_system");

        let params = GenParams::simple(3);
        let frames = gen.generate_stream(1, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 3);
    }

    #[test]
    fn test_default_params() {
        let gen = lorenz_factory(&HashMap::new()).unwrap();
        let params = GenParams::simple(2);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 2);
    }

    #[test]
    fn test_unbounded_stream() {
        let gen = lorenz_factory(&HashMap::new()).unwrap();
        let params = GenParams::simple(0);
        let frames = gen
            .generate_stream(42, &params)
            .unwrap()
            .take(100)
            .collect::<Vec<_>>();
        assert_eq!(frames.len(), 100);
    }
}
