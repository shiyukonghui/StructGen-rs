use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use super::rng::SeedRng;
use crate::core::*;

/// 积分器类型
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum Integrator {
    /// 显式欧拉法
    Euler,
    /// 四阶龙格-库塔法
    #[default]
    Rk4,
}

/// 多体模拟专用参数
#[derive(Debug, Clone, Deserialize)]
struct NBodyParams {
    /// 天体数量
    #[serde(default = "default_num_bodies")]
    num_bodies: usize,
    /// 积分步长
    #[serde(default = "default_dt")]
    dt: f64,
    /// 软化因子 ε（防止近距离奇点）
    #[serde(default = "default_softening")]
    softening: f64,
    /// 积分器类型
    #[serde(default)]
    integrator: Integrator,
}

fn default_num_bodies() -> usize {
    5
}
fn default_dt() -> f64 {
    0.01
}
fn default_softening() -> f64 {
    0.1
}

impl Default for NBodyParams {
    fn default() -> Self {
        NBodyParams {
            num_bodies: default_num_bodies(),
            dt: default_dt(),
            softening: default_softening(),
            integrator: Integrator::default(),
        }
    }
}

/// 多体引力模拟生成器
///
/// 经典牛顿引力：F = G*m1*m2 / (r² + ε²)
/// 每步输出所有天体的位置和速度 (px, py, vx, vy) * N
pub struct NBodySim {
    num_bodies: usize,
    dt: f64,
    softening: f64,
    integrator: Integrator,
}

/// 单个天体的状态
#[derive(Debug, Clone)]
struct Body {
    px: f64,
    py: f64,
    vx: f64,
    vy: f64,
    mass: f64,
}

/// 计算所有天体间的引力加速度
fn compute_accelerations(bodies: &[Body], softening: f64) -> Vec<(f64, f64)> {
    let n = bodies.len();
    let mut acc = vec![(0.0, 0.0); n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = bodies[j].px - bodies[i].px;
            let dy = bodies[j].py - bodies[i].py;
            let dist_sq = dx * dx + dy * dy + softening * softening;
            let inv_dist3 = 1.0 / (dist_sq * dist_sq.sqrt());
            let factor = bodies[j].mass * inv_dist3;
            acc[i].0 += factor * dx;
            acc[i].1 += factor * dy;
        }
    }
    acc
}

/// 显式欧拉步进
fn euler_step(bodies: &mut [Body], dt: f64, softening: f64) {
    let acc = compute_accelerations(bodies, softening);
    for (i, body) in bodies.iter_mut().enumerate() {
        body.vx += acc[i].0 * dt;
        body.vy += acc[i].1 * dt;
        body.px += body.vx * dt;
        body.py += body.vy * dt;
    }
}

/// RK4 步进
fn rk4_step(bodies: &mut [Body], dt: f64, softening: f64) {
    let n = bodies.len();
    let original: Vec<Body> = bodies.to_vec();

    // 计算 k1
    let acc1 = compute_accelerations(bodies, softening);
    let mut k1v = vec![(0.0, 0.0); n];
    let mut k1p = vec![(0.0, 0.0); n];
    for i in 0..n {
        k1v[i] = (acc1[i].0 * dt, acc1[i].1 * dt);
        k1p[i] = (bodies[i].vx * dt, bodies[i].vy * dt);
    }

    // 临时状态用于 k2
    let mut temp_bodies = original.clone();
    for i in 0..n {
        temp_bodies[i].px += 0.5 * k1p[i].0;
        temp_bodies[i].py += 0.5 * k1p[i].1;
        temp_bodies[i].vx += 0.5 * k1v[i].0;
        temp_bodies[i].vy += 0.5 * k1v[i].1;
    }
    let acc2 = compute_accelerations(&temp_bodies, softening);
    let mut k2v = vec![(0.0, 0.0); n];
    let mut k2p = vec![(0.0, 0.0); n];
    for i in 0..n {
        k2v[i] = (acc2[i].0 * dt, acc2[i].1 * dt);
        k2p[i] = (temp_bodies[i].vx * dt, temp_bodies[i].vy * dt);
    }

    // 临时状态用于 k3
    let mut temp_bodies = original.clone();
    for i in 0..n {
        temp_bodies[i].px += 0.5 * k2p[i].0;
        temp_bodies[i].py += 0.5 * k2p[i].1;
        temp_bodies[i].vx += 0.5 * k2v[i].0;
        temp_bodies[i].vy += 0.5 * k2v[i].1;
    }
    let acc3 = compute_accelerations(&temp_bodies, softening);
    let mut k3v = vec![(0.0, 0.0); n];
    let mut k3p = vec![(0.0, 0.0); n];
    for i in 0..n {
        k3v[i] = (acc3[i].0 * dt, acc3[i].1 * dt);
        k3p[i] = (temp_bodies[i].vx * dt, temp_bodies[i].vy * dt);
    }

    // 临时状态用于 k4
    let mut temp_bodies = original;
    for i in 0..n {
        temp_bodies[i].px += k3p[i].0;
        temp_bodies[i].py += k3p[i].1;
        temp_bodies[i].vx += k3v[i].0;
        temp_bodies[i].vy += k3v[i].1;
    }
    let acc4 = compute_accelerations(&temp_bodies, softening);
    let mut k4v = vec![(0.0, 0.0); n];
    let mut k4p = vec![(0.0, 0.0); n];
    for i in 0..n {
        k4v[i] = (acc4[i].0 * dt, acc4[i].1 * dt);
        k4p[i] = (temp_bodies[i].vx * dt, temp_bodies[i].vy * dt);
    }

    // 最终更新
    for i in 0..n {
        bodies[i].px += (k1p[i].0 + 2.0 * k2p[i].0 + 2.0 * k3p[i].0 + k4p[i].0) / 6.0;
        bodies[i].py += (k1p[i].1 + 2.0 * k2p[i].1 + 2.0 * k3p[i].1 + k4p[i].1) / 6.0;
        bodies[i].vx += (k1v[i].0 + 2.0 * k2v[i].0 + 2.0 * k3v[i].0 + k4v[i].0) / 6.0;
        bodies[i].vy += (k1v[i].1 + 2.0 * k2v[i].1 + 2.0 * k3v[i].1 + k4v[i].1) / 6.0;
    }
}

impl Generator for NBodySim {
    fn name(&self) -> &'static str {
        "nbody_sim"
    }

    fn generate_stream(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> CoreResult<Box<dyn Iterator<Item = SequenceFrame> + Send>> {
        let num_bodies = self.num_bodies;
        let dt = self.dt;
        let softening = self.softening;
        let integrator = self.integrator.clone();
        let seq_limit = params.seq_length;

        // 用种子 PRNG 生成初始位置和速度
        let mut rng = SeedRng::new(seed);
        let mut bodies: Vec<Body> = (0..num_bodies)
            .map(|_| Body {
                px: rng.next_f64_range(-10.0, 10.0),
                py: rng.next_f64_range(-10.0, 10.0),
                vx: rng.next_f64_range(-1.0, 1.0),
                vy: rng.next_f64_range(-1.0, 1.0),
                mass: rng.next_f64_range(0.5, 2.0),
            })
            .collect();
        let mut step_counter: u64 = 0;

        let iter = std::iter::from_fn(move || {
            if seq_limit > 0 && step_counter >= seq_limit as u64 {
                return None;
            }

            let step = step_counter;
            step_counter += 1;

            // 输出当前所有天体的位置和速度
            let mut values = Vec::with_capacity(num_bodies * 4);
            for body in &bodies {
                values.push(FrameState::Float(body.px));
                values.push(FrameState::Float(body.py));
                values.push(FrameState::Float(body.vx));
                values.push(FrameState::Float(body.vy));
            }
            let frame = SequenceFrame::new(step, FrameData { values });

            // 执行步进
            match integrator {
                Integrator::Euler => euler_step(&mut bodies, dt, softening),
                Integrator::Rk4 => rk4_step(&mut bodies, dt, softening),
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

/// 多体模拟工厂函数
pub fn nbody_factory(extensions: &HashMap<String, Value>) -> CoreResult<Box<dyn Generator>> {
    let nbody_params: NBodyParams = if extensions.is_empty() {
        NBodyParams::default()
    } else {
        let obj = serde_json::to_value(extensions).map_err(|e| {
            CoreError::SerializationError(format!("failed to serialize extensions: {}", e))
        })?;
        serde_json::from_value(obj).map_err(|e| {
            CoreError::SerializationError(format!("failed to deserialize NBody params: {}", e))
        })?
    };

    if nbody_params.num_bodies == 0 {
        return Err(CoreError::InvalidParams(
            "NBody num_bodies must be greater than 0".into(),
        ));
    }
    if nbody_params.dt <= 0.0 {
        return Err(CoreError::InvalidParams(
            "NBody dt must be positive".into(),
        ));
    }

    Ok(Box::new(NBodySim {
        num_bodies: nbody_params.num_bodies,
        dt: nbody_params.dt,
        softening: nbody_params.softening,
        integrator: nbody_params.integrator,
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
        let gen1 = nbody_factory(&HashMap::new()).unwrap();
        let gen2 = nbody_factory(&HashMap::new()).unwrap();

        let frames1 = collect_frames(gen1.as_ref(), 42, 10);
        let frames2 = collect_frames(gen2.as_ref(), 42, 10);

        assert_eq!(frames1.len(), 10);
        for i in 0..10 {
            assert_eq!(
                frames1[i].state.values, frames2[i].state.values,
                "Frame {} should be identical",
                i
            );
        }
    }

    #[test]
    fn test_output_dimensions() {
        let gen = nbody_factory(&HashMap::new()).unwrap();
        let params = GenParams::simple(3);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        // 默认 5 个天体，每个 4 个值 (px, py, vx, vy)
        for f in &frames {
            assert_eq!(f.state.values.len(), 5 * 4);
        }
    }

    #[test]
    fn test_euler_integrator() {
        let mut extensions = HashMap::new();
        extensions.insert("integrator".to_string(), json!("euler"));
        extensions.insert("num_bodies".to_string(), json!(3));

        let gen = nbody_factory(&extensions).unwrap();
        let params = GenParams::simple(5);
        let frames = gen.generate_stream(1, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 5);
        for f in &frames {
            assert_eq!(f.state.values.len(), 3 * 4);
        }
    }

    #[test]
    fn test_rk4_integrator() {
        let mut extensions = HashMap::new();
        extensions.insert("integrator".to_string(), json!("rk4"));
        extensions.insert("num_bodies".to_string(), json!(3));

        let gen = nbody_factory(&extensions).unwrap();
        let params = GenParams::simple(5);
        let frames = gen.generate_stream(1, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 5);
    }

    #[test]
    fn test_zero_bodies_rejected() {
        let mut extensions = HashMap::new();
        extensions.insert("num_bodies".to_string(), json!(0));
        let result = nbody_factory(&extensions);
        assert!(result.is_err());
    }

    #[test]
    fn test_default_params() {
        let gen = nbody_factory(&HashMap::new()).unwrap();
        assert_eq!(gen.name(), "nbody_sim");
        let params = GenParams::simple(2);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 2);
    }
}
