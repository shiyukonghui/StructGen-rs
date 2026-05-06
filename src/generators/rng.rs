/// 基于种子的简单确定性 PRNG (Xorshift64)
pub(crate) struct SeedRng {
    state: u64,
    /// Box-Muller 缓存：是否有缓存的正态样本
    has_spare: bool,
    /// Box-Muller 缓存：缓存的第二个正态样本
    spare: f64,
}

impl SeedRng {
    /// 从种子创建 PRNG，种子为 0 时使用默认种子
    pub fn new(seed: u64) -> Self {
        let seed = if seed == 0 {
            0xDEAD_BEEF_CAFE_BABE
        } else {
            seed
        };
        SeedRng {
            state: seed,
            has_spare: false,
            spare: 0.0,
        }
    }

    /// 生成下一个 u64 随机数
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// 生成 [0, 1) 范围的 f64
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// 生成 [min, max) 范围的 f64
    pub fn next_f64_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }

    /// 生成 [0, max) 范围的 usize
    pub fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u64() as usize) % max
    }

    /// 生成随机布尔值
    ///
    /// 使用最高有效位（MSB），因为 Xorshift64 的 MSB 比 LSB
    /// 具有更好的统计特性。
    pub fn next_bool(&mut self) -> bool {
        (self.next_u64() >> 63) & 1 == 1
    }

    /// 使用 Box-Muller 变换生成标准正态分布 N(0, 1) 样本
    ///
    /// 每次调用消耗 2 个 Uniform(0,1) 样本，产生 2 个正态样本，
    /// 缓存第二个供下次调用使用。
    pub fn next_normal(&mut self) -> f64 {
        if self.has_spare {
            self.has_spare = false;
            return self.spare;
        }

        let u1: f64;
        let u2: f64;
        loop {
            let raw = self.next_f64();
            if raw > 1e-30 {
                u1 = raw;
                u2 = self.next_f64();
                break;
            }
            // u1 过小会导致 ln(0) 下溢，重新采样
        }

        let mag = (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f64::consts::PI * u2).cos();
        let z1 = mag * (2.0 * std::f64::consts::PI * u2).sin();

        self.has_spare = true;
        self.spare = z1;

        z0
    }
}
