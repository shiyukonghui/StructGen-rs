/// 基于种子的简单确定性 PRNG (Xorshift64)
pub(crate) struct SeedRng {
    state: u64,
}

impl SeedRng {
    /// 从种子创建 PRNG，种子为 0 时使用默认种子
    pub fn new(seed: u64) -> Self {
        let seed = if seed == 0 {
            0xDEAD_BEEF_CAFE_BABE
        } else {
            seed
        };
        SeedRng { state: seed }
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
    pub fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}
