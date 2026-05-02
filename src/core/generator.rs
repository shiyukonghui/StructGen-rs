use super::error::CoreError;
use super::frame::SequenceFrame;
use super::params::GenParams;

/// 生成器抽象接口。所有具体生成器必须实现此 trait。
///
/// 该 trait 要求实现者为 Send + Sync，确保实例可在 rayon 线程间安全共享。
///
/// 生成器的构造不通过 trait 方法完成，而是通过注册在 [`super::registry::GeneratorRegistry`]
/// 中的工厂函数进行。每个生成器模块应提供一个符合 [`super::registry::GeneratorFactory`]
/// 签名的构造函数，并在注册表中注册。
pub trait Generator: Send + Sync {
    /// 返回该生成器的唯一标识名称
    fn name(&self) -> &'static str;

    /// 流式生成（推荐模式）。返回一个惰性迭代器，按时间步产出 SequenceFrame
    ///
    /// 迭代器在 `Some(seq_length)` 时最多产出 `seq_length` 个帧；
    /// 当 `seq_length == 0` 时无限产出，由调用方控制消费数量。
    ///
    /// # 生命周期约束
    /// 返回的迭代器具有 `'static` 生命周期，即不能借用 `&self` 或 `params`。
    /// 生成器实现需将必要状态克隆或移入迭代器闭包中。该约束使得迭代器
    /// 可安全地跨 rayon 线程传递，无需关心借用的生命周期。
    ///
    /// # Arguments
    /// * `seed` - 确定性随机种子，决定初始状态和随机性注入
    /// * `params` - 通用参数，从中提取 seq_length 等共用信息
    fn generate_stream(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> Result<Box<dyn Iterator<Item = SequenceFrame> + Send>, CoreError>;

    /// 批量生成（同步糖）。内部调用 generate_stream 并收集为 Vec
    ///
    /// 仅适用于中小规模数据；大规模请使用流式接口。
    /// 要求 `params.seq_length > 0`，否则返回错误（因为无限流无法收集为 Vec）。
    ///
    /// # Arguments
    /// * `seed` - 确定性随机种子
    /// * `params` - 通用参数
    ///
    /// # Errors
    /// 当 `params.seq_length == 0` 时返回 [`CoreError::InvalidParams`]
    fn generate_batch(
        &self,
        seed: u64,
        params: &GenParams,
    ) -> Result<Vec<SequenceFrame>, CoreError> {
        if params.seq_length == 0 {
            return Err(CoreError::InvalidParams(
                "generate_batch requires seq_length > 0; use generate_stream for unbounded generation".into(),
            ));
        }
        self.generate_stream(seed, params)
            .map(|iter| iter.collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用 mock 生成器，产出固定序列的帧
    struct MockGenerator {
        state_dim: usize,
    }

    impl Generator for MockGenerator {
        fn name(&self) -> &'static str {
            "mock"
        }

        fn generate_stream(
            &self,
            _seed: u64,
            params: &GenParams,
        ) -> Result<Box<dyn Iterator<Item = SequenceFrame> + Send>, CoreError> {
            let limit = if params.seq_length == 0 {
                10
            } else {
                params.seq_length
            };
            let dim = self.state_dim;
            let iter = (0..limit).map(move |i| {
                let values = (0..dim)
                    .map(|j| super::super::frame::FrameState::Integer(
                        (i as u64 * dim as u64 + j as u64) as i64,
                    ))
                    .collect();
                SequenceFrame::new(i as u64, super::super::frame::FrameData { values })
            });
            Ok(Box::new(iter))
        }
    }

    #[test]
    fn test_generator_name() {
        let gen = MockGenerator { state_dim: 3 };
        assert_eq!(gen.name(), "mock");
    }

    #[test]
    fn test_generator_stream_with_seq_length() {
        let gen = MockGenerator { state_dim: 2 };
        let params = GenParams::simple(5);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 5);
        assert_eq!(frames[0].step_index, 0);
        assert_eq!(frames[4].step_index, 4);
    }

    #[test]
    fn test_generator_batch() {
        let gen = MockGenerator { state_dim: 1 };
        let params = GenParams::simple(3);
        let frames = gen.generate_batch(0, &params).unwrap();
        assert_eq!(frames.len(), 3);
    }

    #[test]
    fn test_generator_stream_zero_length() {
        let gen = MockGenerator { state_dim: 1 };
        let params = GenParams::simple(0);
        let frames = gen.generate_stream(0, &params).unwrap().collect::<Vec<_>>();
        assert_eq!(frames.len(), 10); // 默认限制 10
    }

    #[test]
    fn test_generator_batch_zero_length_rejected() {
        let gen = MockGenerator { state_dim: 1 };
        let params = GenParams::simple(0);
        let result = gen.generate_batch(0, &params);
        assert!(result.is_err());
        match result.unwrap_err() {
            CoreError::InvalidParams(msg) => {
                assert!(msg.contains("seq_length > 0"));
            }
            other => panic!("expected InvalidParams, got {:?}", other),
        }
    }
}
