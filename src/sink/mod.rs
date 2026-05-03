//! StructGen-rs 输出适配层（sink）
//!
//! 本模块负责将后处理管道产出的 SequenceFrame 序列转换为特定文件格式并写入磁盘。
//!
//! 支持三种输出格式：
//! - Parquet：列式存储，适合大规模数据分析
//! - Text：Unicode 文本，适合语言模型 DataLoader 直接加载
//! - Binary：紧凑二进制，适合 mmap 随机访问

pub mod adapter;
pub mod binary;
pub mod factory;
pub mod parquet;
pub mod text;

// 重导出适配器类型
pub use adapter::{format_output_filename, OutputConfig, SinkAdapter};
// 重导出工厂类型
pub use factory::SinkAdapterFactory;
// 重导出具体适配器实现
pub use binary::BinaryAdapter;
pub use parquet::ParquetAdapter;
pub use text::TextAdapter;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{FrameData, FrameState, SequenceFrame};
    use tempfile::TempDir;

    /// 构造测试用帧序列
    fn make_test_frames(n: usize) -> Vec<SequenceFrame> {
        (0..n)
            .map(|i| {
                let values: Vec<FrameState> = vec![
                    FrameState::Integer(i as i64),
                    FrameState::Float(i as f64 * 0.5),
                    FrameState::Bool(i % 2 == 0),
                ];
                let data: FrameData = values.into_iter().collect();
                SequenceFrame::new(i as u64, data)
            })
            .collect()
    }

    fn default_config() -> OutputConfig {
        OutputConfig::default()
    }

    /// 原子写入测试：关闭后不应残留 .tmp 临时文件
    #[test]
    fn test_atomic_write_leaves_no_tmp_on_success_parquet() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(5);

        let mut adapter = ParquetAdapter::new();
        adapter
            .open(tmp.path(), "atomic_pq", 1, 0x42, &default_config())
            .unwrap();
        adapter.write_batch(&frames).unwrap();
        let _stats = adapter.close().unwrap();

        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert!(!entries.is_empty());
        for entry in &entries {
            let name = entry.file_name();
            let name_str = name.to_str().unwrap();
            assert!(
                !name_str.ends_with(".tmp"),
                "残留临时文件: {}",
                name_str
            );
        }
    }

    #[test]
    fn test_atomic_write_leaves_no_tmp_on_success_text() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(5);

        let mut adapter = TextAdapter::new();
        adapter
            .open(tmp.path(), "atomic_txt", 2, 0x99, &default_config())
            .unwrap();
        adapter.write_batch(&frames).unwrap();
        let _stats = adapter.close().unwrap();

        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        for entry in &entries {
            let name_str = entry.file_name().to_str().unwrap().to_string();
            assert!(!name_str.ends_with(".tmp"), "残留临时文件: {}", name_str);
        }
    }

    #[test]
    fn test_atomic_write_leaves_no_tmp_on_success_binary() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(5);

        let mut adapter = BinaryAdapter::new();
        adapter
            .open(tmp.path(), "atomic_bin", 3, 0xABCD, &default_config())
            .unwrap();
        adapter.write_batch(&frames).unwrap();
        let _stats = adapter.close().unwrap();

        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        for entry in &entries {
            let name_str = entry.file_name().to_str().unwrap().to_string();
            assert!(!name_str.ends_with(".tmp"), "残留临时文件: {}", name_str);
        }
    }
}
