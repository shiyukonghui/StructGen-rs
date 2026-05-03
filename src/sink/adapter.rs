//! 输出适配器核心接口定义
//!
//! 本模块定义 SinkAdapter trait、输出统计与配置结构体，
//! 以及输出文件命名规则。所有具体适配器实现此 trait。

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::core::*;

/// 输出适配器的抽象接口
///
/// 生命周期：`open()` → 多次 `write_frame()` → `close()`
pub trait SinkAdapter: Send {
    /// 返回此适配器对应的输出格式
    fn format(&self) -> OutputFormat;

    /// 打开输出适配器，准备写入
    ///
    /// # 参数
    /// * `base_dir` - 输出根目录
    /// * `task_name` - 所属任务名称，用于构造输出文件名
    /// * `shard_id` - 分片编号，用于构造输出文件名
    /// * `seed` - 本分片的种子，保证可追溯
    /// * `config` - 输出配置（压缩级别、分片文件大小上限等）
    fn open(
        &mut self,
        base_dir: &Path,
        task_name: &str,
        shard_id: usize,
        seed: u64,
        config: &OutputConfig,
    ) -> CoreResult<()>;

    /// 写入单帧
    fn write_frame(&mut self, frame: &SequenceFrame) -> CoreResult<()>;

    /// 批量写入帧，默认逐帧调用 write_frame，子实现可覆盖以优化性能
    fn write_batch(&mut self, frames: &[SequenceFrame]) -> CoreResult<()> {
        for frame in frames {
            self.write_frame(frame)?;
        }
        Ok(())
    }

    /// 关闭适配器，完成写出并返回写入统计
    ///
    /// 关闭后适配器不可再使用。此方法负责：
    /// - 刷新内部缓冲区
    /// - 写入文件尾部（如 Parquet footer）
    /// - 将临时文件重命名为最终文件名
    /// - 返回 OutputStats
    fn close(&mut self) -> CoreResult<OutputStats>;
}

/// 输出统计信息，记录单个分片文件的写入结果
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutputStats {
    /// 写入的帧总数
    pub frames_written: u64,
    /// 写入的物理字节数（文件大小）
    pub bytes_written: u64,
    /// 最终输出文件路径（重命名后的正式文件）
    pub output_path: Option<PathBuf>,
    /// 输出文件的 SHA-256 哈希（仅在 compute_hash 为 true 时计算）
    pub file_hash: Option<String>,
}

/// 输出配置，控制分片文件的写入行为
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// 压缩级别 (0 = 不压缩, 9 = 最大压缩)，默认 6
    #[serde(default = "default_compression")]
    pub compression_level: u32,
    /// 单个分片文件的最大字节数，超出后自动切分。None 表示不限制
    #[serde(default)]
    pub max_file_bytes: Option<u64>,
    /// 单个分片文件的最大帧数，超出后自动切分。None 表示不限制
    #[serde(default)]
    pub max_frames_per_file: Option<u64>,
    /// 是否在文件关闭时计算 SHA-256 哈希
    #[serde(default)]
    pub compute_hash: bool,
}

fn default_compression() -> u32 {
    6
}

impl Default for OutputConfig {
    fn default() -> Self {
        OutputConfig {
            compression_level: default_compression(),
            max_file_bytes: None,
            max_frames_per_file: None,
            compute_hash: false,
        }
    }
}

/// 构造标准化的输出文件名
///
/// 格式：`{task_name}_{shard_id:05}_{seed:016x}.{ext}`
///
/// # 示例
/// - `rule30_ca_00001_0000000000003039.parquet`
/// - `lorenz_chaos_00042_0000000000010932.txt`
pub fn format_output_filename(task_name: &str, shard_id: usize, seed: u64, ext: &str) -> String {
    format!("{}_{:05}_{:016x}.{}", task_name, shard_id, seed, ext)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_output_filename_basic() {
        let f = format_output_filename("rule30_ca", 1, 0x3039, "parquet");
        assert_eq!(f, "rule30_ca_00001_0000000000003039.parquet");
    }

    #[test]
    fn test_format_output_filename_different_shard_ids_unique() {
        let f1 = format_output_filename("task_a", 1, 0, "parquet");
        let f2 = format_output_filename("task_a", 2, 0, "parquet");
        assert_ne!(f1, f2);
    }

    #[test]
    fn test_format_output_filename_different_seeds_unique() {
        let f1 = format_output_filename("task", 1, 100, "txt");
        let f2 = format_output_filename("task", 1, 200, "txt");
        assert_ne!(f1, f2);
    }

    #[test]
    fn test_format_output_filename_shard_id_padding() {
        let f = format_output_filename("task", 42, 0, "bin");
        assert!(f.contains("00042"));
    }

    #[test]
    fn test_format_output_filename_seed_hex_format() {
        let f = format_output_filename("task", 0, 255, "parquet");
        assert!(f.contains("00000000000000ff"));
    }

    #[test]
    fn test_output_config_default() {
        let config = OutputConfig::default();
        assert_eq!(config.compression_level, 6);
        assert!(config.max_file_bytes.is_none());
        assert!(config.max_frames_per_file.is_none());
        assert!(!config.compute_hash);
    }

    #[test]
    fn test_output_config_serialization() {
        let config = OutputConfig {
            compression_level: 9,
            max_file_bytes: Some(1024 * 1024),
            max_frames_per_file: Some(10000),
            compute_hash: true,
        };
        let json = serde_json::to_string(&config).unwrap();
        let restored: OutputConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.compression_level, 9);
        assert_eq!(restored.max_file_bytes, Some(1048576));
        assert_eq!(restored.max_frames_per_file, Some(10000));
        assert!(restored.compute_hash);
    }

    #[test]
    fn test_output_stats_default() {
        let stats = OutputStats::default();
        assert_eq!(stats.frames_written, 0);
        assert_eq!(stats.bytes_written, 0);
        assert!(stats.output_path.is_none());
        assert!(stats.file_hash.is_none());
    }

    #[test]
    fn test_output_stats_serialization() {
        let stats = OutputStats {
            frames_written: 100,
            bytes_written: 2048,
            output_path: Some(PathBuf::from("/tmp/test.parquet")),
            file_hash: Some("abc123".into()),
        };
        let json = serde_json::to_string(&stats).unwrap();
        let restored: OutputStats = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.frames_written, 100);
        assert_eq!(restored.bytes_written, 2048);
        assert!(restored.output_path.is_some());
        assert_eq!(restored.file_hash.unwrap(), "abc123");
    }
}
