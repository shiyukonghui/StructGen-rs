//! Apache Parquet 输出适配器实现
//!
//! 将帧序列写入 Parquet 列式文件，适合大规模数据分析与高效随机访问。
//! 使用 Arrow/Parquet crate 按列组织数据，支持 Snappy 压缩。

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::core::*;

use super::adapter::{format_output_filename, OutputConfig, OutputStats, SinkAdapter};

use arrow::array::{Array, ArrayRef, BinaryArray, Int32Array, Int64Array, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

/// Parquet 列式输出适配器
///
/// 将 SequenceFrame 序列写入 Apache Parquet 文件。
/// 每帧转换为 RecordBatch 中的一行，列结构为：
/// - step_index: Int64
/// - state_dim: Int32
/// - state_values: Binary（每项 9 字节: type_tag + payload）
/// - label: Utf8 (optional)
pub struct ParquetAdapter {
    /// 最终输出文件路径
    output_path: Option<PathBuf>,
    /// 临时文件路径（写入期间使用 .tmp 后缀）
    tmp_path: Option<PathBuf>,
    /// Arrow Schema（在 open 时构建）
    schema: Option<Arc<Schema>>,
    /// Parquet ArrowWriter 实例
    writer: Option<ArrowWriter<File>>,
    /// 已写入帧数
    frames_written: u64,
    /// 输出配置副本
    config: Option<OutputConfig>,
    /// 所有序列化数据的累计字节数
    bytes_written: u64,
}

impl ParquetAdapter {
    /// 创建新的 ParquetAdapter 实例
    pub fn new() -> Self {
        ParquetAdapter {
            output_path: None,
            tmp_path: None,
            schema: None,
            writer: None,
            frames_written: 0,
            config: None,
            bytes_written: 0,
        }
    }

    /// 定义 Parquet 文件的 Arrow Schema
    fn build_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("step_index", DataType::Int64, false),
            Field::new("state_dim", DataType::Int32, false),
            Field::new("state_values", DataType::Binary, false),
            Field::new("label", DataType::Utf8, true),
        ]))
    }

    /// 将 FrameState 序列化为 9 字节：1 字节类型标签 + 8 字节数据
    fn serialize_frame_state(state: &FrameState) -> [u8; 9] {
        let mut buf = [0u8; 9];
        match state {
            FrameState::Integer(v) => {
                buf[0] = 0x01; // 类型标签：整型
                buf[1..9].copy_from_slice(&v.to_le_bytes());
            }
            FrameState::Float(v) => {
                buf[0] = 0x02; // 类型标签：浮点型
                buf[1..9].copy_from_slice(&v.to_le_bytes());
            }
            FrameState::Bool(v) => {
                buf[0] = 0x03; // 类型标签：布尔型
                buf[1] = *v as u8;
                // 剩余 7 字节保持为零
            }
        }
        buf
    }

    /// 将 SequenceFrame 转换为 RecordBatch 单行
    fn frame_to_record_batch(
        &self,
        frame: &SequenceFrame,
        schema: &Arc<Schema>,
    ) -> CoreResult<RecordBatch> {
        let step_index = Int64Array::from(vec![frame.step_index as i64]);
        let state_dim = frame.state.dim() as i32;
        let state_dim_array = Int32Array::from(vec![state_dim]);

        // 序列化所有 FrameState 值到字节数组
        let serialized: Vec<u8> = frame
            .state
            .values
            .iter()
            .flat_map(|v| Self::serialize_frame_state(v).to_vec())
            .collect();
        let state_values = BinaryArray::from_vec(vec![&serialized[..]]);

        // 标签列（可选）
        let mut label_builder = StringBuilder::new();
        match &frame.label {
            Some(l) => label_builder.append_value(l),
            None => label_builder.append_null(),
        }
        let label_array = label_builder.finish();

        let batch = RecordBatch::try_new(
            Arc::clone(schema),
            vec![
                Arc::new(step_index) as ArrayRef,
                Arc::new(state_dim_array) as ArrayRef,
                Arc::new(state_values) as ArrayRef,
                Arc::new(label_array) as ArrayRef,
            ],
        )
        .map_err(|e| CoreError::SinkError(format!("构建 RecordBatch 失败: {}", e)))?;

        Ok(batch)
    }

    /// 计算文件的 SHA-256 哈希
    fn compute_file_hash(path: &Path) -> CoreResult<String> {
        let data = fs::read(path).map_err(|e| {
            CoreError::SinkError(format!("读取文件进行哈希计算失败: {}", e))
        })?;
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let result = hasher.finalize();
        Ok(format!("{:x}", result))
    }
}

impl Default for ParquetAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl SinkAdapter for ParquetAdapter {
    fn format(&self) -> OutputFormat {
        OutputFormat::Parquet
    }

    fn open(
        &mut self,
        base_dir: &Path,
        task_name: &str,
        shard_id: usize,
        seed: u64,
        config: &OutputConfig,
    ) -> CoreResult<()> {
        // 确保输出目录存在
        fs::create_dir_all(base_dir)?;

        // 构造文件名
        let tmp_name = format_output_filename(task_name, shard_id, seed, "parquet.tmp");
        let final_name = format_output_filename(task_name, shard_id, seed, "parquet");
        let tmp_path = base_dir.join(&tmp_name);
        let output_path = base_dir.join(&final_name);

        // 如果临时文件已存在，先删除
        if tmp_path.exists() {
            fs::remove_file(&tmp_path)?;
        }

        // 创建 Arrow Schema
        let schema = Self::build_schema();

        // 配置 Writer 属性
        let compression = match config.compression_level {
            0 => Compression::UNCOMPRESSED,
            _ => Compression::SNAPPY,
        };
        let props = WriterProperties::builder()
            .set_compression(compression)
            .build();

        // 创建 ArrowWriter
        let file = File::create(&tmp_path)?;
        let writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props))
            .map_err(|e| CoreError::SinkError(format!("创建 Parquet writer 失败: {}", e)))?;

        self.output_path = Some(output_path);
        self.tmp_path = Some(tmp_path);
        self.schema = Some(schema);
        self.writer = Some(writer);
        self.frames_written = 0;
        self.config = Some(config.clone());
        self.bytes_written = 0;

        Ok(())
    }

    fn write_frame(&mut self, frame: &SequenceFrame) -> CoreResult<()> {
        let schema = self
            .schema
            .as_ref()
            .ok_or_else(|| CoreError::SinkError("适配器未打开：schema 为空".into()))?;

        // 先构建 RecordBatch（只读 self），避免与 writer 的借用冲突
        let batch = self.frame_to_record_batch(frame, schema)?;

        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| CoreError::SinkError("适配器未打开：writer 为空".into()))?;

        writer
            .write(&batch)
            .map_err(|e| CoreError::SinkError(format!("写入 Parquet 失败: {}", e)))?;

        self.frames_written += 1;
        self.bytes_written += batch.get_array_memory_size() as u64;

        Ok(())
    }

    fn close(&mut self) -> CoreResult<OutputStats> {
        let writer = self
            .writer
            .take()
            .ok_or_else(|| CoreError::SinkError("适配器未打开或已关闭".into()))?;

        // 关闭 writer，写入 footer
        writer
            .close()
            .map_err(|e| CoreError::SinkError(format!("关闭 Parquet writer 失败: {}", e)))?;

        let tmp_path = self
            .tmp_path
            .as_ref()
            .ok_or_else(|| CoreError::SinkError("临时路径为空".into()))?;
        let output_path = self
            .output_path
            .as_ref()
            .ok_or_else(|| CoreError::SinkError("输出路径为空".into()))?;

        // 将临时文件重命名为最终文件
        fs::rename(tmp_path, output_path)?;

        // 获取文件大小
        let file_meta =
            fs::metadata(output_path).map_err(|e| CoreError::SinkError(format!("获取文件元数据失败: {}", e)))?;
        let bytes_written = file_meta.len();

        // 计算 SHA-256 哈希（如果配置要求）
        let file_hash = if self.config.as_ref().is_some_and(|c| c.compute_hash) {
            Some(Self::compute_file_hash(output_path)?)
        } else {
            None
        };

        Ok(OutputStats {
            frames_written: self.frames_written,
            bytes_written,
            output_path: Some(output_path.clone()),
            file_hash,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{FrameData, FrameState, SequenceFrame};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
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
                let label = if i % 3 == 0 {
                    Some(format!("step_{}", i))
                } else {
                    None
                };
                SequenceFrame {
                    step_index: i as u64,
                    state: data,
                    label,
                }
            })
            .collect()
    }

    fn default_config() -> OutputConfig {
        OutputConfig::default()
    }

    #[test]
    fn test_parquet_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(100);

        // 写入 Parquet
        let mut adapter = ParquetAdapter::new();
        adapter
            .open(tmp.path(), "test", 1, 0xABCD, &default_config())
            .unwrap();
        for f in &frames {
            adapter.write_frame(f).unwrap();
        }
        let stats = adapter.close().unwrap();

        assert_eq!(stats.frames_written, 100);
        assert!(stats.bytes_written > 0);
        assert!(stats.output_path.is_some());

        // 读取并验证
        let output_path = stats.output_path.unwrap();
        assert!(output_path.exists());

        let file = File::open(&output_path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let reader = builder.build().unwrap();

        let mut read_frames: Vec<SequenceFrame> = Vec::new();
        for batch_result in reader {
            let batch = batch_result.unwrap();

            let step_index_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let state_dim_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let state_values_col = batch
                .column(2)
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap();
            let label_col = batch
                .column(3)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();

            for row in 0..batch.num_rows() {
                let step_index = step_index_col.value(row) as u64;
                let _dim = state_dim_col.value(row);
                let raw = state_values_col.value(row);

                // 反序列化 FrameState 值（每 9 字节一组）
                let mut values = Vec::new();
                for chunk in raw.chunks(9) {
                    if chunk.len() < 9 {
                        break;
                    }
                    let tag = chunk[0];
                    let data = &chunk[1..9];
                    let mut payload = [0u8; 8];
                    payload.copy_from_slice(data);
                    let state = match tag {
                        0x01 => FrameState::Integer(i64::from_le_bytes(payload)),
                        0x02 => FrameState::Float(f64::from_le_bytes(payload)),
                        0x03 => FrameState::Bool(payload[0] != 0),
                        _ => panic!("未知类型标签: {}", tag),
                    };
                    values.push(state);
                }

                let label = if label_col.is_null(row) {
                    None
                } else {
                    Some(label_col.value(row).to_string())
                };

                read_frames.push(SequenceFrame {
                    step_index,
                    state: FrameData { values },
                    label,
                });
            }
        }

        assert_eq!(read_frames.len(), 100);
        // 验证第一帧
        assert_eq!(read_frames[0].step_index, frames[0].step_index);
        assert_eq!(read_frames[0].state.values.len(), frames[0].state.values.len());
        assert_eq!(read_frames[0].label, frames[0].label);
        // 验证最后一帧
        assert_eq!(read_frames[99].step_index, frames[99].step_index);
        assert_eq!(read_frames[99].state.values.len(), frames[99].state.values.len());
    }

    #[test]
    fn test_parquet_with_hash() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(10);

        let mut adapter = ParquetAdapter::new();
        let mut config = default_config();
        config.compute_hash = true;
        adapter.open(tmp.path(), "hash_test", 0, 42, &config).unwrap();

        for f in &frames {
            adapter.write_frame(f).unwrap();
        }
        let stats = adapter.close().unwrap();

        assert!(stats.file_hash.is_some());
        let hash = stats.file_hash.unwrap();
        assert_eq!(hash.len(), 64); // SHA-256 产生 64 个十六进制字符
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_parquet_empty() {
        let tmp = TempDir::new().unwrap();
        let mut adapter = ParquetAdapter::new();
        adapter
            .open(tmp.path(), "empty", 0, 0, &default_config())
            .unwrap();
        let stats = adapter.close().unwrap();

        assert_eq!(stats.frames_written, 0);
        assert!(stats.bytes_written > 0); // 空 Parquet 文件仍包含 schema 元数据
        assert!(stats.output_path.is_some());
    }

    #[test]
    fn test_parquet_write_batch() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(50);

        let mut adapter = ParquetAdapter::new();
        adapter
            .open(tmp.path(), "batch", 1, 0, &default_config())
            .unwrap();
        adapter.write_batch(&frames).unwrap();
        let stats = adapter.close().unwrap();

        assert_eq!(stats.frames_written, 50);
    }

    #[test]
    fn test_parquet_close_without_open_panics() {
        let mut adapter = ParquetAdapter::new();
        let result = adapter.close();
        assert!(result.is_err());
    }

    #[test]
    fn test_serialize_frame_state() {
        let int_state = FrameState::Integer(-12345);
        let bytes = ParquetAdapter::serialize_frame_state(&int_state);
        assert_eq!(bytes[0], 0x01);
        assert_eq!(i64::from_le_bytes(bytes[1..9].try_into().unwrap()), -12345);

        let float_state = FrameState::Float(3.14159);
        let bytes = ParquetAdapter::serialize_frame_state(&float_state);
        assert_eq!(bytes[0], 0x02);

        let bool_state = FrameState::Bool(true);
        let bytes = ParquetAdapter::serialize_frame_state(&bool_state);
        assert_eq!(bytes[0], 0x03);
        assert_eq!(bytes[1], 1);
    }
}
