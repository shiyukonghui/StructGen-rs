# StructGen-rs 输出适配层 (sink) 详细设计规格

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| 1.0 | 2026-05-01 | - | 初始版本，定义输出适配器接口及 Parquet/文本/二进制三种输出实现 |

## 1. 模块概述

输出适配层（sink）是 StructGen-rs 系统的数据持久化层。它接收后处理管道产出的 `SequenceFrame` 序列，将其转换为特定文件格式并写入磁盘，同时管理输出文件的分片、命名和统计。

该模块的职责包括：
- 定义统一的输出适配器接口（`SinkAdapter` trait），使调度器可以透明地切换输出格式。
- 实现三种内置输出格式：Apache Parquet（列式存储）、纯文本（Unicode 令牌序列）、二进制（内存映射原始转储）。
- 管理输出分片：根据样本数量或文件大小上限自动拆分输出文件。
- 记录每个输出分片的写入统计（帧数、字节数），供元数据层汇总。

**核心原则**：适配器不关心数据是什么——它只是将帧序列以指定格式序列化到磁盘。适配器需要支持并发写入（多个分片并行输出到不同文件）。

## 2. 设计目标与原则

- **格式透明**：调度器通过 trait 对象调用适配器，不感知底层文件格式。
- **流式写入**：逐帧写入（而非先收集到内存再序列化），保证 TB 级数据输出不会 OOM。
- **并发安全**：不同分片的适配器实例写入不同文件，天然无锁；若需写入同一文件，使用 `Mutex` 或消息通道。
- **原子写入**：每个分片文件先写入临时文件，关闭时重命名为最终文件名，避免写入中断产生残缺文件。
- **可恢复性**：文件名包含任务名、分片索引和种子，支持快速定位与恢复。

## 3. 模块内部结构组织

```
src/sink/
├── mod.rs          # 模块根，重导出类型，register_all()
├── adapter.rs      # SinkAdapter trait、OutputStats、OutputConfig 定义
├── parquet.rs      # ParquetAdapter 实现
├── text.rs         # TextAdapter 实现（纯文本输出）
├── binary.rs       # BinaryAdapter 实现（原始二进制 + 内存映射）
└── factory.rs      # SinkAdapterFactory 工厂类型
```

| 文件 | 职责 |
|------|------|
| `adapter.rs` | 定义 `SinkAdapter` trait、`OutputStats`、`OutputConfig`、分片命名规则 |
| `parquet.rs` | `ParquetAdapter`：使用 Apache Arrow/Parquet 库按列写出帧数据 |
| `text.rs` | `TextAdapter`：将已令牌映射的整数值写出为 Unicode 文本文件 |
| `binary.rs` | `BinaryAdapter`：将帧数据以原始二进制格式转储，支持内存映射读取 |
| `factory.rs` | `SinkAdapterFactory` 类型别名，用于按 `OutputFormat` 创建适配器实例 |

## 4. 公开接口定义

### 4.1 SinkAdapter trait

```rust
use std::path::{Path, PathBuf};
use crate::core::{CoreResult, SequenceFrame, OutputFormat};

/// 输出适配器的抽象接口。
///
/// 生命周期：`open()` → 多次 `write_frame()` → `close()`。
pub trait SinkAdapter: Send {
    /// 返回此适配器对应的输出格式。
    fn format(&self) -> OutputFormat;

    /// 打开输出适配器，准备写入。
    ///
    /// # Arguments
    /// * `base_dir` - 输出根目录。
    /// * `task_name` - 所属任务名称（用于构造输出文件名）。
    /// * `shard_id` - 分片编号（用于构造输出文件名）。
    /// * `seed` - 本分片的种子（用于构造输出文件名，保证可追溯）。
    /// * `config` - 输出配置（压缩级别、分片文件大小上限等）。
    fn open(
        &mut self,
        base_dir: &Path,
        task_name: &str,
        shard_id: usize,
        seed: u64,
        config: &OutputConfig,
    ) -> CoreResult<()>;

    /// 写入单帧。
    fn write_frame(&mut self, frame: &SequenceFrame) -> CoreResult<()>;

    /// 批量写入（默认逐帧调用 write_frame，子类可覆盖优化）。
    fn write_batch(&mut self, frames: &[SequenceFrame]) -> CoreResult<()> {
        for frame in frames {
            self.write_frame(frame)?;
        }
        Ok(())
    }

    /// 关闭适配器，完成写出并返回写入统计。
    ///
    /// 关闭后适配器不可再使用。此方法负责：
    /// - 刷新内部缓冲区
    /// - 写入文件尾部（如 Parquet footer）
    /// - 将临时文件重命名为最终文件名
    /// - 返回 `OutputStats`
    fn close(&mut self) -> CoreResult<OutputStats>;
}

/// 适配器构造函数类型：接受 OutputFormat，返回 SinkAdapter trait 对象。
pub type SinkAdapterFactory = fn(OutputFormat) -> CoreResult<Box<dyn SinkAdapter>>;
```

### 4.2 辅助类型

```rust
/// 输出统计信息。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutputStats {
    /// 写入的帧总数。
    pub frames_written: u64,
    /// 写入的物理字节数（文件大小）。
    pub bytes_written: u64,
    /// 最终输出文件路径（重命名后）。
    pub output_path: Option<PathBuf>,
    /// 输出文件的 SHA-256 哈希（关闭时计算，默认不启用）。
    pub file_hash: Option<String>,
}

/// 输出配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// 压缩级别 (0 = 不压缩, 9 = 最大压缩)。
    #[serde(default = "default_compression")]
    pub compression_level: u32,
    /// 单个分片文件的最大字节数，超出后自动切分。None 表示不限制。
    #[serde(default)]
    pub max_file_bytes: Option<u64>,
    /// 单个分片文件的最大帧数，超出后自动切分。None 表示不限制。
    #[serde(default)]
    pub max_frames_per_file: Option<u64>,
    /// 是否在文件关闭时计算 SHA-256。
    #[serde(default)]
    pub compute_hash: bool,
}

fn default_compression() -> u32 { 6 }

/// 输出文件命名规则：
///
/// `<task_name>_<shard_id:05>_<seed:016x>.<extension>`
///
/// 例如：
/// - `rule30_ca_00001_0000000000003039.parquet`
/// - `lorenz_chaos_00042_0000000000010932.txt`
pub fn format_output_filename(task_name: &str, shard_id: usize, seed: u64, ext: &str) -> String {
    format!("{}_{:05}_{:016x}.{}", task_name, shard_id, seed, ext)
}
```

## 5. 核心逻辑详解

### 5.1 ParquetAdapter —— Parquet 输出

**功能**：将帧序列写入 Apache Parquet 列式文件，适合大规模数据分析与高效随机访问。

**Schema 设计**：

```
Parquet 文件 Schema:
  ├── step_index: INT64 (required)        -- 时间步索引
  ├── state_dim: INT32 (required)          -- 状态维度（values 长度）
  ├── state_values: BYTE_ARRAY (required)  -- 序列化后的 FrameState 值列表
  │     （内部格式：[type_tag:u8][payload:8bytes] 每项 9 字节）
  └── label: BYTE_ARRAY (optional)         -- 语义标签文本
```

**写入流程**：

```
open(base_dir, task_name, shard_id, seed, config):
    1. 构造临时文件名: format_output_filename(task_name, shard_id, seed, "parquet.tmp")
    2. 构造最终文件名: format_output_filename(task_name, shard_id, seed, "parquet")
    3. 创建 Arrow Schema（如上所述）
    4. 创建 Parquet Arrow Writer，设置压缩级别 = config.compression_level
    5. 保存临时路径和最终路径到实例字段

write_frame(frame):
    1. 将 frame 转换为 Arrow RecordBatch（单行）
    2. 调用 writer.write(&batch)
    3. 更新内部计数器 frames_written

close():
    1. 调用 writer.close()，将 Parquet footer 写入文件
    2. 如果 config.compute_hash:
        - 对临时文件计算 SHA-256
    3. 将临时文件重命名为最终文件名
    4. 获取文件大小
    5. 返回 OutputStats { frames_written, bytes_written, output_path, file_hash }
```

### 5.2 TextAdapter —— 纯文本输出

**功能**：将已令牌映射的帧序列以 Unicode 文本格式写出，可直接被语言模型的 DataLoader 加载。

**写入格式**：

```
文件内容格式：
  <char_for_value_0><char_for_value_1>...<char_for_value_N>\n
  <char_for_value_0><char_for_value_1>...\n
  ...
  每行 = 一帧的所有状态值映射后的字符，换行分隔。

  内部编码：UTF-8。
```

**写入流程**：

```
open(base_dir, task_name, shard_id, seed, config):
    1. 构造文件名: format_output_filename(task_name, shard_id, seed, "txt.tmp")
    2. 创建 BufWriter<File>，缓冲区大小 64KB
    3. 保存路径到实例字段

write_frame(frame):
    1. 对 frame.state.values 中的每个 FrameState：
       - Integer(v): 将 v 作为 u32 映射为 Unicode 字符 char::from_u32(v as u32)
       - Float(v): 将 v 映射到 0-255 整数再映射为字符
       - Bool(v): 映射为 '0' 或 '1'
    2. 将所有字符拼接为 String
    3. 如果 frame.label 非空，拼接标签文本
    4. 追加 '\n'
    5. BufWriter::write_all 写入

close():
    1. BufWriter::flush()
    2. 获取文件大小
    3. 重命名临时文件
    4. 返回 OutputStats
```

### 5.3 BinaryAdapter —— 二进制输出

**功能**：将帧序列以紧凑的二进制格式转储，支持后续通过 `mmap` 进行高效随机访问。

**文件格式**：

```
二进制文件格式（小端序）：
  ┌──────────────────────────────────┐
  │ Header (16 bytes)                │
  │   magic:       [u8; 4]  "SGEN"   │
  │   version:     u32      = 1      │
  │   frame_count: u64      = N      │
  │   state_dim:   u32               │
  ├──────────────────────────────────┤
  │ Frame 0                          │
  │   step_index: u64                │
  │   values:     [u8; state_dim * 9]│  (每值 9 字节: 1 字节类型标签 + 8 字节数据)
  │   label_len:  u32 (0 表示无标签)│
  │   [label:     u8; label_len]     │
  ├──────────────────────────────────┤
  │ Frame 1                          │
  │   ...                            │
  ├──────────────────────────────────┤
  │ Frame N-1                        │
  └──────────────────────────────────┘

  每帧大小 = 8 + state_dim * 9 + 4 + label.len()
```

**写入流程**：

```
open(base_dir, task_name, shard_id, seed, config):
    1. 构造文件名: format_output_filename(task_name, shard_id, seed, "bin.tmp")
    2. 创建 BufWriter<File>
    3. 写入 header（frame_count = 0 占位，将在 close() 时回填）

write_frame(frame):
    1. 序列化 step_index (u64 LE)
    2. 对每个 FrameState，写入 1 字节类型标签 + 8 字节数据：
       - Integer(v) → tag=0x01, data=v.to_le_bytes()
       - Float(v)   → tag=0x02, data=v.to_le_bytes()
       - Bool(v)    → tag=0x03, data=(v as u8).to_le_bytes() + 7 字节零填充
    3. 序列化 label: 4 字节长度 + UTF-8 字节
    4. 追加到内部字节缓冲区（64KB 批处理：攒够 64KB 后一次性写出）

close():
    1. 刷新缓冲区
    2. 回到文件起始，回填 header.frame_count
    3. 刷新并关闭文件
    4. 重命名临时文件
    5. 返回 OutputStats
```

## 6. 与其他模块的交互

### 6.1 依赖关系

```
sink
  ├── 依赖 core::{CoreResult, CoreError, SequenceFrame, FrameState, OutputFormat, OutputStats}
  └── 被 scheduler 调用（通过 SinkAdapter trait）
```

### 6.2 调度器中的调用方式

```rust
// scheduler/executor.rs
fn execute_shard(...) -> ShardResult {
    let output_format = task.output_format
        .unwrap_or(manifest.global.default_format);

    let mut adapter: Box<dyn SinkAdapter> = match output_format {
        OutputFormat::Parquet => Box::new(ParquetAdapter::new()),
        OutputFormat::Text    => Box::new(TextAdapter::new()),
        OutputFormat::Binary  => Box::new(BinaryAdapter::new()),
    };

    adapter.open(
        Path::new(&manifest.global.output_dir),
        &task.name,
        shard.shard_idx,
        shard.seed,
        &default_output_config(),
    )?;

    for frame in processed_stream {
        adapter.write_frame(&frame)?;
    }

    let stats = adapter.close()?;
    ShardResult { task_name: task.name.clone(), shard_idx: shard.shard_idx, stats, error: None }
}
```

### 6.3 输出文件命名与元数据关联

```
输出目录结构示例：
output/
├── rule30_ca_00001_0000000000003039.parquet
├── rule30_ca_00002_000000000000303a.parquet
├── lorenz_chaos_00001_0000000000010932.txt
├── lorenz_chaos_00002_0000000000010933.txt
└── metadata.json                    -- 元数据文件，列出所有生成文件及其哈希
```

文件名中的种子值直接对应分片的 `derive_seed(task.seed, shard_idx)` 结果，确保可完整复现。

## 7. 错误处理策略

| 错误情景 | 适配器行为 | 调度器处理 |
|----------|-----------|-----------|
| 输出目录不存在/无权限 | `open()` 返回 `CoreError::IoError` | 记录错误，终止当前 Shard |
| 写入期间磁盘满 | `write_frame()` 返回 `CoreError::IoError` | 记录错误，Shard 失败 |
| 临时文件重命名失败 | `close()` 返回 `CoreError::IoError` | Shard 失败，临时文件保留供调试 |
| Unicode 码点无效 | TextAdapter 钳位到有效范围，不出错 | 无影响 |
| Parquet schema 不匹配 | 开发阶段 static_assert 防止 | N/A |

**原子写入保证**：通过"写临时文件→重命名"策略，确保输出目录中只出现完整文件。中断的写入留下 `.tmp` 文件，可由清理脚本删除。

## 8. 性能考量

- **BufWriter 缓冲**：所有适配器使用 `BufWriter<File>`，缓冲区大小 64KB–256KB，减少系统调用次数。
- **Parquet 压缩**：使用 Snappy（默认）或 Gzip 压缩，列式格式天然适合状态向量的压缩。
- **BinaryAdapter 的批写入**：内部维护 64KB 缓冲区，攒满后一次性 `write_all`。避免逐帧小 IO。
- **BinaryAdapter 对 mmap 的友好性**：固定长度的帧头 + 确定的字节偏移，支持随机访问单帧而不解析全文件。
- **零序列化开销**（BinaryAdapter）：`FrameState` 的二进制表示与内存布局完全一致，写入直接 `as_bytes()` 拷贝。

## 9. 可测试性设计

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_test_frames(n: usize) -> Vec<SequenceFrame> { /* ... */ }

    #[test]
    fn test_parquet_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(100);

        // 写入
        let mut adapter = ParquetAdapter::new();
        adapter.open(tmp.path(), "test", 1, 0, &default_config()).unwrap();
        for f in &frames { adapter.write_frame(f).unwrap(); }
        let stats = adapter.close().unwrap();

        // 读取验证
        let read_back = read_parquet_frames(&stats.output_path.unwrap());
        assert_eq!(read_back.len(), 100);
        assert_eq!(read_back[0], frames[0]);
    }

    #[test]
    fn test_text_output_is_valid_utf8() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(10);
        let mut adapter = TextAdapter::new();
        adapter.open(tmp.path(), "test", 1, 0, &default_config()).unwrap();
        for f in &frames { adapter.write_frame(f).unwrap(); }
        let stats = adapter.close().unwrap();
        let content = std::fs::read_to_string(&stats.output_path.unwrap()).unwrap();
        assert!(content.lines().count() >= 10);
    }

    #[test]
    fn test_binary_header_is_valid() {
        let tmp = TempDir::new().unwrap();
        let frames = make_test_frames(5);
        let mut adapter = BinaryAdapter::new();
        adapter.open(tmp.path(), "test", 1, 0, &default_config()).unwrap();
        for f in &frames { adapter.write_frame(f).unwrap(); }
        let stats = adapter.close().unwrap();
        let raw = std::fs::read(&stats.output_path.unwrap()).unwrap();
        assert_eq!(&raw[0..4], b"SGEN");  // magic
    }

    #[test]
    fn test_atomic_write_leaves_no_tmp_on_success() {
        let tmp = TempDir::new().unwrap();
        // ...
        // 关闭后，.tmp 文件不应存在，只有最终文件
        let entries: Vec<_> = std::fs::read_dir(tmp.path()).unwrap()
            .filter_map(|e| e.ok()).collect();
        assert!(entries.iter().all(|e| !e.file_name().to_str().unwrap().ends_with(".tmp")));
    }

    #[test]
    fn test_filename_format_uniqueness() {
        let f1 = format_output_filename("task_a", 1, 0, "parquet");
        let f2 = format_output_filename("task_a", 2, 0, "parquet");
        assert_ne!(f1, f2);  // 不同分片，文件名不同
    }
}
```

## 10. 配置与参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `compression_level` | u32 | 6 | 压缩级别（Parquet 用 Snappy 时不适用） |
| `max_file_bytes` | Option\<u64\> | None | 分片文件最大字节数 |
| `max_frames_per_file` | Option\<u64\> | None | 分片文件最大帧数 |
| `compute_hash` | bool | false | 关闭时是否计算 SHA-256 |

通过上述配置，用户可精确控制输出的文件大小和完整性校验行为。

---

通过以上设计，输出适配层以格式透明、流式写入、并发安全的方式将结构化帧序列落地为高性能的持久文件，支持从快速文本验证到大规模列式分析的全场景需求。
