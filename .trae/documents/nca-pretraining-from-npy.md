# NCA 预训练数据适配计划

## 分析结果

### 数据格式兼容性

**训练代码期望的数据格式**：
- `generate_nca_dataset` 生成 `(B, T, H, W, C)` 形状的 JAX 数组
- 通过 `NCA_Tokenizer.encode_task()` 编码为 token 序列
- 训练使用 LLaMA 模型，输入是 tokenized 序列

**我们生成的 NpyBatch 数据格式**：
- `NpyBatchAdapter` 输出 `(B, T, H, W, C)` 形状的 NumPy 数组
- **完全兼容！**

### 参数匹配

| 参数 | 训练代码默认值 | 我们生成的数据 | 匹配 |
|------|---------------|---------------|------|
| grid (rows/cols) | 12 | 12 | ✅ |
| num_colors (channels) | 10 | 10 | ✅ |
| num_examples (num_frames) | 10 | 10 | ✅ |
| patch | 3 | - | 需配置 |

### 关键差异

1. **数据来源**：训练代码使用 JAX 实时生成，我们使用 Rust 预生成 `.npy` 文件
2. **数据加载**：需要编写适配脚本从 `.npy` 文件加载
3. **Tokenizer**：需要使用相同的 `NCA_Tokenizer` 进行编码

## 实施计划

### Task 1: 创建数据适配脚本

编写 `train_from_npy.py` 脚本，实现：
1. 从 `output/batch_generation/nca2d_batch_*.npy` 加载数据
2. 使用 `NCA_Tokenizer` 进行编码
3. 创建 PyTorch `NCADataset` 和 `DataLoader`

### Task 2: 配置训练参数

根据生成的数据配置训练参数：
- `grid=12, num_colors=10, patch=3`
- `seq_len` 根据 tokenizer 计算
- `vocab_size=16384` (默认)

### Task 3: 运行训练

使用适配脚本进行训练：
- 使用 LLaMA 模型
- 配置合适的 batch_size 和学习率
- 监控训练进度

## 技术细节

### NCA_Tokenizer 编码逻辑

```python
# 输入: (B, T, H, W, C) 的网格数据
# 输出: (B, seq_len) 的 token 序列

# 编码步骤:
# 1. 将网格分割为 patch×patch 的块
# 2. 每个块转换为单个 token (num_colors^(patch²) 种可能)
# 3. 添加 start/end token
# 4. 展平为序列
```

### 数据加载流程

```
.npy 文件 → numpy.load → jnp.array → tokenizer.encode_task → torch.Tensor → DataLoader
```

### 文件结构

```
Code/nca-pre-pretraining-main/
├── src/
│   ├── nca_ppt.py          # 原训练脚本
│   └── train_from_npy.py   # 新适配脚本 (待创建)
├── utils/
│   ├── tokenizers.py       # NCA_Tokenizer
│   ├── training_args.py    # NCATrainingArgs
│   └── models.py           # LLaMA 模型
│   └── util.py             # 工具函数
```

## 预期结果

1. 成功加载预生成的 `.npy` 数据
2. 使用 LLaMA 模型进行 NCA 预训练
3. 训练 loss 正常下降
4. 可保存和加载 checkpoint