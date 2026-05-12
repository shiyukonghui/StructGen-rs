# 元胞自动机输出格式对齐训练格式 Spec

## Why
Rust 版本的 NCA 生成器输出格式已经修改为匹配 Python 训练格式，但其他元胞自动机生成器（CA2D、CA3D、BooleanNetwork）的输出格式尚未对齐。这些生成器产生的数据需要经过适当的处理才能用于模型训练，包括 patch tokenization、序列串联和批量收集等步骤。

## What Changes
- **新增** CA2D 训练格式配置示例，展示如何使用 PatchTokenizer、SequenceStitcher 和 BatchCollector 处理器
- **新增** CA3D 训练格式配置示例，展示如何处理 3D 网格数据
- **新增** BooleanNetwork 训练格式配置示例，展示如何处理布尔网络数据
- **新增** 文档说明如何为不同类型的元胞自动机配置训练格式输出
- **修改** CA2D 生成器，添加默认参数以匹配训练需求（可选）
- **修改** CA3D 生成器，添加默认参数以匹配训练需求（可选）

## Impact
- Affected specs: CA2D 数据生成、CA3D 数据生成、BooleanNetwork 数据生成、训练数据格式
- Affected code:
  - `tests/manifests/gen_ca2d_training.yaml` - 新增 CA2D 训练格式配置示例
  - `tests/manifests/gen_ca3d_training.yaml` - 新增 CA3D 训练格式配置示例
  - `tests/manifests/gen_boolean_training.yaml` - 新增 BooleanNetwork 训练格式配置示例
  - `docs/元胞自动机训练格式配置指南.md` - 新增配置指南文档
  - `src/generators/ca2d.rs` - 可选：添加默认参数
  - `src/generators/ca3d.rs` - 可选：添加默认参数

## ADDED Requirements

### Requirement: CA2D 训练格式配置
系统 SHALL 提供 CA2D 训练格式配置示例，展示如何使用 PatchTokenizer、SequenceStitcher 和 BatchCollector 处理器将 CA2D 数据转换为训练格式。

#### Scenario: CA2D 使用 PatchTokenizer 处理
- **WHEN** 用户配置 CA2D 生成器并添加 PatchTokenizer 处理器
- **THEN** CA2D 的 2D 网格数据被转换为 patch token 序列

#### Scenario: CA2D 使用 SequenceStitcher 串联多帧
- **WHEN** 用户配置 CA2D 生成器并添加 SequenceStitcher 处理器
- **THEN** 多个独立帧被串联成训练序列格式

#### Scenario: CA2D 使用 BatchCollector 批量收集
- **WHEN** 用户配置 CA2D 生成器并添加 BatchCollector 处理器
- **THEN** 流式帧被收集为批量数据格式

### Requirement: CA3D 训练格式配置
系统 SHALL 提供 CA3D 训练格式配置示例，展示如何处理 3D 网格数据以用于训练。

#### Scenario: CA3D 数据格式转换
- **WHEN** 用户配置 CA3D 生成器用于训练
- **THEN** 3D 网格数据被适当地展平或切片以适配训练格式

### Requirement: BooleanNetwork 训练格式配置
系统 SHALL 提供 BooleanNetwork 训练格式配置示例，展示如何处理布尔网络数据以用于训练。

#### Scenario: BooleanNetwork 数据格式转换
- **WHEN** 用户配置 BooleanNetwork 生成器用于训练
- **THEN** 布尔网络状态被转换为适合训练的格式

### Requirement: 训练格式配置文档
系统 SHALL 提供详细的配置文档，说明如何为不同类型的元胞自动机配置训练格式输出。

#### Scenario: 用户查阅配置指南
- **WHEN** 用户需要为元胞自动机配置训练格式输出
- **THEN** 用户可以查阅文档了解如何配置处理器和参数

## MODIFIED Requirements

### Requirement: CA2D 生成器默认参数（可选）
如果需要，可以修改 CA2D 生成器的默认参数以更好地匹配训练需求。

### Requirement: CA3D 生成器默认参数（可选）
如果需要，可以修改 CA3D 生成器的默认参数以更好地匹配训练需求。

## REMOVED Requirements
无移除的需求。
