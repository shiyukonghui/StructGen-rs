# Tasks

- [x] Task 1: 创建 CA2D 训练格式配置示例：展示如何使用处理器将 CA2D 数据转换为训练格式
  - [x] SubTask 1.1: 创建 `tests/manifests/gen_ca2d_training.yaml` 配置文件
  - [x] SubTask 1.2: 配置 PatchTokenizer 处理器参数（patch=3, num_colors=2, rows=12, cols=12）
  - [x] SubTask 1.3: 配置 SequenceStitcher 处理器参数（frames_per_sequence=10）
  - [x] SubTask 1.4: 配置 BatchCollector 处理器参数（batch_size=5, num_frames=10）
  - [x] SubTask 1.5: 测试配置文件是否能正确生成训练格式数据

- [x] Task 2: 创建 CA3D 训练格式配置示例：展示如何处理 3D 网格数据
  - [x] SubTask 2.1: 创建 `tests/manifests/gen_ca3d_training.yaml` 配置文件
  - [x] SubTask 2.2: 分析 3D 网格数据的最佳处理方式（展平或切片）
  - [x] SubTask 2.3: 配置适当的处理器参数
  - [x] SubTask 2.4: 测试配置文件是否能正确生成训练格式数据

- [x] Task 3: 创建 BooleanNetwork 训练格式配置示例：展示如何处理布尔网络数据
  - [x] SubTask 3.1: 创建 `tests/manifests/gen_boolean_training.yaml` 配置文件
  - [x] SubTask 3.2: 分析布尔网络数据的最佳处理方式
  - [x] SubTask 3.3: 配置适当的处理器参数
  - [x] SubTask 3.4: 测试配置文件是否能正确生成训练格式数据

- [x] Task 4: 创建训练格式配置指南文档：详细说明如何配置不同类型的元胞自动机
  - [x] SubTask 4.1: 创建 `docs/元胞自动机训练格式配置指南.md` 文档
  - [x] SubTask 4.2: 说明 CA2D 的配置方法和参数选择
  - [x] SubTask 4.3: 说明 CA3D 的配置方法和参数选择
  - [x] SubTask 4.4: 说明 BooleanNetwork 的配置方法和参数选择
  - [x] SubTask 4.5: 提供完整的配置示例和最佳实践

- [x] Task 5: 验证和测试：确保所有配置都能正确生成训练格式数据
  - [x] SubTask 5.1: 运行 CA2D 训练格式配置测试
  - [x] SubTask 5.2: 运行 CA3D 训练格式配置测试
  - [x] SubTask 5.3: 运行 BooleanNetwork 训练格式配置测试
  - [x] SubTask 5.4: 验证输出数据格式是否符合训练要求

# Task Dependencies
- [Task 1] 可独立进行
- [Task 2] 可独立进行
- [Task 3] 可独立进行
- [Task 4] 依赖 [Task 1], [Task 2], [Task 3]（需要基于实际配置编写文档）
- [Task 5] 依赖 [Task 1], [Task 2], [Task 3], [Task 4]