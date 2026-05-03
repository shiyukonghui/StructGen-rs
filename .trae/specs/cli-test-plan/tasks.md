# 测试任务列表

## 准备工作

- [x] Task 1: 编译 release 版本程序
  - 在项目根目录执行 `cargo build --release 2>&1`
  - 确认编译无错误
  - 确认 `target/release/structgen-rs.exe` 存在

- [x] Task 2: 创建测试清单目录与所有测试 YAML 文件
  - 在项目根目录创建 `tests/manifests/` 目录
  - 创建 `tests/output/` 目录作为测试输出根目录
  - 编写所有测试用 YAML 清单文件（详见下方清单列表）

## 基础 CLI 测试

- [x] Task 3: 基础命令行参数测试
  - 运行 `structgen-rs.exe --help` 验证所有参数说明正确显示
  - 运行 `structgen-rs.exe --version` 验证版本号为 0.1.0

## 清单解析与校验测试

- [x] Task 4: 合法清单测试
  - 运行合法最小清单（CA + count=10 + Text 格式）
  - 验证退出码为 0
  - 验证输出目录有生成文件
  - 验证 metadata.json 包含必要字段

- [x] Task 5: 错误清单测试
  - 分别测试以下非法场景，验证退出码为 2 且 stderr 包含对应错误信息：
    - 清单文件不存在
    - YAML 格式错误
    - 空任务列表
    - 未注册的生成器名称
    - 未注册的处理器名称
    - 样本数量为 0
    - 重复任务名称
    - output_dir 为空

## 生成器覆盖测试

- [x] Task 6: 十种生成器逐一测试
  - 对每种生成器创建专用 YAML 清单
  - 运行并验证退出码为 0
  - 验证输出文件非空
  - 验证 metadata.json 包含该任务记录
  - 生成器列表：
    - CellularAutomaton (ca / cellular_automaton)
    - LorenzSystem (lorenz_system / lorenz)
    - LogisticMap (logistic_map / logistic)
    - NBodySim (nbody / nbody_sim)
    - LSystem (lsystem)
    - AlgorithmVM (algorithm_vm / vm)
    - BooleanNetwork (boolean_network)
    - IFS (ifs)
    - FormalGrammar (formal_grammar)

## 输出格式覆盖测试

- [x] Task 7: 三种输出格式测试
  - Parquet 格式：验证产出 .parquet 文件
  - Text 格式：验证产出文本文件
  - Binary 格式：验证产出 .bin 文件

## 处理器管道测试

- [x] Task 8: 处理器管道测试
  - 空管道（无处理器）
  - 单处理器：normalizer
  - 两处理器组合：normalizer + token_mapper
  - 全管道链：normalizer + dedup + diff_encoder + token_mapper
  - 使用 clip_stitcher 处理器

## 多任务与综合测试

- [x] Task 9: 多任务混合清单测试
  - 创建包含 3+ 种不同生成器的清单
  - 运行并验证退出码为 0
  - 验证每种生成器产出独立的输出文件
  - 验证 metadata.json 包含所有任务

## CLI 参数覆盖测试

- [x] Task 10: 命令行参数覆盖测试
  - --output-dir 覆盖清单中的 output_dir
  - --format 覆盖清单中的 default_format
  - --num-threads 指定线程数
  - --log-level 验证不同日志级别

## 确定性复现测试

- [x] Task 11: 确定性复现测试
  - 用相同 YAML 清单运行两次
  - 计算两次输出文件的 SHA256 校验和
  - 验证校验和一致

## 进度条与流式输出测试

- [x] Task 12: 进度条与日志测试
  - 带 --no-progress 运行（CI 模式）
  - 不带 --no-progress 运行（TTY 模式，验证进度条输出到 stderr）
  - --stream-write=false 关闭流式输出
  - 验证日志文件 structgen_run.log 在输出目录生成

---

# 测试用 YAML 清单文件列表

以下清单文件需创建在 `tests/manifests/` 目录下：

| 文件名 | 用途 | 关键配置 |
|--------|------|----------|
| `minimal_ca.yaml` | 最小合法清单 | CA + count=10 + Text |
| `bad_format.yaml` | YAML 格式错误 | 非法 YAML 语法 |
| `empty_tasks.yaml` | 空任务列表 | tasks: [] |
| `unknown_generator.yaml` | 未注册生成器 | nonexistent_gen |
| `unknown_processor.yaml` | 未注册处理器 | nonexistent_proc |
| `zero_count.yaml` | count=0 | count: 0 |
| `duplicate_names.yaml` | 重复任务名 | 两个同名任务 |
| `gen_ca.yaml` | CA 生成器 | rule=30 |
| `gen_lorenz.yaml` | Lorenz 生成器 | sigma/rho/beta |
| `gen_logistic.yaml` | Logistic 生成器 | r 参数 |
| `gen_nbody.yaml` | NBody 生成器 | num_bodies 参数 |
| `gen_lsystem.yaml` | LSystem 生成器 | 默认参数 |
| `gen_vm.yaml` | AlgorithmVM 生成器 | 默认参数 |
| `gen_boolean.yaml` | BooleanNetwork 生成器 | num_nodes 参数 |
| `gen_ifs.yaml` | IFS 生成器 | num_transforms 参数 |
| `gen_grammar.yaml` | FormalGrammar 生成器 | max_derivations 参数 |
| `format_parquet.yaml` | Parquet 输出 | output_format: Parquet |
| `format_text.yaml` | Text 输出 | output_format: Text |
| `format_binary.yaml` | Binary 输出 | output_format: Binary |
| `pipeline_full.yaml` | 全管道链 | normalizer+dedup+diff+token |
| `pipeline_clip.yaml` | clip_stitcher 处理器 | clip_stitcher |
| `multi_task.yaml` | 多任务混合 | 3+ 种生成器 |
| `override_format.yaml` | 格式覆盖测试 | 清单 Text + 命令行 Binary |

# 任务依赖
- Task 4-12 均依赖 Task 1（编译）
- Task 3-12 均依赖 Task 2（测试文件准备）
- Task 6-12 均依赖 Task 4（验证基本流程正常）
- Task 9 可与 Task 6-8 并行
- Task 11 依赖 Task 4 通过
