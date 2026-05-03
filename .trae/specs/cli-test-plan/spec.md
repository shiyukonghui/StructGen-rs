# CLI 测试规划 Spec

## Why
StructGen-rs 项目已完成开发，需要对程序进行系统化的命令行运行测试，验证编译正确性、CLI 参数解析、全流程生成功能、输出文件质量和错误处理能力。

## What Changes
- 编写 CLI 测试规划文档，定义所有测试场景和预期结果
- 编译 release 版本二进制程序 `structgen-rs`
- 准备测试用 YAML 清单文件（覆盖所有生成器、输出格式、处理器组合）
- 通过命令行逐项运行测试，验证退出码、输出文件内容、元数据完整性
- 记录测试结果与问题

## Impact
- Affected specs: 无（新规划）
- Affected code: 无代码变更，仅编译与测试

## ADDED Requirements

### Requirement: 编译验证
系统 SHALL 能够成功编译 release 版本二进制程序。

#### Scenario: cargo build 成功
- **WHEN** 在项目根目录执行 `cargo build --release`
- **THEN** 编译成功，无错误，产出 `target/release/structgen-rs.exe`

### Requirement: 基础 CLI 功能
系统 SHALL 正确响应 `--help` 和 `--version` 参数。

#### Scenario: 显示帮助信息
- **WHEN** 执行 `structgen-rs --help`
- **THEN** 输出包含所有命令行参数说明（manifest、output-dir、format、num-threads、log-level、stream-write、no-progress）

#### Scenario: 显示版本号
- **WHEN** 执行 `structgen-rs --version`
- **THEN** 输出版本号 "StructGen-rs 0.1.0"

### Requirement: 清单解析与校验
系统 SHALL 正确解析合法 YAML 清单并运行，对非法清单返回错误退出码 2。

#### Scenario: 合法最小清单运行成功
- **WHEN** 提供一个包含单个 CA 任务、count=10 的最小合法 YAML
- **THEN** 退出码为 0，输出目录包含生成文件，包含 metadata.json

#### Scenario: 清单文件不存在
- **WHEN** 指定不存在的 manifest 文件路径
- **THEN** 退出码为 2（FULL_FAILURE），stderr 包含 "not found"

#### Scenario: YAML 格式错误
- **WHEN** 提供格式错误的 YAML 内容
- **THEN** 退出码为 2，stderr 包含 "YAML parse error"

#### Scenario: 空任务列表
- **WHEN** 提供 tasks: [] 的空清单
- **THEN** 退出码为 2，stderr 包含错误信息

#### Scenario: 未注册的生成器
- **WHEN** 引用不存在的生成器名称
- **THEN** 退出码为 2，stderr 包含 "未注册的生成器"

#### Scenario: 未注册的处理器
- **WHEN** 引用不存在的处理器名称
- **THEN** 退出码为 2，stderr 包含 "未注册的处理器"

#### Scenario: 样本数量为 0
- **WHEN** 任务 count 为 0
- **THEN** 退出码为 2，stderr 包含 "样本数量不能为 0"

### Requirement: 十种生成器验证
系统 SHALL 支持所有 10 种已注册生成器的正常产出。

#### Scenario: CellularAutomaton (ca) 生成成功
- **WHEN** 使用 generator: "ca" 或 "cellular_automaton"，extends rule=30
- **THEN** 退出码为 0，产出有效数据文件

#### Scenario: LorenzSystem (lorenz) 生成成功
- **WHEN** 使用 generator: "lorenz_system" 或 "lorenz"，扩展参数 sigma/rho/beta
- **THEN** 退出码为 0，产出有效数据文件

#### Scenario: LogisticMap (logistic) 生成成功
- **WHEN** 使用 generator: "logistic_map" 或 "logistic"
- **THEN** 退出码为 0，产出有效数据文件

#### Scenario: NBodySim (nbody) 生成成功
- **WHEN** 使用 generator: "n_body" 或 "nbody"
- **THEN** 退出码为 0，产出有效数据文件

#### Scenario: LSystem (lsystem) 生成成功
- **WHEN** 使用 generator: "lsystem" 或 "lindenmayer_system"
- **THEN** 退出码为 0，产出有效数据文件

#### Scenario: AlgorithmVM (vm) 生成成功
- **WHEN** 使用 generator: "algorithm_vm" 或 "vm"
- **THEN** 退出码为 0，产出有效数据文件

#### Scenario: BooleanNetwork (boolean_network) 生成成功
- **WHEN** 使用 generator: "boolean_network"
- **THEN** 退出码为 0，产出有效数据文件

#### Scenario: IFS (ifs) 生成成功
- **WHEN** 使用 generator: "iterated_function_system" 或 "ifs"
- **THEN** 退出码为 0，产出有效数据文件

#### Scenario: FormalGrammar (formal_grammar) 生成成功
- **WHEN** 使用 generator: "formal_grammar"
- **THEN** 退出码为 0，产出有效数据文件

### Requirement: 三种输出格式验证
系统 SHALL 正确产出 Parquet、Text、Binary 三种格式的输出文件。

#### Scenario: Parquet 格式输出
- **WHEN** 指定 output_format: "Parquet" 或 --format parquet
- **THEN** 退出码为 0，输出文件扩展名为 .parquet，文件可被 parquet 工具读取

#### Scenario: Text 格式输出
- **WHEN** 指定 output_format: "Text" 或 --format text
- **THEN** 退出码为 0，输出文件扩展名为 .txt（或相应文本格式）

#### Scenario: Binary 格式输出
- **WHEN** 指定 output_format: "Binary" 或 --format binary
- **THEN** 退出码为 0，输出文件扩展名为 .bin

### Requirement: 处理器管道验证
系统 SHALL 支持所有 6 种处理器的链式组合。

#### Scenario: 单处理器 (normalizer) 正常执行
- **WHEN** 指定 pipeline: ["normalizer"]
- **THEN** 退出码为 0，数据经过标准化处理

#### Scenario: 全管道链执行
- **WHEN** 指定 pipeline: ["normalizer", "dedup_filter", "diff_encoder", "token_mapper"]
- **THEN** 退出码为 0，数据依次经过所有处理器

#### Scenario: 空管道（无处理）正常执行
- **WHEN** 指定 pipeline: [] 或不指定 pipeline
- **THEN** 退出码为 0，原始生成数据直接输出

### Requirement: 多任务清单
系统 SHALL 支持包含多个任务的清单并行执行。

#### Scenario: 多生成器混合执行
- **WHEN** 清单包含 3+ 种不同生成器的任务
- **THEN** 退出码为 0，每个任务产出独立文件，metadata.json 包含所有任务信息

### Requirement: 命令行参数覆盖
系统 SHALL 允许命令行参数覆盖清单中的对应配置。

#### Scenario: --output-dir 覆盖清单输出目录
- **WHEN** 命令行指定 --output-dir 不同于清单中的 output_dir
- **THEN** 输出产在命令行指定的目录中，清单原目录不产生文件

#### Scenario: --format 覆盖清单默认格式
- **WHEN** 命令行 --format binary 而清单默认格式为 Text
- **THEN** 输出格式为 Binary

#### Scenario: --num-threads 覆盖线程数
- **WHEN** 命令行 --num-threads 4
- **THEN** 使用 4 线程并行

#### Scenario: --log-level 控制日志输出
- **WHEN** 命令行 --log-level debug
- **THEN** 调试级别日志被输出

### Requirement: 确定性复现
系统 SHALL 保证相同种子、相同配置产生相同输出。

#### Scenario: 两次运行相同配置产生相同文件
- **WHEN** 用相同 YAML 清单运行两次
- **THEN** 两次运行输出文件的 SHA256 校验和一致

### Requirement: 退出码语义
系统 SHALL 根据运行结果返回正确的退出码。

#### Scenario: 全成功返回 0
- **THEN** 退出码为 SUCCESS (0)

#### Scenario: 部分失败返回 1
- **THEN** 退出码为 PARTIAL_FAILURE (1)

#### Scenario: 全失败返回 2
- **THEN** 退出码为 FULL_FAILURE (2)

### Requirement: metadata.json 完整性
系统 SHALL 在每次成功运行后产出格式正确的 metadata.json。

#### Scenario: metadata.json 包含必要字段
- **WHEN** 运行成功
- **THEN** metadata.json 包含 software_version、start_time、end_time、global_config、summary、tasks 字段
