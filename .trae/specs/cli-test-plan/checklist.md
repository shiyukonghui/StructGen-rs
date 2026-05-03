# CLI 测试验证清单

## 编译验证
- [x] `cargo build --release` 无编译错误
- [x] `target/release/structgen-rs.exe` 文件存在且可执行

## 基础 CLI 验证
- [x] `structgen-rs --help` 显示完整参数说明（manifest, output-dir, format, num-threads, log-level, stream-write, no-progress）
- [x] `structgen-rs --version` 显示 "StructGen-rs 0.1.0"

## 清单解析与校验
- [x] 合法最小清单（CA + count=10）运行成功，退出码为 0
- [x] 输出目录包含生成的数据文件（非空）
- [x] 输出目录包含 metadata.json
- [x] 清单文件不存在时退出码为 2，stderr 包含 "not found"
- [x] YAML 格式错误时退出码为 2，stderr 包含 "YAML parse error"
- [x] 空任务列表时退出码为 2，stderr 包含错误信息
- [x] 未注册生成器时退出码为 2，stderr 包含 "未注册的生成器"
- [x] 未注册处理器时退出码为 2，stderr 包含 "未注册的处理器"
- [x] count=0 时退出码为 2，stderr 包含 "样本数量不能为 0"
- [x] 重复任务名时退出码为 2，stderr 包含 "任务名称重复"
- [x] output_dir 为空时退出码为 2，stderr 包含错误信息

## 生成器覆盖验证
- [x] CellularAutomaton (ca) 生成成功，退出码为 0，输出文件非空
- [x] LorenzSystem (lorenz_system) 生成成功，退出码为 0，输出文件非空
- [x] LogisticMap (logistic_map) 生成成功，退出码为 0，输出文件非空
- [x] NBodySim (nbody) 生成成功，退出码为 0，输出文件非空
- [x] LSystem (lsystem) 生成成功，退出码为 0，输出文件非空
- [x] AlgorithmVM (algorithm_vm) 生成成功，退出码为 0，输出文件非空
- [x] BooleanNetwork (boolean_network) 生成成功，退出码为 0，输出文件非空
- [x] IFS (ifs) 生成成功，退出码为 0，输出文件非空
- [x] FormalGrammar (formal_grammar) 生成成功，退出码为 0，输出文件非空

## 输出格式验证
- [x] Parquet 格式输出成功，产出 .parquet 文件
- [x] Text 格式输出成功，产出文本文件
- [x] Binary 格式输出成功，产出 .bin 文件

## 处理器管道验证
- [x] 空管道（无处理器）执行成功，退出码为 0
- [x] 单处理器 normalizer 执行成功，退出码为 0
- [x] 两处理器 normalizer + token_mapper 执行成功，退出码为 0
- [x] 全管道链 normalizer + dedup + diff_encoder + token_mapper 执行成功
- [x] clip_stitcher 处理器执行成功，退出码为 0

## 多任务混合验证
- [x] 包含 3 种不同生成器的清单运行成功，退出码为 0
- [x] 每种生成器产出独立的输出文件
- [x] metadata.json 包含所有任务的信息

## CLI 参数覆盖验证
- [x] --output-dir 覆盖生效，文件产出在命令行指定目录
- [x] --format 覆盖全局默认格式生效
- [x] --num-threads 参数正确传递
- [x] --log-level debug 输出调试日志

## 确定性复现验证
- [x] 相同配置运行两次，输出文件内容一致 (DETERMINISTIC: PASS)

## 进度条与日志验证
- [x] --no-progress 模式下无进度条输出
- [x] 非 --no-progress 模式下进度条正常运行
- [x] --stream-write=false 执行成功

## metadata.json 完整性验证
- [x] 包含 software_version 字段
- [x] 包含 start_time 字段
- [x] 包含 end_time 字段
- [x] 包含 global_config 字段
- [x] 包含 summary 字段（含 total_tasks, total_samples 等）
- [x] 包含 tasks 数组，其中每项含 name, generator, count 等

## 退出码语义验证
- [x] 全成功场景退出码为 0
- [x] 全失败场景退出码为 2
