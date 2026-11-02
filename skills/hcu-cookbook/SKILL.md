---
name: hcu-cookbook
description: 从训练日志解析 tflops / time per iter / tgs，按 dcu-train-cookbook 的格式写入性能表、日志和 all_experiments.csv。
when_to_use: 跑完测试要汇总性能、重新解析已有日志，或把结果写进 dcu-train-cookbook 时使用。
---

# hcu-cookbook（步骤 4）

```bash
K=/public/home/wangxj3/Projects/BenchAgent/skills/hcu-cookbook
R=/public/home/wangxj3/Projects/BenchAgent/runs
```

## 解析

```bash
python3 $K/scripts/parse_metrics.py $R/<run_id> [--warmup 5] [--case ID]
```

- seq_length、gbs、mbs、tp/pp/cp/ep/etp、sp、dispatcher、精度从日志里的参数打印读取（Megatron-LM 的 `name .... value`、bridge 的 YAML），读不到时用测试列表里的值。
- 跳过前 warmup 个 iteration 取中位数，同时保留 best。`tgs = gbs × seq_len / gpus / time_per_iter`。
- 写 `$R/<run_id>/metrics/<id>.json` 和 `results.json`。

## 写入 cookbook

```bash
python3 $K/scripts/export_cookbook.py $R/<run_id>              # 预览，写到 <run_id>/cookbook/
python3 $K/scripts/export_cookbook.py $R/<run_id> --apply      # 写入 cookbook 工作区
```

`--apply` 会：
- 先比较：同一个模型、同一套配置（nodes/gpus/并行参数/mbs/gbs/seq_len/精度等都相同）如果表格里已有非占位的真实数据行，只有本次 tgs（tgs 不可比时退回比 tflops）比已有的好才会写入，并替换掉旧行；不如已有结果就跳过（该用例记入问题表，不复制日志、不改文档，避免覆盖更好的历史结果）。预览模式下也会提示 `[would skip]`，方便 `--apply` 前先看一眼会不会被跳过；
- 真正要写入的：日志按 cookbook 命名（`<模型>-dcu<卡数>-b<gbs>-s<seq>-tp..pp..ep..cp..-<lm|bridge|swift>.log`）复制到 `docs/perf-benchmarks/results/<megatron|ms-swift>/<平台>/logs/`；
- 在模型文档第一张表里：优先填充同名空占位行（保留 modelscope 链接），否则（没有同配置的已有行可比较/替换时）插在该模型已有行之后，再否则追加到表尾；
- `all_experiments.csv` 不受"只留更好结果"影响，每次 `--apply` 的用例都会追加一行，作为完整的实验历史记录（哪怕这次结果没写进表格）。

写完后在 cookbook 仓库跑一下 `git diff` 把改动展示给用户看即可。**不要自己 `git add`/`commit`/`push`**：提交、建分支、推送、提 MR 都留给用户自己审查后处理。

**AI 自己的操作也要写日志**：执行命令用 `benchlog.py exec`（自动写前后两条），改文件、排查等非命令操作用 `begin` + `note`，执行前后各一条；写法见主 skill `skills/SKILL.md`「AI 操作也要写日志」。

```bash
X="python3 /public/home/wangxj3/Projects/BenchAgent/skills/common/benchlog.py exec --site $R/site.json --run-dir $R/<run_id>"
C="cd /public/home/wangxj3/Projects/BenchAgent/frameworks/dcu-train-cookbook &&"
$X --target "cookbook git diff" --plan "汇总改动给用户看，不做提交" -- "$C git diff --stat && git diff"
```

## 排查

- 没解析出 iteration：迭代数不超过 warmup 时调小 `--warmup`；或日志 iteration 行格式变了（改 `common/common.py` 的 `ITER_RE`）。
- 切分与预期不符：以日志为准，修正 hcu-test-plan 的 catalog。
- 没写进 cookbook：`cookbook_doc` 为空或文件不存在（新模型先按同目录模板建文档）；或者是被「只留更好结果」的比较逻辑跳过了（问题表里会写"跳过：已有结果不差于本次"），确认是否真的要用更差的结果覆盖，需要的话手动编辑 cookbook 文档。
- 新模型列名前缀：`export_cookbook.FAMILY`；新框架结果目录：`export_cookbook.RESULT_DIR`。
