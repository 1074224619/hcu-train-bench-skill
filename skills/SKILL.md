---
name: hcu-bench
description: HCU 训练性能测试总流程（主 skill）。串起节点准备 → 测试计划 → 跑模型 → 解析并写入 dcu-train-cookbook 四个子 skill，管理共享配置 site.json 和中文运行日志。
when_to_use: 用户要从头跑一轮框架/模型性能测试、问整体流程或进度、不确定该用哪个子 skill 时使用；具体某一步直接看对应子目录的 SKILL.md。
---

# hcu-bench（主 skill）

测试 Megatron-LM-das / Megatron-bridge-das / ms-swift 在 HCU 上的训练性能，结果写入 dcu-train-cookbook。本文件只讲流程、分工和共享约定；每一步怎么做看子目录。

## 子 skill

| 步骤 | 子目录 | 做什么 | 产出（在 runs/ 下） |
|---|---|---|---|
| 1 节点准备 | [hcu-node-prep](hcu-node-prep/SKILL.md) | idle 筛节点、建容器、platform/resource/RCCL 检查 | `hostfile` `idle_hosts` `node_status.md` `healthy_hosts` |
| 2 测试计划 | [hcu-test-plan](hcu-test-plan/SKILL.md) | 扫描框架 example 生成用例目录，按可用节点排测试列表 | `<run_id>/matrix.json` `matrix.md` |
| 3 跑模型 | [hcu-run-model](hcu-run-model/SKILL.md) | 单个模型调试；按测试列表批量执行 | `<run_id>/logs/` `status/` |
| 4 结果入库 | [hcu-cookbook](hcu-cookbook/SKILL.md) | 解析 tflops/time/tgs，写入 cookbook | `<run_id>/results.json`，cookbook 改动 |

步骤之间只通过 `runs/` 下的文件交接，每个子 skill 可以单独跑、单独调试。

## 目录

```
BenchAgent/
├── frameworks/          三个框架 + cluster-manager-das（hcu-envcheck）+ dcu-train-cookbook
├── runs/                运行数据（site.json、节点列表、bench_log.md、每次测试的 <run_id>/、hcu-run-model 单模型调试输出）
└── skills/
    ├── SKILL.md         本文件
    ├── common/          共享代码和配置，不是 skill：common.py、benchlog.py、site.example.json
    ├── hcu-node-prep/  hcu-test-plan/  hcu-run-model/  hcu-cookbook/
```

## 共享约定

- **配置**：`runs/site.json`，从 `common/site.example.json` 复制后填写（镜像、容器名、ssh 端口、挂载、平台、每节点卡数、集群环境脚本 `env_script`、框架和 cookbook 路径）。所有子 skill 都读它。现阶段只跑哪些框架（如先只跑 megatron-lm，bridge/swift 后面再开）用 `frameworks_enabled` 字段限制，见 hcu-test-plan/SKILL.md。
- **路径**：frameworks、runs、数据集、模型在登录节点、计算节点、容器内绝对路径相同；容器挂载只来自 `container.mounts`。
- **节点列表**：`hostfile.all`（用户给的全部）→ `hostfile`（可 ssh，建容器用）→ `idle_hosts`（空闲，跑测试用）→ `healthy_hosts`（空闲且检查通过）。
- **日志**：脚本的每一步操作在 `runs/bench_log.md` 写两条中文记录：执行前一条「开始」（时间、步骤、命令、将执行的操作），执行后一条结果（状态、结果、问题与处理方案、耗时、明细路径）；批量跑模型时每个用例也各有一对。测试相关的同时写 `runs/<run_id>/run_log.md`。脚本异常退出或被 Ctrl-C 中断时会补写一条「异常/中断」结果，所以每条「开始」后面都有对应的结果；只有「开始」没有结果，说明进程被强杀（如 kill -9）。实现在 `common/benchlog.py`（Python 脚本用 `benchlog.Step`），常见错误识别表是其中的 `DIAGNOSES`。AI 自己的操作见下一节。**所有日志/运行输出都放 `runs/` 下，不要放进 `skills/`**（`skills/` 只放代码和固定配置，和运行产出混在一起会很乱）；`hcu-run-model` 单模型调试默认也写到 `runs/hcu-run-model/`，只有没给 `--site` 时才会退回 `skills/hcu-run-model/runs/`（没有别的地方可以参照）。

## AI 操作也要写日志（必须遵守）

脚本会自己写日志；**AI 自己做的每一步操作也必须写**，同样是执行前一条「开始」、执行后一条结果，写进 `runs/bench_log.md`（和某次测试相关时加 `--run-dir runs/<run_id>`，同时写 `run_log.md`）。

要记的操作：

| 类型 | 例子 |
|---|---|
| 在登录节点 / 计算节点 / 容器里执行命令 | `ssh <节点> ls <路径>` 核对数据集/模型、`docker ps`、`ps -fp <PID>` 查占用、`ssh-keygen -R`、`kill` 残留进程 |
| 改文件 | `site.json`、`catalog.json`、`hostfile*`、经用户同意后改框架脚本、cookbook 文档 |
| git 操作 | cookbook `git diff` 给用户看改动（不自动建分支/提交/推送，交给用户自己处理） |
| 排查 | 读了哪些日志 / 脚本，定位到什么原因 |
| 按用户决定做的处理 | 排除某节点、跳过某用例、换路径；在 `--plan` 里写明是用户的决定 |

不用另记：调用本身会写日志的脚本（`prepare_nodes.sh`、各 `scripts/*.py`）不要重复记；向用户汇报时读 `bench_log.md` 不用记。

怎么记（命令假定在项目根目录下执行，`K`/`S`/`R` 换成绝对路径也一样）：

```bash
L="python3 skills/common/benchlog.py"
S=runs/site.json

# 执行命令：用 exec 包一层，自动写前后两条，按退出码记成功/失败并自动诊断；命令的退出码原样返回
$L exec --site $S --target "核对 qwen3 数据路径（bw67）" --plan "run_qwen.sh 里 DATA_PATH 写死，确认 .bin/.idx 存在" \
    -- 'ssh bw67 ls /path/to/datasets/qwen3_text_document.{bin,idx}'

# 不是命令的操作（用编辑工具改文件、读日志排查、记录用户决定）：手动写 begin、note 两条
T0=$(date +%s)
$L begin --site $S --target "改 site.json 的 container.mounts" --plan "集群禁止挂载 /data/，改成允许的子目录"
#   ……做操作……
$L note --site $S --target "改 site.json 的 container.mounts" --since $T0 --status 成功 \
    --result "改为 /data/:/data/" [--issue '对象|问题|处理方案'] --detail $S
```

要求：

- **先写「开始」再动手**，做完马上写结果；失败、放弃、被用户叫停也要写结果（状态写 失败/未执行/中断），不能只有「开始」。
- `--target` 写清对象（哪个节点 / 文件 / 用例），`--plan` 写为什么做、准备怎么做，`--result` 写结论（看到了什么、改成了什么），有问题用 `--issue` 写处理方案。
- 同一目的的几个只读检查可以合成一条（例如一次 ssh 里 ls 多个路径）。
- 每一步结束向用户报告前，确认 `bench_log.md` 里每条「开始」都有对应的结果。

## 执行规则

- 按 1 → 4 顺序；每一步结束读 `bench_log.md` 最新一条，向用户报告状态和问题表。
- AI 自己的每一步操作按上一节写日志：执行前写「开始」，执行后写结果。
- 出问题先报告再处理；需要用户决定的先停下问：建/删/重建容器（`--yes`）、释放别人占用的卡、修改框架仓库脚本、确认测试列表（matrix.md）。cookbook 导出后只 `git diff` 给用户看，不自动提交或推送。
- 占用状态会变，开始跑模型前重跑一次 idle。
- 跑模型不顺时，先用 hcu-run-model 单个调通，再回到批量。
