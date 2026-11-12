---
name: hcu-run-model
description: 在 HCU 节点的容器里跑单个模型训练（Megatron-LM-das / Megatron-bridge-das / ms-swift），跑到指定步数自动停，给出中文结果和报错原因。用于单独调试某个框架+模型能不能跑通。
when_to_use: 用户要单独跑/调试某个模型、某个框架的 example 脚本，确认数据集/模型/tokenizer 路径、端口、容器是否正确，或排查训练启动失败。
---

# hcu-run-model

只做一件事：在已建好的容器里跑一个模型，跑到第 N 步停，留下日志和结论。节点筛查和建容器见 hcu-node-prep，测试列表见 hcu-test-plan，写 cookbook 见 hcu-cookbook，总流程见主 skill `skills/SKILL.md`。

```
skills/hcu-run-model/
├── SKILL.md
└── scripts/
    ├── run_model.py          启动 + 监控 + 到步数自动停 + 写结果（只依赖标准库）
    └── tune_plan.py          按运行记录/手工输入给显存调优建议（只读，只依赖标准库）
```

日志/输出不放在 `skills/` 下面（避免和 skill 代码混在一起）：给了 `--site` 时默认写到 `<runs_dir>/hcu-run-model/<时间>_<脚本名>/`（`hostfile`、`cmd.txt`、`train.log`、`status.json`、`result.md`、`run_log.md`）；没给 `--site` 就没有别的地方可以参照，退回 `skills/hcu-run-model/runs/<时间>_<脚本名>/`。想自己指定目录用 `--out DIR`。批量跑（`run_cases.py`）的输出一直都在 `<run_dir>/`（hcu-test-plan 生成，同样在 `runs/` 下），不受这个影响。

## 前提

- 节点 ssh 免密可达，`hcu-bench` 容器已在节点上运行（`docker ps` 能看到）。
- 节点卡空闲。
- 框架目录、数据集、模型在容器里路径相同（容器挂载了 site.json `container.mounts` 里配的数据/模型目录）。

## 调试流程（每个模型按这个顺序，出问题就停下来报告）

**AI 自己的操作也要写日志**：执行命令用 `benchlog.py exec`（自动写前后两条），改文件、排查、按用户决定处理等非命令操作用 `begin` + `note`，执行前后各一条；写法见主 skill `skills/SKILL.md`「AI 操作也要写日志」。下面每一步都要记，第 1、2 步是 AI 手动做的，最容易漏：

```bash
L="python3 skills/common/benchlog.py"; S=runs/site.json
T0=$(date +%s); $L begin --site $S --target "读 qwen3 run_qwen.sh / train_qwen3_8B.sh" --plan "找数据集、tokenizer、端口、env 脚本"
$L note --site $S --target "读 qwen3 run_qwen.sh / train_qwen3_8B.sh" --since $T0 --status 成功 \
    --result "DATA_PATH=… 可用 --env 覆盖（见各框架一节）" --result "mpirun 端口 11451，容器 sshd 25901，不一致" --issue 'run_qwen.sh|端口不一致|多节点前和用户确认怎么改'
$L exec --site $S --target "核对 qwen3 数据/tokenizer（bw67）" --plan "确认第 1 步列出的路径在节点上存在" -- 'ssh bw67 ls <前缀>.bin <前缀>.idx <tokenizer目录>'
```

1. **读脚本**：按下面「各框架」一节找到 launcher 和训练脚本，列出它实际用的数据集、模型/tokenizer 路径、端口、env 脚本。
2. **核对路径**：上面每个路径在节点上 `ls` 一下（`ssh <节点> ls <路径>`；Megatron-LM 数据是前缀，检查 `<前缀>.bin`/`.idx`）。缺的、写死在别人目录的、`/path/to/` 占位的，先报告给用户，**不要自己改框架仓库**，等用户决定。
3. **预演**：`run_model.py ... --dry-run` 看命令；bridge 先用 `MODE=dryrun` 只解析配置。
4. **单节点跑**：先 1 个节点、`--max-iters 10`，看能否出现 iteration 行。
5. **再多节点**：单节点通了再加节点，重点看端口和 ssh。
6. **汇报**：读 `runs/<目录>/result.md`，报告状态、性能、问题和可能原因。
7. **调优**：OOM 或显存富余时跑 `tune_plan.py <运行目录>`，按建议一次改一项，再回到第 3 步验证。
8. **改动**：用户同意后改脚本、换路径、清理残留进程，每个改动前后各记一条（`--plan` 写用户的决定）。

## 命令

以下命令假定在项目根目录下执行（`R`/`F` 换成绝对路径也一样）：

```bash
R=skills/hcu-run-model/scripts/run_model.py
F=frameworks

python3 $R --hosts bw67 --launcher <launcher 绝对路径> --arg ... --env KEY=VAL ... \
    [--max-iters 20] [--warmup 5] [--timeout-min 60] [--container hcu-bench] [--dry-run] \
    --site runs/site.json
```

- 在 `--hosts` 第一个节点上 `ssh + docker exec <容器>` 启动 launcher，launcher 自己的 mpirun 负责其他节点。
- `--arg` 可写 `{hostfile}`、`{nodes}`，自动替换成本次生成的 hostfile 和节点数。
- 出现第 `--max-iters` 个 iteration 后结束整个进程组；超时记 timeout；Ctrl-C 记 stopped。
- 输出实时打印，同时写 `train.log`；结束后写 `result.md`。
- 跑的同时后台每 5 秒对 `--hosts` 里每个节点 `docker exec <容器> hy-smi --showmemuse --json` 一次，取各卡的最大值，训练结束后记下每个节点的显存占用峰值（百分比）：写进 `status.json` 的 `vram_peak_percent`，`result.md` 里也有一行「显存占用峰值（hy-smi --showmemuse）」。某个节点偶尔连不上或输出解析不出来，这一轮就不计入，不影响跑模型本身。
- 执行前、执行后各写一条中文日志到输出目录的 `run_log.md`；给了 `--site` 时同时写 `runs/bench_log.md`（调试时都带上）。

## 显存调试（OOM / 显存富余）

先看证据，再改参数，一次只改一项：

- `status.json` 的 `vram_peak_percent`：执行期间每个节点 `hy-smi --showmemuse` 的最大值。峰值低说明切分过细/重计算过多；接近 100% 说明确实吃满。
- `train.log` 里 OOM 的堆栈：决定该往哪调。`result.md` 的「可能原因」会自动区分；也可以让工具按运行记录给建议（只读，不改配置）：

```bash
python3 skills/hcu-run-model/scripts/tune_plan.py runs/hcu-run-model/<运行目录>
python3 skills/hcu-run-model/scripts/tune_plan.py --state failed --oom-at optimizer \
    --vram-peak 97 --framework megatron-bridge
```

工具输出结论、按优先级排列的建议、bridge/Megatron-LM 参数入口和并行约束。

| OOM 位置（堆栈关键字） | 说明 | 优先动作 |
|---|---|---|
| optimizer.step / get_grad_norm / clip_grads | 参数+梯度+优化器状态放不下；recompute 只省激活，通常无效 | 优化器 offload + 分布式优化器；加大 pp/tp/ep 或减少层数 |
| ParamAndGradBuffer / DDP | 梯度缓冲随参数量增长 | 加大切分、分布式优化器；recompute 帮助有限 |
| TransformerLayer / attention / mlp / backward | 激活值峰值 | selective→full recompute、activation offload；再减 mbs；最后加切分 |
| 无堆栈，只有 `HIP failure: 'out of memory'` | 位置不明 | 先按参数类处理（切分+offload），再用 `--oom-at` 细化 |

显存峰值对应的调整方向（多节点取最高值）：

| 峰值 | 判断 | 动作 |
|---|---|---|
| <40% | 富余明显 | 关 recompute/activation offload；切分降一档（优先 pp，其次 ep，最后 tp）；GBS 不变时调大 mbs |
| 40–55% | 有富余 | 先关 offload 或降一档切分，目标 55–75% |
| 55–75% | 合适 | 保持；要提速优先调大 mbs |
| 75–90% | 余量偏小 | 不要减切分；加 recompute/activation offload 或加一档切分 |
| ≥90% | 接近上限 | 先加 recompute/offload 或切分，再跑长一点确认稳定 |

参数入口（以 2026-11 读到的脚本为准；改框架脚本/recipe 前先让用户确认）：

- **bridge**：launcher 环境变量 `TP/PP/EP/ETP/CP`（minimax 支持）；`qwen35_vl` 还支持 `MICRO_BATCH_SIZE/GLOBAL_BATCH_SIZE/SEQ_LENGTH`，并把 hostfile/nodes 之后的 `KEY=VALUE` 透传给 `run_recipe.py`；其他 launcher 的批次参数多数要改 recipe。
- **bridge recipe 字段**：`cfg.model.recompute_granularity`、`cfg.model.recompute_modules`（如 `["moe_act"]`）、`cfg.model.fine_grained_activation_offloading` + `cfg.model.offload_modules`、`cfg.optimizer.optimizer_cpu_offload`、`cfg.ddp.use_distributed_optimizer`。
- **Megatron-LM**：`TP/PP/CP/MBS/GBS/SEQ_LEN` 写死在 `train_*.sh`；recompute/offload 用 MCore 命令行参数，测试脚本里已有：`--recompute-granularity selective|full`、`--recompute-method block|uniform --recompute-num-layers N`、`--recompute-modules moe_act`、`--optimizer-cpu-offload --optimizer-offload-fraction 0.5`、`--use-distributed-optimizer`。

并行约束：world = TP×PP×CP×DP；EP 整除 DP；TP 整除 attention head 数；PP 不超过层数；GBS 能被 MBS×DP 整除。

实测参考：20 层 MiniMax-M3 开 full recompute 仍在 `optimizer.step → get_grad_norm` OOM（最后是 2 MB 的 NCCL calloc 失败，说明瓶颈是参数/梯度/优化器状态，不是激活值）；改成 10 层、PP2/EP8、关闭全部 recompute 后 20 步成功，峰值 43%/63%，中位数 18.1 s/iter、92.6 TFLOP/s/GPU。

## 各框架

三个框架接收路径的方式不同，调试时以脚本实际内容为准，下面是 2026-10-27 读脚本得到的情况。

### Megatron-LM-das

| 项 | 内容 |
|---|---|
| launcher | `examples/<模型族>/run_*.sh <hostfile> <节点数>`；训练脚本 `train_*.sh` 由 launcher 调用 |
| 选模型 | 大部分 launcher 用 `TRAIN_SCRIPT=${TRAIN_SCRIPT:-默认}`，可 `--env TRAIN_SCRIPT=train_qwen3_8B.sh` |
| 数据集 | launcher 里 `DATA_PATH="${DATA_PATH:-默认}"`，和 `TRAIN_SCRIPT` 一样可用 `--env DATA_PATH=...` 覆盖；只有 qwen3 带默认值（oscar-en-10k qwen3 前缀），其他族默认是空字符串——这些用 hcu-test-plan 时由 `site.json` 的 `models.<model>.data_path` 自动注入（见 hcu-test-plan/SKILL.md「数据集/模型路径」），单独调试本 skill 时要自己 `--env DATA_PATH=...` 传 |
| tokenizer | launcher 里 `TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-默认}"`，同上可 `--env` 覆盖或走 `site.json` 的 `models.<model>.tokenizer_path`；qwen3 默认都用 `qwen3-30b-a3b` |
| DTK_ENV / NCCL_ENV | 同样是 `${VAR:-默认}`，默认值各 launcher 不一定一致。**除 qwen3/run_qwen.sh 外，其余所有 launcher 的 DTK_ENV 默认值都是空字符串**（`${DTK_ENV:-}`），`source ${DTK_ENV}` 会因为缺文件名直接报 `source: filename argument required` 退出，必须显式传 `--env DTK_ENV=/opt/dtk/env.sh`。**NCCL_ENV 必须显式覆盖成 site.json 的 `env_script`（`skills/common/env.sh`），不能用框架自带 `requirements/env.sh` 的默认值**：后者在本集群下会让 PyTorch 的 NCCL 初始化（`torch.distributed.barrier()`）报 `ncclInternalError: ... Attribute arch of node cpu not found`，即便单机 rccl-tests 基线刚测过是 PASS 的（根因和排查过程见 QA.md 3.2）。单独调试本 skill 时要自己 `--env DTK_ENV=/opt/dtk/env.sh --env NCCL_ENV=skills/common/env.sh` 传；走 hcu-test-plan/`build_matrix.py` 生成测试列表时两者都已经自动注入，不用手传 |
| 端口 | mpirun 跨机用 `plm_rsh_args '-p ${SSH_PORT}'` ssh 到其他节点拉起进程，`SSH_PORT` 是 `${SSH_PORT:-}`（留空走 mpirun 默认 22）；**必须覆盖成 site.json 的 `container.ssh_port`**，否则多机会卡在连接超时/被拒（`root@<node>: Permission denied`）。单独调试本 skill 要自己 `--env SSH_PORT=<container.ssh_port>` 传；走 `build_matrix.py` 时已自动注入，不用手传（2026-11-04 之前所有 launcher 要么没传这个参数、要么写死 `-p 11451`，和容器实际 sshd 端口不一致，已统一改成 `SSH_PORT` 变量，见 QA.md） |
| 日志 | launcher 自己还会 tee 一份 `log-<N>nodes-<时间>.log` 到 example 目录 |
| 并行 | `TP/PP/EP/CP/MBS/GBS/SEQ_LEN` 写在 train 脚本开头 |

```bash
python3 $R --hosts bw67 --launcher $F/Megatron-LM-das/examples/qwen3/run_qwen.sh \
    --arg '{hostfile}' --arg '{nodes}' --env TRAIN_SCRIPT=train_qwen3_8B.sh \
    --env NCCL_ENV=skills/common/env.sh --env SSH_PORT=25951 --max-iters 10
```

### Megatron-bridge-das

| 项 | 内容 |
|---|---|
| launcher | `examples/<族>/train_*.sh <RECIPE> <MODE> [hostfile] [节点数]`，MODE = `dryrun`（只解析配置）/ `train` |
| recipe | 名字里带卡数，如 `qwen35_text_9b_pretrain_8gpu_bw1000_bf16_config`；脚本 `SUPPORTED_RECIPES` 里有列表 |
| 模型 | `LOCAL_HF_PATH`：qwen35_text / qwen35_vl 是 `${LOCAL_HF_PATH:-默认}`，**可用 --env 覆盖**；deepseek / glm / minimax / qwen38 是 `export LOCAL_HF_PATH=/path/to/...` **写死占位** |
| 数据集 | pretrain 用 mock（`DATASET_BLEND=null`），不需要数据；sft：qwen35_text_sft 的 `DATA_PATH` 可覆盖，deepseek / glm52 sft 写死 `/path/to/...`；qwen35_vl 非 mock recipe 需要 `DATA_PATH`（Energon 格式） |
| 已知问题 | qwen35_text_pretrain 默认模型路径 `qwen3.5-35B-A3B`（小写 q），实际目录是 `Qwen3.5-35B-A3B`；9b recipe 也默认指向 35B 目录，跑 9b 要 `--env LOCAL_HF_PATH=.../qwen3.5-9b` |
| 端口 | `PORT`（mpirun ssh 端口，默认 11453 等）、`MASTER_PORT` 写死在脚本；多节点要 `--env PORT=25901` |
| 卡数 | `GPUS`（每节点，默认 8） |
| 日志 | 另存一份到 `examples/<族>/<RECIPE>_tp.._pp.._ep.._cp../train_logs/` |

```bash
# 先 dryrun，再 train
python3 $R --hosts bw67 --launcher $F/Megatron-bridge-das/examples/qwen/train_qwen35_text_pretrain.sh \
    --arg qwen35_text_9b_pretrain_8gpu_bw1000_bf16_config --arg dryrun \
    --env LOCAL_HF_PATH=/path/to/modules/qwen3.5/qwen3.5-9b
```

### ms-swift

| 项 | 内容 |
|---|---|
| launcher | `ms-swift/run_qwen.sh <hostfile> <节点数>`，每节点起 1 个进程（slots=1），再由 `megatron sft` 起 8 卡 |
| 选模型 | launcher 里 `script=train_qwen35_35b_a3b.sh` **写死** |
| 模型 / 数据 | 训练脚本里 `--model`、`--dataset` **写死**；`--dataset` 指向 `/data/home/wangxj/...`，本机不存在 |
| env 脚本 | 训练脚本 `source /cfs/megatron-code/ms-swift/env.sh`，`/cfs` 在本机**不存在** |
| 日志 | launcher `tee` 到 `/data/home/wangxj/.../logs/`，目录不存在会让 tee 失败 |
| 端口 | mpirun ssh `-p 29511`、`MASTER_PORT=29510` 写死 |

ms-swift 目前**不能直接跑**：上面三处写死路径要用户先决定怎么改（改脚本 / 复制一份改 / 换路径）。

## 常见报错 → 原因

| 现象 | 原因 / 处理 |
|---|---|
| `Permission denied (publickey)` | 节点或容器间 ssh 免密没配 |
| `Connection refused/timed out` 在 mpirun 阶段 | `plm_rsh_args` 端口和容器 sshd 端口（25901）不一致 |
| `No such container` | 容器没建或没运行，回 hcu-node-prep 建容器 |
| `No such file` / `/path/to/` | 路径写死、占位或不在挂载内 |
| `out of memory` / `HIP failure: 'out of memory'` | 先按上面「显存调试」看 OOM 堆栈位置：优化器/DDP 阶段用优化器 offload+切分，激活阶段用 recompute/activation offload；不要一律加 recompute。`status.json`/`result.md` 的 `vram_peak_percent` 接近 100% 说明确实是显存吃满 |
| 没有 iteration 行就退出 | 看 `train.log` 末尾，多为初始化（环境、路径、通信）失败 |

`result.md` 会自动匹配这些模式写「可能原因」；匹配不到就贴末尾报错行。

## 批量执行（测试列表）

单个模型调通后，按 hcu-test-plan 生成的测试列表批量跑（命令假定在项目根目录下执行）：

```bash
B=skills/hcu-run-model/scripts/run_cases.py
R=runs
python3 $B $R/<run_id> --dry-run          # 打印每个用例的命令
python3 $B $R/<run_id>                    # 全部 wave
python3 $B $R/<run_id> --wave 1
python3 $B $R/<run_id> --case <id>        # 单个用例（会重跑）
```

- 同一 wave 的用例在不同节点上并发；到 site.json 的 `max_iters` 停，超过 `case_timeout_min` 记 timeout。
- 已完成（status 为 done）的默认跳过，中断后重跑同一命令即可续跑。
- 写 `$R/<run_id>/logs/<id>.log`、`status/<id>.json`。`bench_log.md` / `run_log.md` 里整批执行前后各一条，每个用例启动前（节点、命令、停止条件）和结束后（状态、步数、问题与处理方案）也各一条。
- 和单模型调试一样，每个用例跑的同时也会每 5 秒轮询各节点的 `hy-smi --showmemuse`，把峰值写进 `status/<id>.json` 的 `vram_peak_percent`，`bench_log.md`/终端输出里每个用例结束那行也会带上。
- 长时间运行放后台，定期看 `status/*.json` 和日志尾部向用户报告进度。
