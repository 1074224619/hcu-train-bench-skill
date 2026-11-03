# BenchAgent

在 HCU 集群上批量测试 Megatron-LM-das / Megatron-bridge-das / ms-swift 的训练性能，并把结果整理进 dcu-train-cookbook。

流程分 4 步，每步由一个 skill 负责，步骤之间只通过 `runs/` 下的文件交接：

| 步骤 | skill | 做什么 | 产出（`runs/` 下） |
|---|---|---|---|
| 1 节点准备 | `hcu-node-prep` | idle 筛节点、建容器、platform/resource/RCCL 检查 | `hostfile` `idle_hosts` `node_status.md` `healthy_hosts` |
| 2 测试计划 | `hcu-test-plan` | 扫描框架 example 生成用例目录，按可用节点排测试列表 | `<run_id>/matrix.json` `matrix.md` |
| 3 跑模型 | `hcu-run-model` | 单个模型调试；按测试列表批量执行 | `<run_id>/logs/` `status/` |
| 4 结果入库 | `hcu-cookbook` | 解析 tflops / time per iter / tgs，写入 cookbook | `<run_id>/results.json`，cookbook 改动 |

两种用法：**手动执行**（逐条跑脚本）和 **自动测试**（交给 Claude Code 按 skill 执行）。两者用同一套脚本、同一份配置和日志。

## 目录结构

```
BenchAgent/
├── frameworks/                 被测框架和工具（各自是独立的 git 仓库）
│   ├── Megatron-LM-das/  Megatron-bridge-das/  ms-swift/
│   ├── cluster-manager-das/    节点筛查，使用其中的 hcu-envcheck
│   └── dcu-train-cookbook/     性能数据最终写入这里
├── skills/                     脚本只依赖 Python 3 标准库
│   ├── SKILL.md                主 skill hcu-bench：总流程、分工、共享约定
│   ├── common/                 共享代码（common.py、benchlog.py）、site.example.json、env.sh
│   ├── hcu-node-prep/  hcu-test-plan/  hcu-run-model/  hcu-cookbook/
└── runs/                       运行时数据
    ├── site.json               本集群配置
    ├── hostfile.all            全部节点（用户给的原始列表，脚本不修改）
    ├── hostfile                可 ssh 的节点，建容器用
    ├── idle_hosts              可 ssh 且空闲的节点
    ├── node_status.md          节点标注（ssh / 空闲 / 占用容器或进程）
    ├── healthy_hosts           空闲且通过检查的节点，测试用它
    ├── bench_log.md            中文运行日志
    ├── envcheck/               hcu-envcheck 输出
    └── <run_id>/               每次测试的矩阵、日志、状态和结果
```

节点列表是一条流水线：`hostfile.all`（用户给的全部节点）→ `hostfile`（可 ssh，建容器用）→ `idle_hosts`（空闲，跑测试用）→ `healthy_hosts`（空闲且检查通过，真正测试用）。

## 依赖的仓库

| 目录 | 仓库 |
|---|---|
| Megatron-LM-das | git@github.com:1074224619/Megatron-LM-das.git |
| Megatron-bridge-das | git@github.com:1074224619/Megatron-bridge-das.git |
| ms-swift | https://github.com/modelscope/ms-swift.git |
| cluster-manager-das | https://github.com/zhaoyu111111/cluster-manager-das.git |
| dcu-train-cookbook | http://42.228.13.241:10068/dcutoolkit/deeplearing/dcu-train-cookbook.git |

新环境下 clone 到 `frameworks/` 下对应的目录名即可。

## 环境要求

- 登录节点有 Python 3，能免密 ssh 到各计算节点；ssh 不通的节点会被报告并排除在 `hostfile` 之外。
- 有一套密钥对（默认各节点宿主机的 `/root/.ssh`，也可以用 site.json 的 `ssh_dir` 指向别的目录，如共享 home 下的 `.ssh`），公钥在自己的 `authorized_keys` 里。建容器后会把这套密钥 `docker cp` 进容器的 `/root/.ssh`（不是挂载），再在容器内免密 ssh 自己做校验，多节点 mpirun 也靠它；缺密钥的节点建容器会报 `ssh-identity` 失败，需要管理员配置。
- `frameworks/`、`runs/`、数据集、模型，以及 `ssh_dir` 指向的目录，在登录节点、计算节点、容器内绝对路径相同。容器挂载只来自 site.json 的 `container.mounts`（加默认的 `/opt/hyhal` 只读挂载），没有 `/root/.ssh` 挂载。

## 配置（首次）

```bash
cd /path/to/BenchAgent     # 项目根目录
cp skills/common/site.example.json runs/site.json
vi runs/site.json        # 镜像、容器名、ssh 端口、挂载、平台、每节点卡数、env_script、框架和 cookbook 路径
vi runs/hostfile.all     # 每行一个节点名
```

关键字段：

| 字段 | 说明 |
|---|---|
| `container.name` / `image` / `ssh_port` | 测试容器名、镜像、容器内 sshd 端口 |
| `container.mounts` | 挂进容器的目录，需覆盖 frameworks、runs、数据集、模型 |
| `frameworks_enabled` | 现阶段只跑哪些框架（`megatron-lm`/`megatron-bridge`/`ms-swift` 的子集），不写 = 三个框架都不限制。改了之后不用重新 `scan_examples.py`，`build_matrix.py` 下次生成列表就生效 |
| `env_script` | 集群环境脚本（库路径、IB/RCCL 变量），不同集群各自指定，默认 `skills/common/env.sh`。check、rccl 在容器内先 `source` 它；所在目录以只读方式同路径挂进容器，**改位置要重建容器**，只改内容不用。`NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` 会自动探测每个节点自己的默认路由网卡（`/proc/net/route`，没有再退回 `route -n`），不用手动按节点改。Megatron-LM-das 的 `run_*.sh` 默认不会用它（退回框架自带 `requirements/env.sh`，没有网卡自动探测，在本集群会导致 NCCL 初始化报错），`build_matrix.py` 批量跑时已自动把它注入每个用例的 `NCCL_ENV`；单独用 `hcu-run-model` 调试要自己 `--env NCCL_ENV=<env_script>` 传。`build_matrix.py` 同时会给每个 megatron-lm 用例的 `DTK_ENV` 设默认值 `/opt/dtk/env.sh`（除 qwen3 外其余 launcher 的 `DTK_ENV` 默认是空字符串，`source` 会直接报错退出） |
| `rccl_baseline` | np=8 的 RCCL busbw 性能基线文件，默认 `skills/common/rccl_baseline`。缺这个文件 `rccl` 步骤每个 collective 会在 prepare 阶段直接报错「未找到性能基准」；和 `env_script` 一样**改位置要重建容器**，只改内容不用 |
| `models.<model>` | 按 `catalog.json` 的 case model 名手动填数据集/模型/tokenizer 路径（Megatron-LM：`data_path`/`tokenizer_path`；Megatron-bridge：`data_path`/`model_path`），`build_matrix.py` 批量跑时自动注入进每个用例的 env，不用手改各框架的 `run_*.sh`。`site.example.json` 里已按 `examples/` 下的全部模型族（qwen3 全系列、llama2/llama3/mixtral/qwen1.5/qwen2/qwen2.5）列好条目：没有真实预训练权重的规格复用同族 tokenizer、按 `docs/getting-started.md` 的方法重新分词生成数据集，只做随机初始化的性能测试；真没有可用 tokenizer/数据集的（glm5_300b、deepseekv3_671b、gpt_567b、aibenchmark）留空占位，不会被误启用 |
| `max_iters` / `warmup_iters` / `case_timeout_min` | 批量跑时每个用例跑多少步停、解析时跳过的步数、超时 |

## 手动执行

以下命令都在项目根目录运行。每个脚本的参数用 `-h` 查看，细节见各子目录的 SKILL.md。

```bash
K=skills
R=runs
P="bash $K/hcu-node-prep/scripts/prepare_nodes.sh $R/site.json"
```

### 1. 节点准备

```bash
$P idle                     # 只读空闲检查，自动生成 hostfile / idle_hosts / node_status.md
$P status                   # 查看各节点测试容器状态
$P create                   # 预演，打印 docker run 计划，不改动节点
$P create --yes             # 在 hostfile 的全部节点建容器（含被占用的节点）
$P check                    # idle_hosts 上做 platform / resource 检查
$P rccl                     # idle_hosts 上做单机 RCCL 测试（自动带 site.json 的 rccl_baseline 和 --skip-idle-check）
$P rccl2                    # idle_hosts 两两配对做双机 RCCL 测试；节点数为奇数时最后一个单独跑单机 RCCL
$P healthy <check目录> <rccl目录>   # 生成 healthy_hosts
```

- 容器在所有可 ssh 节点都建，节点空出来就能直接用；建在被占用的节点上不影响别人的任务。check、rccl 和跑模型只用空闲节点。
- 建容器的预检任一节点失败，所有节点都不会建。同名容器已存在时用 `$P recreate --yes`，它会删除旧容器。
- `idle` 发现节点被占用时只标注，不要用 `--release` 停别人的容器。
- 占用情况随时变化，跑模型前重跑一次 `$P idle`。
- `rccl` 默认带 `--skip-idle-check`，跳过它自己开跑前的卡空闲复检：如果节点在 `idle` 之后变忙，`rccl` 会照样在忙碌的卡上跑而不是报错停下，结果可能不准、也可能影响别人的任务。跑 `rccl` 前最好保证 `idle_hosts` 是刚刷新的。
- `rccl2` 是多机训练前的排查手段：`rccl`/`check` 都只测单机范围，测不出跨节点 RDMA/RoCE 链路和容器间 ssh 是否通；真遇到多机用例报 `root@<node>: Permission denied` 或 `PRTE has lost communication with a remote daemon` 这类错误，先按 `idle_hosts` 顺序两两跑一次 `rccl2`，比直接跑多机训练失败后再查更快定位到具体哪一对节点有问题。双机那组用 `--container-ssh-port`（取 site.json 的 `container.ssh_port`），和单机 `rccl` 共用同一份 `rccl_baseline`/`env_script`/`--skip-idle-check`；某一对 FAIL 就把这对节点排除出多机测试组，不要直接改框架脚本绕过。
- `check`（platform/resource，含跨节点网络/平台指标）FAIL 不代表节点不能用：`rccl` 本身是单机测试（group-size 1），不依赖跨节点网络，`check` 失败照常跑 `rccl`。如果单机 `rccl` PASS，`$P healthy` 只传 rccl 的结果目录（`$P healthy <rccl目录>`，不传 check 目录），这样的节点会保留下来但只排单机（1 节点）用例，多机用例仍要排除，等网络侧修好、`check` 重跑通过再放开。
- hcu-envcheck 的报告聚合曾有「分组内最后一项测试被误判成 INCOMPLETE」的 bug（原始日志明明 PASS，`summary.md` 判成 INCOMPLETE/FAIL），根因是容器内 `stat()` 缓存跨容器/宿主机文件系统边界读到旧属性，已于 2026-10-28 定位并修复，细节见 `frameworks/cluster-manager-das/hcu-envcheck/docs/KNOWN_LIMITATIONS.md`「报告聚合」条目；未走过这次修复的旧版本部署，`check`/`rccl`/`healthy` 判定为失败时仍建议去对应 `group-00N.log` 原始输出交叉核对。

### 2. 测试计划

```bash
python3 $K/hcu-test-plan/scripts/scan_examples.py frameworks -o $K/hcu-test-plan/catalog.json --site $R/site.json
python3 $K/hcu-test-plan/scripts/build_matrix.py --site $R/site.json --catalog $K/hcu-test-plan/catalog.json \
    --hosts $R/healthy_hosts [--only 'qwen3|glm'] [--max-nodes 4]
```

- 扫描后人工审 `catalog.json`：`enabled`、`nodes`、`parallel`、`cookbook_doc`。有未配置路径的用例默认关闭，原因在 `notes`。重新扫描会覆盖 catalog，有人工改动先备份。
- `build_matrix.py` 按节点数装箱成 wave，同一 wave 的用例占用不同节点并发。先检查 `runs/<run_id>/matrix.md`，再执行。

### 3. 跑模型

先单个模型调通，再批量：

```bash
# 单个模型：预演 → 单节点 10 步 → 多节点
python3 $K/hcu-run-model/scripts/run_model.py --site $R/site.json --hosts bw67 \
    --launcher $PWD/frameworks/Megatron-LM-das/examples/qwen3/run_qwen.sh \
    --arg '{hostfile}' --arg '{nodes}' --env TRAIN_SCRIPT=train_qwen3_8B.sh \
    --env NCCL_ENV=$PWD/skills/common/env.sh --max-iters 10 [--dry-run]

# 批量
python3 $K/hcu-run-model/scripts/run_cases.py $R/<run_id> --dry-run   # 打印每个用例的命令
python3 $K/hcu-run-model/scripts/run_cases.py $R/<run_id>             # 全部 wave；中断后重跑同一命令续跑
python3 $K/hcu-run-model/scripts/run_cases.py $R/<run_id> --wave 1    # 或 --case <id>、--rerun
```

Megatron-LM-das 的用例必须显式传 `NCCL_ENV=<site.json 的 env_script>`，不能用框架自带 `requirements/env.sh` 的默认值（会导致 NCCL 初始化报错，见「当前限制」）；走 `run_cases.py` 批量执行时由 `build_matrix.py` 自动注入，不用手传。单模型结果在 `runs/hcu-run-model/<时间>_<脚本名>/result.md`，含状态、性能和可能原因（日志统一放在 `runs/` 下，不放 `skills/` 里；没给 `--site` 时才会退回 `skills/hcu-run-model/runs/`）。三个框架接收数据集、模型路径和端口的方式不同，调试前先读 [hcu-run-model/SKILL.md](skills/hcu-run-model/SKILL.md) 的「各框架」一节。

单模型调试和批量执行跑的同时都会每 5 秒对每个节点 `docker exec <容器> hy-smi --showmemuse --json` 一次，取各卡最大值，训练结束后把每个节点的显存占用峰值（百分比）写进 `status.json`/`status/<id>.json` 的 `vram_peak_percent`，单模型的 `result.md` 和批量的 `bench_log.md`/终端输出里也能看到。

### 4. 解析并写入 cookbook

```bash
python3 $K/hcu-cookbook/scripts/parse_metrics.py $R/<run_id> [--warmup 5]
python3 $K/hcu-cookbook/scripts/export_cookbook.py $R/<run_id>            # 预览，写到 <run_id>/cookbook/
python3 $K/hcu-cookbook/scripts/export_cookbook.py $R/<run_id> --apply    # 写入 cookbook 工作区
cd frameworks/dcu-train-cookbook && git diff                              # 展示改动给用户看
```

`--apply` 会先比较：同一个模型、同一套配置（nodes/gpus/并行参数/mbs/gbs/seq_len/精度等都相同）如果表格里已有真实数据行，只有本次 tgs（tgs 不可比时退回 tflops）比已有的好才会写入并替换旧行，不如已有结果就跳过（不复制日志、不改文档，记进问题表）；预览模式会提示 `[would skip]`。`all_experiments.csv` 不受这个影响，每次 `--apply` 都会追加一行作完整实验历史。

不自动 `git add`/`commit`/`push`：提交、建分支、推送、提 MR 都留给用户自己审查后处理。

### 日志

每个脚本的每一步都在 `runs/bench_log.md` 写两条中文记录：执行前一条「开始」（命令、将执行的操作），执行后一条结果（状态、问题与处理方案、耗时、明细路径）。和某次测试相关的同时写 `runs/<run_id>/run_log.md`。脚本异常退出或被 Ctrl-C 中断时会补写一条「异常/中断」结果，所以每条「开始」后面都有对应的结果；只有「开始」没有结果，说明进程被强杀（如 `kill -9`）。实现在 `skills/common/benchlog.py`（`benchlog.Step`），常见错误识别表是其中的 `DIAGNOSES`。出错时先看最新一条和其中的明细路径。

手动执行脚本以外的命令（ssh 核对路径、查占用进程、改配置等）也建议用 `benchlog.py` 记录，方便追溯：

```bash
python3 skills/common/benchlog.py exec --site runs/site.json --target "查 bw67 卡占用" --plan "确认占用进程归属" \
    -- 'ssh bw67 ps -fp <PID>'
```

## 自动测试（Claude Code）

Claude Code 从项目的 `.claude/skills/` 加载 skill，首次建软链接：

```bash
mkdir -p .claude/skills
ln -sfn ../../skills .claude/skills/hcu-bench
for s in hcu-node-prep hcu-test-plan hcu-run-model hcu-cookbook; do ln -sfn ../../skills/$s .claude/skills/$s; done
```

在项目根目录启动 Claude Code，然后：

| 想做什么 | 怎么说 |
|---|---|
| 从头跑一轮 | `/hcu-bench`，或「按 hcu-bench 跑一遍测试」 |
| 只做某一步 | `/hcu-node-prep`、`/hcu-test-plan`、`/hcu-run-model`、`/hcu-cookbook` |
| 调试单个模型 | 「用 hcu-run-model 在 bw67 上跑 qwen3 8B」 |
| 看进度 | 「看一下 bench_log.md 最新状态」 |

Claude Code 的执行方式：

- 按 1 → 4 顺序执行，每步结束读 `bench_log.md` 最新一条，汇报状态和问题表。
- 占用状态会变，跑模型前重跑一次 idle；跑模型不顺时先用 hcu-run-model 单个调通，再回到批量。
- 脚本会自己写日志；它自己做的每一步操作（ssh 核对路径/占用、改文件、排查定位原因、按你的决定做的处理）同样要写「开始」+ 结果两条进 `bench_log.md`（和某次测试相关时同时写 `run_log.md`），不只是脚本自己写——这样 `bench_log.md` 才能完整复盘整个过程。
- 出问题先报告再处理。以下操作一定先停下来问你：建 / 删 / 重建容器（`--yes`）、释放别人占用的卡、修改框架仓库脚本、确认测试列表（matrix.md）。cookbook 导出后只 `git diff` 给你看，不会自动提交或推送。
- 首次使用需要在权限设置里允许 `ssh` 和 `prepare_nodes.sh` 等命令。

## 当前限制

- ms-swift 的 `run_qwen.sh` 写死了模型、数据集、env 脚本和日志路径，指向本集群不存在的目录，改之前不能直接跑。
- Megatron-LM-das 的 launcher 里 `DATA_PATH`、`TOKENIZER_MODEL_PATH`、`DTK_ENV`、`NCCL_ENV` 都是 `${VAR:-默认}` 可覆盖写法，但默认值不一定能直接用：`DATA_PATH`/`TOKENIZER_MODEL_PATH` 没有默认值的（除 qwen3）要在 `site.json` 的 `models.<model>` 填，见 `skills/hcu-test-plan/SKILL.md`；`NCCL_ENV` 的框架默认值（`requirements/env.sh`）在本集群会让 NCCL 初始化报 `ncclInternalError ... Attribute arch of node cpu not found`，必须显式覆盖成 `env_script`；`DTK_ENV` 除 qwen3 外其余 launcher 默认是空字符串，`source` 会直接报错退出，必须覆盖成实际的 dtk env 路径（如 `/opt/dtk/env.sh`）。`NCCL_ENV`/`DTK_ENV` 这两个 `build_matrix.py` 批量跑时都已自动注入，单独用 `hcu-run-model` 调试要自己 `--env` 传。mpirun ssh 端口和容器 sshd 端口（25901）不一致，多节点前要核对。
- `examples/` 下的 train 脚本目前都临时注释掉了 `--save`/`--load`（不写/不读 checkpoint），只做性能测试，不占用磁盘保存权重；需要断点续训或保存权重时再手动放开。
- Megatron-bridge-das 部分 recipe 的 `LOCAL_HF_PATH` 是 `/path/to/...` 占位；多节点要 `--env PORT=25901`。
- 现阶段默认只跑 megatron-lm（`site.json` 的 `frameworks_enabled`），bridge/swift 稳定后再开。
- 新模型如果在 cookbook 里还没有对应文档，要先按同目录模板建好，导出时才会写入。
- `rccl` 默认 `--skip-idle-check`，节点在 `idle` 之后变忙也会照跑，没有反向开关可以强制恢复这项检查；跑之前自己保证 `idle_hosts` 是新的。
