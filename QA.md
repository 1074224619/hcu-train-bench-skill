# QA 记录

按 `skills/SKILL.md`（hcu-bench 主流程）执行过程中发现的问题。按"已修复""待处理""环境知识"
分类整理，不按时间顺序；每条尽量给文件位置和根因，方便复用。

## 一、已修复的 bug

### 集群环境 / `skills/common/env.sh`

- **【最关键】多节点训练卡死/报 NCCL bootstrap 错误**：根因是 shca 分支里
  `export NCCL_NET_PLUGIN=shca`，强制 NCCL 加载 `librccl-net-shca.so` 这个自定义 RDMA 插件，
  这个插件的 `ncclIbAccept` 函数本身有 bug（`socketFinalizeAccept ... wrong type` 协议错误）。
  **已删除这一行**，NCCL 改用自己内置的传输层，2 节点训练验证通过（真实 `qwen3_32b` 跑进训练
  循环，不再卡在 barrier/bootstrap）。
  - 排查方法留档（遇到类似"多机 NCCL 报错但单机正常、`rccl2` 标准测试也 PASS"时可参考）：
    写一个不经过训练框架的最小复现脚本（纯 `torch.distributed.init_process_group`+`barrier()`+
    `all_reduce`，`mpirun --map-by slot --rank-by slot`），依次对照排除 GDR
    （`NCCL_NET_GDR_LEVEL=0`）、IB 开关（`NCCL_IB_DISABLE=1`）、并发 channel 数
    （`NCCL_MAX_NCHANNELS=2`）、`NCCL_TOPO_FILE`、具体节点硬件（换 3 组节点对复现一致）；
    真正见效的是对比"完全不 source 自定义 env.sh、只用 `/opt/dtk/env.sh`"是否能跑通——能，
    再二分 `env.sh` 里的变量，定位到 `NCCL_NET_PLUGIN`。
  - 排查脚本留在 `runs/debug/`（`nccl_minimal_test.py` + `run_minimal.sh`），可复用；注意
    `run_minimal.sh` 改 `HOST` 后要重新生成 `hostfile_slots`
    （`sed 's/$/ slots=8/' hostfile > hostfile_slots`），这个文件不会跟 `hostfile` 自动同步。
- shca 分支的 `NCCL_IB_HCA` 改成用 `ibstatus` 自动探测 ACTIVE+LinkUp 端口（参考 mlnx 分支的写法），
  探测失败才退回硬编码 `shca_0:1,shca_1:1,shca_2:1,shca_3:1`；原来多出的
  `NCCL_SOCKET_IFNAME=eth0` 硬编码已删除（会覆盖前面已有的网卡自动探测逻辑）。

### Megatron-LM-das（`frameworks/Megatron-LM-das`）

- 旧分支（`core_v0.18.2`）的全部 `examples/*/run_*.sh` 里 `DTK_ENV`/`DATA_PATH`/
  `TOKENIZER_MODEL_PATH`/`NCCL_ENV` 都是普通赋值（不是 `${VAR:-默认}`），`--env` 完全不生效。
  换到 `core_v0.19.2` 分支后这几个变量都已经是可覆盖写法，问题消失，不用再改框架脚本。
- 容器镜像里 `/usr/local/lib/python3.12/site-packages/megatron/core/datasets` 缺 Makefile，
  编译数据集索引 helper 会报 `make: *** No targets specified`。根因是
  `3rdparty/Megatron-LM`（嵌套 git submodule，指向 `github.com/NVIDIA/Megatron-LM.git`）
  没有 checkout，PYTHONPATH 指向了空目录才掉回镜像自带的残缺版本。
  **已 `git submodule update --init` 补齐**，问题解决。

### Megatron-bridge-das（`frameworks/Megatron-bridge-das`）

- 全部 `examples/*/train_*.sh`（除 `qwen35_vl_pretrain.sh`）开头都有一行多余的
  `PORT="" # Set the SSH port...`，把后面 `PORT="${PORT:-12345}"` 的可覆盖逻辑废掉
  （bash 的 `:-` 对空字符串和未设置一样处理）；`LOCAL_HF_PATH` 也是普通赋值
  `export LOCAL_HF_PATH=/path/to/X`，同样不可覆盖。**已修复 9 个文件**：删掉多余的
  `PORT=""` 行，`LOCAL_HF_PATH` 改成 `${LOCAL_HF_PATH:-默认}`。
- `source ${MEGATRON_PATH}/env/env.sh` 原来是硬编码，没法像 Megatron-LM-das 的 `NCCL_ENV`
  一样换一份。**已修复**：全部 10 个 `train_*.sh` 加了
  `BRIDGE_ENV_SCRIPT="${BRIDGE_ENV_SCRIPT:-${MEGATRON_PATH}/env/env.sh}"`，3 处 `source`
  调用换成 `source ${BRIDGE_ENV_SCRIPT}`；`build_matrix.py` 对 `megatron-bridge` 用例自动注入
  `BRIDGE_ENV_SCRIPT=site.json 的 env_script`（和 `megatron-lm` 用例的 `NCCL_ENV` 注入方式一致）。
- Megatron-Bridge 自己嵌套的 `3rdparty/Megatron-LM`（PYTHONPATH 里指向的那份）同样是空的，
  且这份环境没有 `.git`/`.gitmodules`，不能 `git submodule update --init`。**已修复**：
  `skills/common/env.sh` 改成复用 Megatron-LM-das 已验证可用的
  `frameworks/Megatron-LM-das/3rdparty/Megatron-LM`（用 `[ -d frameworks/Megatron-bridge-das ]`
  判断要不要做这段，不影响只跑 Megatron-LM-das 时的行为）。
- `recipes/qwen/__init__.py` 只 import 了 `qwen35_text_35b_a3b_...` 漏了同目录
  `qwen35.py` 里本来就有导出的 `qwen35_text_9b_pretrain_8gpu_bw1000_bf16_config`，
  导致 `qwen35_text_9b` 报 `Recipe not found`。**已修复**：补上这个函数的 import 和 `__all__`。
- `recipes/qwen/bw1000/qwen35.py` 的两个 recipe 函数用
  `AutoConfig.from_pretrained(硬编码 hub id)`，`LOCAL_HF_PATH` 对这条路径完全不生效（会去
  hub 拉取而不是用本地路径）；同目录 `qwen35_vl.py` 已有正确写法（`_hf_path()` 优先读
  `LOCAL_HF_PATH`）。**已修复 `qwen35.py`**，搬了同样的 `_hf_path()` 逻辑。
  **`qwen38.py` 是同样的模式、同样的问题，这次按用户要求只改了 `qwen35.py`，`qwen38.py`
  还没改**（见下面"待处理"）。
- 所有 `train_*.sh` 的 tokenizer 实际都是 HuggingFace 格式目录，但 10 个脚本
  （`mixtral_8x7B/8x22B`、`gpt_567B`、`aibenchmark`、`glm5_300B`）硬编码
  `--tokenizer-type Llama2Tokenizer`（期望 sentencepiece `.model` 文件，实测
  `mixtral_8x7b` 报 `RuntimeError: ... Is a directory`）。**已全部改成 `HuggingFaceTokenizer`**。
- 容器缺 `megatron-energon` 包导致 `megatron.bridge.data.__init__.py` 启动就
  `ModuleNotFoundError`（Energon 相关 import 是无条件的，纯文本 pretrain 用例也会被挡住）。
  **换新镜像（`sf_b075-20261105`）后已内置，不需要再手动装**。

### `hcu-test-plan`（`skills/hcu-test-plan/scripts/build_matrix.py`）

- 多个框架族的 `run_*.sh` 共享同一个硬编码默认端口（`qwen1.5`/`qwen2`/`qwen2.5`/`llama2`/
  `llama3`/`GLM` 默认 `25900`；`mixtral`/`gpt3`/`aibenchmark`/`deepseek_v3` 默认 `25905`；
  `qwen3` 默认 `11452`），同一 wave 里不同节点组但同端口族的用例并发跑时端口冲突
  （`EADDRINUSE` 或 collective watchdog 超时）。**已修复**：`build_matrix.py` 给每个
  `megatron-lm` 用例的 env 按全局序号分配独立的 `PORT`/`MASTER_PORT`（从 `28000` 起，
  全矩阵唯一）。

## 二、待处理 / 未解决（需要用户决定或后续跟进）

- **`scan_examples.py` 的 `scan_ms_swift()` 扫描路径写错**：用的是 `root / "ms-swift"`，
  实际目录名是 `ms-swift-das`（和 Megatron-LM-das/Megatron-bridge-das 两个用完整目录名的
  约定不一致），导致 ms-swift 用例永远扫不到。当前 `frameworks_enabled` 没包含 ms-swift，
  不阻塞；以后要开 ms-swift 时除了补子模块内容还要修这个路径。
- **Megatron-bridge 的 `switch_mcore.sh` 缺 `.main.commit`/`.dev.commit` pin 文件**：
  10 个 `train_*.sh` 里 8 个（`minimax_m3_pretrain`、`qwen35_vl_pretrain`、`glm52_pretrain`/
  `glm52_sft`、`glm53_pretrain`、`deepseekv4_flash_pretrain`/`flash_sft`/`pro_pretrain`）
  会主动调用这一行，报 `[switch_mcore] ERROR: Missing commit file: .main.commit` 直接退出；
  只有 `qwen35_text_pretrain`/`qwen38_2.4t_pretrain` 这行是注释掉的，所以能跑。需要用户决定
  怎么处理（补 pin 文件 / 注释掉调用 / 其它）。
- **`qwen38.py` 的 `LOCAL_HF_PATH` 不生效**：和已修复的 `qwen35.py` 同样的模式
  （`AutoConfig.from_pretrained` 硬编码 hub id），这次只改了 `qwen35.py`，`qwen38.py`
  还没处理。
- **`qwen35_vl_2b/4b/9b`（含 mock 变体）缺模型路径**：`site.json` 的 `models` 里一直没配，
  跑的时候用脚本默认占位 `/path/to/Qwen3.5-VL-2B` 报 `model directory not found`。
- **`qwen3_30b_a3b`（2 节点/16 卡）默认并行配置（tp=1/pp=2/ep=8）会 OOM**，`train_qwen3_30B_A3B.sh`
  完全没有 `recompute`/`offload` 选项。是否要加开重计算、以及加哪种 granularity，待用户确认。
- **"所有脚本应该是非 offload/非重计算版本，需要重计算的要标注" 这条策略还没落地**：现状核对
  （`qwen3_32B` 里仅有的 offload 相关 3 行本来就是注释掉的，其余脚本完全没有相关选项）符合
  "非 offload" 这一半，但"重计算"选项目前是彻底没有、不是"有但关掉"，按这条策略该怎么补
  （哪些用例需要、标注在哪）还没确认。

## 三、环境 / 集群知识（供参考，非 bug）

- **节点占用随时变化**：批量跑模型前务必重跑一次 `idle`；本次会话里同一节点在几分钟内从空闲
  变占用、又变回空闲的情况多次发生，是集群共享的正常现象，不是本项目的问题。
- **容器用 `--network=host` + `--ipc=host`**：不是独立网络/IPC 命名空间，`/dev/shm` 是主机级共享的
  （排查时发现过几周积压的其他任务留下的 `nccl-*` 共享内存文件，属正常现象，不影响功能）。
- **RCCL/NCCL 版本 2.27.7**：torch 自带的 nccl 和系统 `/opt/dtk/lib/librccl.so.1` 是同一份库
  （没有单独打包的 bundled 版本），排查版本不兼容时不用怀疑这一层。
- **镜像 tag 要和 `skills/common/site.example.json` 保持同步**：曾发生过 `runs/site.json`
  停留在旧 tag（`sf_b075-20261014-1059`）而 `site.example.json` 已经更新到新 tag
  （`sf_b075-20261105`）的情况，导致新镜像里修好的问题（如缺 `megatron-energon`）在旧镜像上
  一直复现。换镜像时记得 `docker pull` 确认 5 个节点 digest 一致后再 `recreate --yes`。
- **跑模型/跑诊断脚本中断（超时、`TaskStop`、Ctrl-C）后，远端 `mpirun`/训练进程不会自动清理**：
  本地包装进程被杀不影响已经通过 ssh 在远端节点上拉起的 `prterun`/`pretrain_gpt.py`/测试脚本，
  需要手动 `ps -ef | grep ...` 找到 PID 再 `kill -9`，否则下次用同端口会报 `EADDRINUSE`。
- **`hostfile_slots` 不会跟 `hostfile` 自动同步**：手动改 `hostfile` 后要重新
  `sed 's/$/ slots=8/' hostfile > hostfile_slots`，否则 `mpirun --hostfile` 用的还是旧节点对。
