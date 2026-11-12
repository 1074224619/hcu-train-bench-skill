# QA 记录

按 `skills/SKILL.md`（hcu-bench 主流程）执行过程中发现的问题。只保留可复用的根因/排查方法/结论和待办；一次性的修复历史已删除。

## 一、环境与可复用结论

### NCCL / RCCL

- **多节点训练卡死 / NCCL bootstrap 报错**：根因是 `env.sh` 的 shca 分支 `export NCCL_NET_PLUGIN=shca`，强制加载有 bug 的 `librccl-net-shca.so`（`ncclIbAccept` → `socketFinalizeAccept ... wrong type`）。已删除该行，改用 NCCL 内置传输层。
  - 排查方法：先用纯 `torch.distributed` 最小脚本（`runs/debug/nccl_minimal_test.py` + `run_minimal.sh`）复现，再对照 GDR/IB/channel/topo/节点对逐项排除，最后二分 `env.sh` 变量；对比“完全不 source 自定义 env.sh”最快。
- `NCCL_PXN_DISABLE=1` 会影响集合通信和 alltoall 性能；压测时确认 env.sh 是否打开了它。
- `UCX_NET_DEVICES=mlx5_bond_0:1,mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1` 影响能否跑通；缺失时报 `UCX returned connect error: Destination is unreachable` / `hcol ... Hcol library init failed`。
- RCCL/NCCL 2.27.7：torch 自带的 nccl 和系统 `/opt/dtk/lib/librccl.so.1` 是同一份库，版本问题不用怀疑这一层。

### 容器 / 集群

- 容器用 `--network=host --ipc=host`，`/dev/shm` 与主机共享；看到遗留的 `nccl-*` 文件是正常现象，不影响功能。
- 节点占用随时变化，批量跑前务必重跑 `idle`。
- 跑模型/诊断脚本被中断（超时、Ctrl-C、TaskStop）后，远端 `mpirun`/`prterun`/训练进程不会自动清理，需 `ps -ef` 找到 PID 后 `kill -9`，否则同端口会报 `EADDRINUSE`。
- `hostfile_slots` 不会跟 `hostfile` 自动同步：改完要 `sed 's/$/ slots=8/' hostfile > hostfile_slots`。
- 镜像 tag 要和 `skills/common/site.example.json` 同步（曾因 `runs/site.json` 停在旧 tag，导致新镜像里修好的问题反复复现）；换镜像后 `docker pull` 确认各节点 digest 一致再 `recreate --yes`。
- 嵌套 submodule 要初始化：`3rdparty/Megatron-LM` 为空会掉回镜像自带的残缺 datasets（缺 Makefile，报 `make: *** No targets specified`）。bridge 的嵌套副本无 `.git`，`skills/common/env.sh` 复用 `Megatron-LM-das/3rdparty/Megatron-LM`。
- ms-swift 暂不可跑：`frameworks/ms-swift-das` 子模块内容未拉全，`run_qwen.sh` 写死模型/数据集/env/日志路径；`frameworks_enabled` 暂不含它。

### Megatron-bridge / recipe

- **BW 卡 MoE 只有 `deepep` 和 `alltoall` 两个通信后端，没有 HybridEP**（镜像里只有 `deep_ep`）。遇到 `ImportError: HybridEP is not installed` 就改 recipe 的 `moe_token_dispatcher_type`（如 `alltoall`），不要试图安装 HybridEP。
- launcher 的命令行参数会**无条件覆盖** recipe 里的默认值，两者要显式对齐；不要照抄 recipe 函数内部的 `pipeline_model_parallel_size`。
- 并行参数按模型类型给：dense（9B/27B）`EP>1` 会被 Megatron-Core 拒绝（`num_moe_experts must be non None to use expert-parallel`）；MoE 才用 EP 摊专家。
- **`Recipe '...' not found` 看三处注册链**：`env/env.sh` 把 `hcu_megatron_bridge/.../recipes/__init__.py` cp 到 `3rdparty/Megatron-Bridge/src/megatron/bridge/recipes/`，它 `import *` 上层 `recipes/qwen/__init__.py`，上层再从 `recipes/qwen/qwen35.py` 别名层导入到 `bw1000/*`。新增 recipe 要三处同步导出。
- recipe/tokenizer 路径优先读 `LOCAL_HF_PATH`（`_hf_path()`）；tokenizer 是 HuggingFace 目录时必须用 `--tokenizer-type HuggingFaceTokenizer`，用 `Llama2Tokenizer` 会报 `Is a directory`。
- 新增 bridge launcher 时：单节点 `train` 分支也要 `source /opt/dtk/env.sh`（否则 `device_count()=0`，报 `No CUDA GPUs are available`）；`HOSTFILE` 为空时不要生成 `hostfile_slots`（`set -euo pipefail` 下 `grep` 返回 1 会让脚本直接退出）。
- 平台变体（bw1000/bw1100）要拼进 **case id**，不能拼进 `model` 字段——`build_matrix.py` 用 `model` 查 `site.json` 的 `models` 预设。
- 显存估算：397B bf16 权重+梯度+fp32 Adam 约 4.4 TB；32 卡约 139 GB/卡（144G 卡没有余量放激活），64 卡约 69 GB/卡。大模型切分按这个量级先估，再 dryrun 验证。

## 二、待处理 / 未解决

- **Megatron-bridge `switch_mcore.sh` 缺 `.main.commit`/`.dev.commit` pin 文件**：10 个 `train_*.sh` 里 8 个会调用它（`minimax_m3`、`qwen35_vl`、`glm52`/`glm53`、`deepseekv4_flash`/`pro`），报 `[switch_mcore] ERROR: Missing commit file` 退出。需决定补 pin 文件、注释掉调用还是其它。
- **`qwen35_vl_2b/4b/9b`（含 mock）缺模型路径**：`site.json` 的 `models` 和 `modules/qwen3.5/` 下都没有对应 VL 权重，跑起来会报 `model directory not found`。要跑 VL 需先补目录和 `models` 条目。
- **`qwen3_30b_a3b`（2 节点/16 卡）默认 tp=1/pp=2/ep=8 会 OOM**，`train_qwen3_30B_A3B.sh` 没有 recompute/offload 选项；是否加、加哪种粒度待确认（可先用 `tune_plan.py` 看建议）。
- **“所有脚本默认非 offload/非重计算，需要重计算的要标注”的策略未落地**：现状是相关选项基本没有（不是有但关掉），哪些用例需要、标注在哪还没定。
- **rccltest 仍有问题**：具体现象待补充。
