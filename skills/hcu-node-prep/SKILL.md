---
name: hcu-node-prep
description: 用 hcu-envcheck 准备 HCU 测试节点：idle 检测并筛出可 ssh / 空闲节点，建容器，platform/resource/单机+双机 RCCL 检查，生成健康节点列表。
when_to_use: 筛节点、看哪些节点空闲/被谁占用、建或查测试容器、做节点健康检查，或 ssh/docker/挂载相关报错时使用。
---

# hcu-node-prep（步骤 1）

以下命令假定在项目根目录下执行（`P`/`R` 换成绝对路径也一样）：

```bash
P="bash skills/hcu-node-prep/scripts/prepare_nodes.sh runs/site.json"
R=runs
```

| 步骤 | 命令 | 读 | 写 |
|---|---|---|---|
| idle | `$P idle` | `hostfile.all` | `envcheck/idle-check_*/`，然后自动 screen |
| screen | `$P screen [--idle-dir DIR]` | 最新的完整 idle 结果 | `hostfile`、`idle_hosts`、`node_status.md` |
| 容器状态 | `$P status` | `hostfile` | — |
| 建容器 | `$P create`（预演）→ `$P create --yes`（`hostfile` 全部节点，含被占用的） | `hostfile` | `envcheck/per-node-container_container-create_*` |
| 重建 | `$P recreate --yes`（会删旧容器，先征得同意） | `hostfile` | 同上 |
| 检查 | `$P check` | `idle_hosts` | `envcheck/per-node-container_platform*` |
| RCCL（单机） | `$P rccl` | `idle_hosts`、site.json 的 `rccl_baseline` | `envcheck/per-node-container_rccl*` |
| RCCL（双机） | `$P rccl2` | `idle_hosts`、site.json 的 `rccl_baseline` | `envcheck/rccl2_paired_hosts` + 对应的 `envcheck/per-node-container_rccl*`；奇数节点时还有 `envcheck/rccl2_leftover_host` + 它自己的单机 `per-node-container_rccl*` |
| 健康 | `$P healthy <check目录> <rccl目录>` | `idle_hosts` | `healthy_hosts` |

check 和 rccl 在容器内先 `source` site.json 的 `env_script`（默认 `skills/common/env.sh`，不同集群按需修改）；建容器时它所在目录以只读方式同路径挂进容器，所以**改了 `env_script` 的位置要 `recreate`**，只改脚本内容不用。命令行显式传 `--env-script` 时以命令行为准。

rccl 还要读 site.json 的 `rccl_baseline`（np=8 的 busbw 性能基线文件，默认 `skills/common/rccl_baseline`），没有基线每个 collective 会在 prepare 阶段直接报错「未找到性能基准」；它和 `env_script` 一样，**改了 `rccl_baseline` 的位置也要 `recreate`**（只改文件内容不用），因为 rccl 测试程序是在容器内读这个文件，只能靠建容器时同路径只读挂载进去。命令行显式传 `--baseline-file` 时以命令行为准。

`$P rccl` 默认带 `--skip-idle-check`：跳过 `rccl_perf_test.sh` 自己在开跑前做的卡空闲复检（否则卡在 `idle`/`screen` 之后变busy 就会在 `phase=prepare` 直接报错拒绝开跑，整组 FAIL）。这意味着**如果节点在 `idle` 之后、`rccl` 之前被别人占用，rccl 会照样在忙碌的卡上跑**，而不是报错停下——测出来的 busbw 可能不准，也可能影响到上面别人的任务。跑 `rccl` 前最好先确认 `idle_hosts` 是刚刷新的（occupancy 变化快，隔太久要重跑 `idle`）；命令行显式传 `--skip-idle-check` 时同样生效（没有反向开关可以强制重新打开这项检查）。

**`$P rccl2`：多节点训练前先测双机 RCCL**，排查跨节点 RDMA/RoCE 链路和容器间 ssh 是否通（上一轮 `lm-qwen25_32b-pretrain-16g` 这类多机用例报 `root@<node>: Permission denied` 就是这类问题，`rccl`/`check` 都测不出来，因为它们都是单机范围）。按 `idle_hosts` 的顺序两两配对、每对跑一次 `group-size 2` 的双机 RCCL（`--container-ssh-port` 用 site.json 的 `container.ssh_port`，和建容器时的 `--port` 一致）；**节点数是奇数时，配不上对的最后一个节点单独跑一次单机（group-size 1）RCCL**，不会被晒掉。两部分各自是独立的 `hcu-cluster-run` 调用、各自的 `envcheck/per-node-container_rccl_*` 结果目录，都会各写一条 `bench_log.md` 记录。`--skip-idle-check`、`rccl_baseline`、`env_script` 和 `$P rccl` 共用同一份 site.json 配置。双机 RCCL FAIL（尤其是组内两个节点的 RoCE 链路对不上、或 ssh 在建容器时没把 `ssh_dir` 的密钥对分发到位）时，该节点对**不要**编进多机测试组，换一对节点重测或等网络侧修好再重跑 `rccl2`。

**容器在 `hostfile` 的所有节点（可 ssh 的全部节点）都建，不管当前是否空闲**：占用情况随时会变，先把容器备好，节点空出来就能直接用。check / rccl 和跑模型只挑空闲节点（`idle_hosts` → `healthy_hosts`），跑模型前重跑 idle 刷新。所以 `create --yes` 不用只挑空闲节点，也不要因为节点被占用就跳过建容器；建在被占用的节点上不会影响上面别人的任务。每一步都会在 `$R/bench_log.md` 追加两条中文日志：执行前「开始」（目标节点、将执行的操作），执行后结果（状态、问题与处理方案、耗时）；脚本中途退出也会补写「异常」。终端原始输出存 `$R/envcheck/terminal/`。

**`check` 不过的节点不要直接弃用**：`$P check`（platform/resource）对某个节点报 FAIL 时（不管是 RoCE/RDMA 链路 DOWN、VRAM/利用率超阈值，还是别的原因），不要因此跳过它的 `rccl` 步骤——`$P rccl` 本身就是单机测试（group-size 1，只用本机卡做集合通信，不依赖跨节点网络），`check` 失败不代表单机 RCCL 也测不了，照常跑。

**check 不过、单机 rccl 过：只跑单机模型，`healthy` 只认 rccl 的结果**：如果 `check` FAIL 但 `$P rccl`（单机）是 PASS，说明节点内部通信没问题，问题只在跨节点网络/平台指标上。这种节点要留着用，不要排除：
- `$P healthy` 调用时**只传 `rccl` 的结果目录，不要传 `check` 的**——healthy 只会用传进去的目录判定失败节点，不传 `check` 目录就不会因为 `check` FAIL 把这些节点刷掉：
  ```bash
  # 不要：$P healthy <check目录> <rccl目录>
  # 只传 rccl：
  $P healthy $R/envcheck/per-node-container_rccl_stages_<时间戳>_<id>
  ```
- 这样进了 `healthy_hosts` 的节点里，凡是 `check` 之前报过 FAIL 的，后面测试计划/跑模型阶段**只给它排单机（1 节点）的用例**，不要编进任何多机（≥2 节点）的测试组或训练任务——多机用例仍然必须排除这些节点，等网络/运维侧修好 RoCE/RDMA 链路、`check` 重跑也过了再放开。
- 如果 `check` 和 `rccl` 都过，才是真正不受限的多机可用节点，正常把两个目录都传给 `$P healthy`。

## screen 的判定

| idle 退出码 / 节点日志 | 结论 |
|---|---|
| `0`，`[IDLE]` | 可 ssh，空闲 → 进 `idle_hosts` |
| `[BUSY]` + `[OWNER] container=<名>` | 被容器占用，记容器名和卡 |
| `[BUSY]`、无 OWNER、有 `PID:` | 被宿主机进程占用，记 PID（`--release` 不处理宿主机进程） |
| `255` / `Permission denied` / `Connection timed out` | ssh 不通：自动 ssh 复核一次，仍不通则不进 `hostfile` |
| 其他 | 检测失败，不当作空闲 |

- 中断的 idle（`results.tsv` 只有表头）会被跳过，不会覆盖节点列表。
- `hostfile.all` 永不修改；节点恢复后下次 idle 会重新纳入。

## 处理原则

- **AI 自己的操作也要写日志**：执行命令用 `benchlog.py exec`（自动写前后两条），改文件、排查、按用户决定处理等非命令操作用 `begin` + `note`，执行前后各一条；写法见主 skill `skills/SKILL.md`「AI 操作也要写日志」。本步常见的 AI 操作：手动 ssh 复核节点、`ps -fp <PID>` 查占用、`ssh-keygen -R`、改 `hostfile.all` / `site.json`、经同意后释放卡。
  ```bash
  python3 skills/common/benchlog.py exec --site $R/site.json --target "查 bw67 卡占用" --plan "screen 报宿主机进程占用，确认归属" -- 'ssh bw67 ps -fp 4172449'
  ```
- 出错先看 `bench_log.md` 最新一条和明细日志，不要盲目重试。
- ssh 不通：报告节点和原始报错，建议 `ssh-copy-id` 或找管理员。
- 被占用：只标注、汇报；不要 `idle --release` 停别人的容器，除非用户明确要求。
- 建容器：hcu-envcheck 预检只查 docker 可用、镜像能拉取且各节点镜像 ID 一致、同名容器不存在、镜像带 ssh 工具、端口未占用，**不查卡是否空闲**，所以被占用的节点照常建。任一节点预检失败就不会在任何节点建容器：ssh 不通的先重跑 idle，把它从 `hostfile` 去掉；同名容器已存在的改用 `recreate --yes`（先征得同意）；端口被占的报告给用户。
- `DENIED: Mounting ... is not allowed`：挂载目录不被允许，改 `container.mounts` 为允许的子目录。
- `check`/`rccl` 的 `summary.md` 显示「未完成」（INCOMPLETE）但对应 `group-00N.log` 原始输出明明打印「执行结果：执行成功（PASS）/失败（FAIL），退出码 0」「任务清理：已确认」：这是 hcu-envcheck 已知过的报告聚合 bug（已修复，见 `docs/KNOWN_LIMITATIONS.md`「报告聚合（分组最后一项）」）；如果还是遇到类似情况，先看 `group-00N.log` 原始输出，必要时用 `hcu-cluster-run ... --report-only <run_dir>` 重建一次报告再读，不要直接信刚跑完、没 rebuild 过的 `summary.md`。
- **节点间通信（RCCL/RDMA/RoCE）有问题，但节点内通信没问题**（常见表现：`check` 里 `ROCE_LINK:*`/`ROCE_LEAVES:*` 报某些端口 `DOWN`，多机 collective 跨节点失败但单机内测试能跑）：这类问题是网络/硬件层面的，不是 skill 能修的，不要去改框架脚本绕过检查，`check`/`rccl`/`rccl2` 照常跑、照常让它们反映真实结果（哪怕因此 FAIL）；只跑单机用例、`healthy` 只传 rccl 目录的处理方式见上面「check 不过、单机 rccl 过」；多机用例（RCCL 跨节点、多机训练）仍然必须排除这些节点，等网络/运维侧修好 RoCE/RDMA 链路后重跑 `check`/`rccl2` 确认。真要跑多机模型前，建议先 `$P rccl2` 过一遍，比直接跑多机训练失败后再查 `root@<node>: Permission denied`/`PRTE has lost communication` 这类报错更快定位到具体是哪一对节点的问题。
- 细节见 `frameworks/cluster-manager-das/hcu-envcheck/docs/USER_GUIDE.md`。
