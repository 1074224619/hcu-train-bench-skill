---
name: hcu-test-plan
description: 扫描 Megatron-LM-das / Megatron-bridge-das / ms-swift 的 example 生成用例目录 catalog.json，并按可用节点数生成测试列表（matrix）。
when_to_use: 要知道有哪些模型/框架能测、每个用例占几张卡，或要根据当前空闲节点排一份测试列表时使用。
---

# hcu-test-plan（步骤 2）

以下命令假定在项目根目录下执行（`K`/`R` 换成绝对路径也一样）：

```bash
K=skills/hcu-test-plan
R=runs
```

## 扫描用例

```bash
python3 $K/scripts/scan_examples.py frameworks -o $K/catalog.json --site $R/site.json
```

`catalog.json` 每个用例：id、框架、model、stage、launcher 和参数/环境变量、节点数、并行参数、mbs/gbs/seq_len、cookbook_doc、enabled、notes。扫描后人工审一遍：

- `enabled`：Megatron-LM 的 launcher 里 `DATA_PATH`/`TOKENIZER_MODEL_PATH` 没默认值（`""` 或 `${VAR:-}`）、或 launcher 没选中该模型时默认关闭，原因在 `notes`；这类用例不用改框架脚本，在 `site.json` 的 `models.<model>` 填上 `data_path`/`tokenizer_path` 就行，见下面「数据集/模型路径」。另外，不在 `site.json` 的 `frameworks_enabled` 里的框架也会被标成关闭，见下面「当前只跑哪些框架」。
- `parallel` / `nodes` 取自脚本默认值；bridge recipe 的真实切分以运行日志为准。
- `cookbook_doc`、可选 `display_name` 决定写入 cookbook 的文件和模型名。
- 重新扫描会覆盖 catalog，有人工改动先备份。
- 审 catalog 时改 `enabled` / `nodes` / `cookbook_doc`、补 site.json 路径、备份 catalog，都是 AI 操作，每改一处前后各写一条日志（`--target` 写用例 id 和字段，`--result` 写改成了什么）。

**AI 自己的操作也要写日志**：执行命令用 `benchlog.py exec`（自动写前后两条），改文件、排查、按用户决定处理等非命令操作用 `begin` + `note`，执行前后各一条；写法见主 skill `skills/SKILL.md`「AI 操作也要写日志」。

## 当前只跑哪些框架（site.json 的 frameworks_enabled）

`site.json` 可选字段 `frameworks_enabled`（如 `["megatron-lm"]`）限制现阶段只跑哪些框架；不写这个字段 = 三个框架都不限制。不在列表里的框架，`scan_examples.py` 会把它的用例标成 `enabled: false`（notes 写明原因），`build_matrix.py` 对同一限制再过滤一遍（即使拿到一份没按这份 site.json 扫描过的 `catalog.json` 也不会漏过滤）。要开 bridge/swift 时把名字加进数组即可：已有 `catalog.json` 直接重新 `build_matrix.py` 就会放开，不用重新 `scan_examples.py`。

## 数据集/模型路径（site.json 的 models）

`site.json` 的 `models.<model>` 是按 catalog 的 model 名（全小写，见 `scan_examples.py` 的命名规则，如 `train_qwen3_8B.sh` -> `qwen3_8b`）预先填好的数据集/模型/tokenizer 路径，字段见 `common/site.example.json` 的示例：

| 框架 | 字段 | 对应 launcher 里的环境变量 |
|---|---|---|
| megatron-lm | `data_path` / `tokenizer_path` | `DATA_PATH` / `TOKENIZER_MODEL_PATH` |
| megatron-bridge | `data_path` / `model_path` | `DATA_PATH` / `LOCAL_HF_PATH` |

`build_matrix.py` 生成测试列表时会自动把这些值按上表注入每个用例的 `env`（不用手改 `catalog.json` 或各框架的 `run_*.sh`）；megatron-lm 的用例如果本来是因为缺 `DATA_PATH`/`TOKENIZER_MODEL_PATH` 被 `enabled=false`，填了对应的 `models.<model>` 后重新 `build_matrix.py` 会自动变成启用（不用去改 `catalog.json` 的 `enabled`）。这依赖 Megatron-LM-das 的 `run_*.sh` 已经把 `DTK_ENV`/`DATA_PATH`/`TOKENIZER_MODEL_PATH`/`NCCL_ENV` 改成了和 `TRAIN_SCRIPT` 一样的 `${VAR:-默认值}` 可覆盖写法；如果后续改动了框架仓库但没保留这个写法，这条链路会失效，`enabled` 还是会被扫描判定关闭。

填 `models.<model>` 之前先确认对应数据集/模型目录已经存在、且在 `container.mounts` 内（否则容器里读不到），手动填 `site.json` 即可——比笼统的根目录自动猜测更精确。

```bash
python3 $K/scripts/build_matrix.py --site $R/site.json --catalog $K/catalog.json \
    --hosts $R/healthy_hosts [--only 'qwen3|glm'] [--max-nodes 4]
```

保留已启用且节点数 ≤ 可用节点的用例，按节点数降序装箱成 wave：同一 wave 内用例占用互不重叠的节点并发执行。写 `$R/<run_id>/matrix.json` 和 `matrix.md`（含「未纳入」及原因）。**把 `matrix.md` 给用户确认后再交给 hcu-run-model 执行。** 用户确认或要求增删用例也记一条（`--plan` 写用户的决定）。

## 扩展

- 新框架：在 `scan_examples.py` 加 `scan_<fw>()`；在 `common/common.py` 的 `FRAMEWORK_SHORT` 加缩写。
- 新模型文档映射：`scan_examples.COOKBOOK_DOCS`。
