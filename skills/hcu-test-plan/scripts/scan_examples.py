#!/usr/bin/env python3
"""Scan the three frameworks' example scripts and write a candidate test catalog.

Usage: scan_examples.py <frameworks_root> [-o catalog.json] [--gpus-per-node 8]

The catalog is a starting point: review it, fix `nodes`, add model/data paths in
`env`, and set `enabled` before building a matrix. Values come from the script
defaults; the authoritative parallel config of a run is read back from its log.
"""
import argparse
import re
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import benchlog
from common import FRAMEWORK_SHORT, abspath_literal, load_json, min_world_size, save_json

ASSIGN_RE = r"^\s*{name}=\"?\$?\{{?(?:[A-Z_]+:-)?(\d+)\}}?\"?\s*(?:#.*)?$"
COOKBOOK_DOCS = [
    (r"^qwen1\.5", "Qwen-1.5.md"), (r"^qwen2(?!5)", "Qwen-2.md"), (r"^qwen25", "Qwen-2.md"),
    (r"^qwen35_vl", "Qwen-3.5-VL.md"), (r"^qwen35", "Qwen-3.5.md"), (r"^qwen38", "Qwen-3.8.md"),
    (r"^qwen3_vl", "Qwen-3-VL.md"), (r"^qwen3", "Qwen-3.md"),
    (r"^llama", "Llama.md"), (r"^deepseekv4|^deepseek_v4", "DeepSeek-4.md"),
    (r"^deepseekv3|^deepseek_v3", "DeepSeek-3.md"),
    (r"^glm53", "GLM-5.3.md"), (r"^glm52|^glm5_", "GLM-5.2.md"), (r"^minimax", "MiniMax-M3.md"),
]


def grab(text, name):
    m = re.search(ASSIGN_RE.format(name=name), text, re.M)
    return int(m.group(1)) if m else None


def grab_flag(text, flag):
    m = re.search(rf"--{flag}\s+(\d+)", text)
    return int(m.group(1)) if m else None


def cookbook_doc(model):
    for pattern, doc in COOKBOOK_DOCS:
        if re.search(pattern, model):
            return doc
    return None


def parallel_of(text, getter):
    keys = {"tp": "TP", "pp": "PP", "cp": "CP", "ep": "EP", "etp": "ETP"}
    return {k: getter(text, v) or 1 for k, v in keys.items()}


def case(framework, model, stage, launcher, args, env, par, gpn, extra, notes, nodes_hint=None):
    world = min_world_size(**par)
    nodes = nodes_hint or max(1, -(-world // gpn))
    return {
        "id": f"{FRAMEWORK_SHORT[framework]}-{model}-{stage}-{nodes * gpn}g",
        "enabled": not notes,
        "framework": framework, "model": model, "stage": stage,
        "launcher": launcher, "args": args, "env": env,
        "nodes": nodes, "min_world": world, "parallel": par, **extra,
        "cookbook_doc": cookbook_doc(model), "notes": notes,
    }


def scan_megatron_lm(root, gpn):
    cases, base = [], root / "Megatron-LM-das/examples"
    for train in sorted(base.glob("*/train_*.sh")):
        text = train.read_text(errors="ignore")
        launchers = sorted(train.parent.glob("run_*.sh"))
        launcher = next((l for l in launchers if train.name in l.read_text(errors="ignore")), None)
        env, notes = {}, []
        overridable = [l for l in launchers if "TRAIN_SCRIPT:-" in l.read_text(errors="ignore")]
        if launcher is None and overridable:
            launcher = overridable[0]
        if launcher is None:
            notes.append("没有 launcher 引用这个训练脚本")
            continue
        if "TRAIN_SCRIPT:-" in launcher.read_text(errors="ignore"):
            env["TRAIN_SCRIPT"] = train.name
        elif train.name not in launcher.read_text(errors="ignore"):
            notes.append(f"{launcher.name} 没有选中 {train.name}，需修改 launcher")
        ltext = launcher.read_text(errors="ignore")
        # Both the old literal-empty form (DATA_PATH="") and the now env-overridable
        # form (DATA_PATH="${DATA_PATH:-}") mean "no default baked into the script";
        # build_matrix later fills these from site.json's `models` preset when present.
        if (re.search(r'^DATA_PATH="(?:\$\{DATA_PATH:-\})?"', ltext, re.M)
                or re.search(r'^TOKENIZER_MODEL_PATH="(?:\$\{TOKENIZER_MODEL_PATH:-\})?"', ltext, re.M)):
            notes.append(f"需在 site.json 的 models.<model> 或 {launcher.name} 中设置 DATA_PATH/TOKENIZER_MODEL_PATH")
        m = re.search(r"_(\d+)nodes", train.stem)
        model = re.sub(r"_\d+nodes$", "", train.stem.replace("train_", "")).lower()
        extra = {"mbs": grab(text, "MBS"), "gbs": grab(text, "GBS"), "seq_len": grab(text, "SEQ_LEN")}
        cases.append(case(
            "megatron-lm", model, "pretrain", str(launcher.relative_to(root)),
            ["{hostfile}", "{nodes}"], env, parallel_of(text, grab), gpn, extra, notes,
            int(m.group(1)) if m else None))
    return cases


def scan_megatron_bridge(root, gpn):
    cases, base = [], root / "Megatron-bridge-das/examples"
    for train in sorted(base.glob("*/train_*.sh")):
        text = train.read_text(errors="ignore")
        recipes = sorted(set(re.findall(r"\b([a-z0-9_]+_(?:pretrain|sft)_\d+gpu_[A-Za-z0-9]+_[a-z0-9_]*config)\b", text)))
        if not recipes:
            cases.append({
                "id": f"bridge-{train.stem}", "enabled": False, "framework": "megatron-bridge",
                "launcher": str(train.relative_to(root)),
                "notes": ["脚本中未找到 recipe 列表，需手动添加用例"],
            })
            continue
        for recipe in recipes:
            gpus = int(re.search(r"_(\d+)gpu_", recipe).group(1))
            model, stage = re.match(r"(.+?)_(pretrain|sft)_", recipe).groups()
            suffix = "".join(f"_{t}" for t in ("mock", "test") if f"_{t}_" in recipe + "_")
            notes = ["使用 mock 数据的 recipe"] if "mock" in recipe else []
            c = case(
                "megatron-bridge", model + suffix, stage,
                str(train.relative_to(root)), [recipe, "train", "{hostfile}", "{nodes}"],
                {"GPUS": "{gpus_per_node}", "PORT": "{ssh_port}"},
                parallel_of(text, grab), gpn, {"recipe": recipe, "seq_len": grab(text, "SEQ_LENGTH")},
                [], max(1, gpus // gpn))
            c["notes"] = notes + ["并行参数取自脚本默认值，需按 recipe 核对"]
            c["enabled"] = True
            cases.append(c)
    return cases


def scan_ms_swift(root, gpn):
    cases, base = [], root / "ms-swift"
    launcher = base / "run_qwen.sh"
    for train in sorted(base.glob("train_*.sh")):
        text = train.read_text(errors="ignore")
        par = {
            "tp": grab_flag(text, "tensor_model_parallel_size") or 1,
            "pp": grab_flag(text, "pipeline_model_parallel_size") or 1,
            "cp": grab_flag(text, "context_parallel_size") or 1,
            "ep": grab_flag(text, "expert_model_parallel_size") or 1,
            "etp": grab_flag(text, "expert_tensor_parallel_size") or 1,
        }
        stage = "sft" if "megatron sft" in text else "pretrain"
        extra = {"mbs": grab_flag(text, "micro_batch_size"), "gbs": grab_flag(text, "global_batch_size"),
                 "seq_len": grab_flag(text, "max_length")}
        notes = []
        if launcher.exists() and train.name not in launcher.read_text(errors="ignore").split("#")[0]:
            notes.append(f"run_qwen.sh 的 'script=' 没有选中 {train.name}")
        notes.append("run_qwen.sh 把日志写到写死的 /data/home/... 路径，运行前需修改")
        cases.append(case("ms-swift", train.stem.replace("train_", ""), stage,
                          str(launcher.relative_to(root)), ["{hostfile}", "{nodes}"], {},
                          par, gpn, extra, notes))
    return cases


def apply_frameworks_enabled(cases, site):
    """If site.json sets `frameworks_enabled` (a subset of megatron-lm/megatron-bridge/
    ms-swift), disable any case whose framework isn't in it -- used to stage rollout
    (e.g. only megatron-lm for now, bridge/swift later) without touching the framework
    scripts. Missing the key means no restriction (all frameworks as scanned)."""
    allowed = site.get("frameworks_enabled") if site else None
    if allowed is None:
        return cases
    for c in cases:
        if c["framework"] not in allowed and c.get("enabled"):
            c["enabled"] = False
            c["notes"].append(f"当前阶段暂停：site.json 的 frameworks_enabled 未包含 {c['framework']}（后面要跑时把它加进数组，重新 scan 或直接 build_matrix 即可）")
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frameworks_root", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("catalog.json"))
    ap.add_argument("--gpus-per-node", type=int, default=8)
    ap.add_argument("--site", type=Path, help="site.json; 给出时把扫描结果写入 <runs_dir>/bench_log.md，并按 frameworks_enabled 过滤")
    a = ap.parse_args()
    root, gpn = abspath_literal(a.frameworks_root), a.gpus_per_node
    site = load_json(a.site) if a.site else None
    plan = [f"扫描 {root} 下 Megatron-LM-das / Megatron-bridge-das / ms-swift 的 example 脚本（每节点 {gpn} 卡）",
            f"写用例目录 {a.output}（覆盖已有文件）"]
    with benchlog.Step(site["runs_dir"] if site else None, "scan", plan) as st:
        cases = scan_megatron_lm(root, gpn) + scan_megatron_bridge(root, gpn) + scan_ms_swift(root, gpn)
        cases = apply_frameworks_enabled(cases, site)
        save_json(a.output, {"frameworks_root": str(root), "gpus_per_node": gpn, "cases": cases})
        n_on = sum(1 for c in cases if c.get("enabled"))
        print(f"共 {len(cases)} 个用例（启用 {n_on}，关闭 {len(cases) - n_on}），已写入 {a.output}")
        for c in cases:
            flag = "启用" if c.get("enabled") else "关闭"
            print(f"  [{flag}] {c['id']:<55} 节点数={c.get('nodes', '?')}  {'; '.join(c.get('notes', []))}")
        log_scan(st, a, cases, n_on)


def log_scan(st, a, cases, n_on):
    by_fw = {}
    for c in cases:
        fw = by_fw.setdefault(c["framework"], [0, 0])
        fw[0 if c.get("enabled") else 1] += 1
    results = [f"共 {len(cases)} 个用例，启用 {n_on}，关闭 {len(cases) - n_on}，写入 {a.output}"]
    results += [f"{fw}：启用 {on}，关闭 {off}" for fw, (on, off) in by_fw.items()]
    issues, groups = [], {}
    for c in cases:
        if not c.get("enabled"):
            groups.setdefault("；".join(c.get("notes", [])) or "未注明", []).append(c["id"])
    for why, ids in groups.items():
        fix = ("在 site.json 的 frameworks_enabled 加上该框架后重新 scan 或直接 build_matrix" if "frameworks_enabled" in why else
               "在 site.json 填写数据集/模型/tokenizer 路径" if "PATH" in why else
               "改成可传参后重新扫描" if "run_qwen.sh" in why else "按原因修改脚本或在 catalog.json 中手动启用")
        issues.append((f"{len(ids)} 个：{', '.join(ids[:4])}{' …' if len(ids) > 4 else ''}", f"已关闭：{why}", fix))
    st.end("成功" if not issues else "部分关闭", results, issues, [a.output])


if __name__ == "__main__":
    main()
