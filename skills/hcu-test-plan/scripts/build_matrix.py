#!/usr/bin/env python3
"""Build a runnable test matrix from the catalog and the healthy-node hostfile.

Usage: build_matrix.py --site site.json --catalog catalog.json [--hosts healthy_hosts]
                       [--only REGEX] [--max-nodes N] [-o <run_dir>]

Keeps enabled cases whose node count fits, then packs them into waves: within a
wave every case gets a disjoint host slice so cases run concurrently on
different nodes. Writes <run_dir>/matrix.json and <run_dir>/matrix.md.
"""
import argparse
import re
import time
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import benchlog
from common import load_json, read_hosts, save_json, abspath_literal


def inject_model_paths(c, site):
    """Fill a case's env from site.json's `models.<model>` preset: DATA_PATH /
    TOKENIZER_MODEL_PATH for megatron-lm, DATA_PATH / LOCAL_HF_PATH (LOCAL_HF_PATH
    is megatron-bridge's own env-overridable var for the HF model dir) for
    megatron-bridge. A megatron-lm case the scanner only disabled for a missing
    path (scan_examples.py's "需在 site.json 的 models.<model> 或 ... 中设置
    DATA_PATH/TOKENIZER_MODEL_PATH" note) is lifted back to enabled once the
    preset covers both; megatron-bridge cases are enabled by default already, so
    only their env gets filled, not `enabled`/`notes`.

    Every megatron-lm case also gets NCCL_ENV forced to site.json's env_script,
    regardless of whether a models preset exists: the launchers' own default
    (framework's requirements/env.sh) makes PyTorch's NCCL init fail with
    "Attribute arch of node cpu not found" on this cluster even though rccl-tests
    passes fine (see QA.md 3.2).

    Every megatron-lm case also gets DTK_ENV forced to /opt/dtk/env.sh: every
    launcher except qwen3's run_qwen.sh defaults DTK_ENV to an empty string
    (`${DTK_ENV:-}`), so `source ${DTK_ENV}` fails with "source: filename
    argument required" and the job never starts (confirmed on llama2_13b)."""
    if c["framework"] == "megatron-lm" and site.get("env_script"):
        env = dict(c.get("env", {}))
        env["NCCL_ENV"] = site["env_script"]
        c = dict(c, env=env)
    if c["framework"] == "megatron-lm":
        env = dict(c.get("env", {}))
        env.setdefault("DTK_ENV", "/opt/dtk/env.sh")
        c = dict(c, env=env)

    preset = site.get("models", {}).get(c.get("model"), {})
    if not preset:
        return c
    env = dict(c.get("env", {}))
    if c["framework"] == "megatron-lm":
        if preset.get("data_path"):
            env["DATA_PATH"] = preset["data_path"]
        if preset.get("tokenizer_path"):
            env["TOKENIZER_MODEL_PATH"] = preset["tokenizer_path"]
        c = dict(c, env=env)
        if "DATA_PATH" in env and "TOKENIZER_MODEL_PATH" in env:
            path_note = next((n for n in c.get("notes", []) if "DATA_PATH" in n), None)
            if path_note:
                c["notes"] = [n for n in c["notes"] if n != path_note]
                c["enabled"] = not c["notes"]
    elif c["framework"] == "megatron-bridge":
        if preset.get("model_path"):
            env["LOCAL_HF_PATH"] = preset["model_path"]
        if preset.get("data_path"):
            env["DATA_PATH"] = preset["data_path"]
        c = dict(c, env=env)
    return c


def pack_waves(cases, hosts):
    """First-fit-decreasing by node count; returns [[(case, host_slice), ...], ...]."""
    waves = []
    for c in sorted(cases, key=lambda c: -c["nodes"]):
        for wave in waves:
            if wave["free"] >= c["nodes"]:
                break
        else:
            wave = {"free": len(hosts), "items": []}
            waves.append(wave)
        start = len(hosts) - wave["free"]
        wave["items"].append((c, hosts[start:start + c["nodes"]]))
        wave["free"] -= c["nodes"]
    return [w["items"] for w in waves]


def render_md(matrix, skipped):
    lines = [
        f"# Test matrix {matrix['meta']['run_id']}", "",
        f"可用节点 {matrix['meta']['hosts_total']}，每节点 {matrix['meta']['gpus_per_node']} 卡，平台 {matrix['meta']['platform']}", "",
        "| wave | case | 框架 | 模型 | 阶段 | 节点 | 卡数 | tp/pp/cp/ep/etp | hosts |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in matrix["cases"]:
        p = c.get("parallel", {})
        par = "/".join(str(p.get(k, "")) for k in ("tp", "pp", "cp", "ep", "etp"))
        lines.append(f"| {c['wave']} | {c['id']} | {c['framework']} | {c['model']} | {c['stage']} | "
                     f"{c['nodes']} | {c['gpus']} | {par} | {','.join(c['hosts'])} |")
    if skipped:
        lines += ["", "## 未纳入", "", "| case | 原因 |", "|---|---|"]
        lines += [f"| {cid} | {why} |" for cid, why in skipped]
    return "\n".join(lines) + "\n"


def log_matrix(st, a, site, hosts, cases, skipped):
    waves = sorted({c["wave"] for c in cases})
    results = [f"可用节点 {len(hosts)} 个（{a.hosts or site['hostfile']}）：{', '.join(hosts)}",
               f"纳入 {len(cases)} 个用例，分 {len(waves)} 个 wave；未纳入 {len(skipped)} 个"]
    results += [f"wave {w}：" + "，".join(f"{c['id']}@{'+'.join(c['hosts'])}" for c in cases if c["wave"] == w)
                for w in waves]
    groups = {}
    for cid, why in skipped:
        key = "节点不足" if why.startswith("needs") else why
        groups.setdefault(key, []).append(cid)
    issues = []
    for why, ids in groups.items():
        fix = ("等更多空闲节点后重新 idle 并生成列表，或用 --max-nodes 明确只跑小规模" if why == "节点不足"
               else "见 scan 步骤的处理方案")
        issues.append((f"{len(ids)} 个：{', '.join(ids[:4])}{' …' if len(ids) > 4 else ''}", f"未纳入：{why}", fix))
    if not cases:
        issues.append(("全部", "没有可执行的用例", "检查 healthy_hosts 是否为空、catalog 是否有启用的用例"))
    st.end("成功" if cases else "失败", results, issues, [st.run_dir / "matrix.md"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--hosts", type=Path, help="healthy hosts (default: site hostfile)")
    ap.add_argument("--only", help="regex on case id")
    ap.add_argument("--max-nodes", type=int, help="cap nodes per case")
    ap.add_argument("-o", "--run-dir", type=Path)
    a = ap.parse_args()

    site = load_json(a.site)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = abspath_literal(a.run_dir or Path(site["runs_dir"]) / run_id)
    plan = [f"读取用例目录 {a.catalog} 和可用节点 {a.hosts or site['hostfile']}",
            "保留已启用且节点数不超过可用节点的用例" + (f"（只要 id 匹配 `{a.only}`）" if a.only else "")
            + (f"，每个用例最多 {a.max_nodes} 节点" if a.max_nodes else ""),
            "按节点数降序装箱成 wave（同一 wave 内节点互不重叠）",
            f"写 {run_dir / 'matrix.json'} 和 matrix.md"]
    with benchlog.Step(site["runs_dir"], "matrix", plan, run_dir=run_dir) as st:
        build(a, site, run_dir, st)


def build(a, site, run_dir, st):
    catalog = load_json(a.catalog)
    hosts = read_hosts(a.hosts or site["hostfile"])
    gpn = site["gpus_per_node"]
    limit = min(len(hosts), a.max_nodes or len(hosts))

    keep, skipped = [], []
    allowed_fw = site.get("frameworks_enabled")
    for c in catalog["cases"]:
        if a.only and not re.search(a.only, c["id"]):
            continue
        if allowed_fw is not None and c["framework"] not in allowed_fw:
            skipped.append((c["id"], f"framework disabled: {c['framework']} not in site.json frameworks_enabled"))
            continue
        c = inject_model_paths(dict(c), site)
        if not c.get("enabled"):
            skipped.append((c["id"], "disabled: " + "; ".join(c.get("notes", []))))
        elif c["nodes"] > limit:
            skipped.append((c["id"], f"needs {c['nodes']} nodes > {limit} available"))
        else:
            keep.append(c)

    cases = []
    for w, wave in enumerate(pack_waves(keep, hosts[:limit]) if keep else []):
        for c, slice_ in wave:
            c.update(wave=w, hosts=slice_, gpus=c["nodes"] * gpn)
            cases.append(c)

    matrix = {
        "meta": {"run_id": run_dir.name, "site": str(abspath_literal(a.site)), "catalog": str(abspath_literal(a.catalog)),
                 "hosts_total": len(hosts), "gpus_per_node": gpn, "platform": site["platform"]},
        "cases": cases,
    }
    save_json(run_dir / "matrix.json", matrix)
    (run_dir / "matrix.md").write_text(render_md(matrix, skipped))
    log_matrix(st, a, site, hosts, cases, skipped)
    print(f"{len(cases)} cases in {len({c['wave'] for c in cases})} waves, {len(skipped)} skipped -> {run_dir}")
    print((run_dir / "matrix.md").read_text())


if __name__ == "__main__":
    main()
