#!/usr/bin/env python3
"""Parse each case log into metrics/<id>.json and write <run_dir>/results.json.

Usage: parse_metrics.py <run_dir> [--warmup N] [--case ID]

Run config (seq_length, gbs, mbs, parallel sizes, precision...) is read from the
log's argument dump, falling back to the matrix values. Megatron-LM prints
"  name ..... value"; Megatron-Bridge prints a YAML dump with model:/train:
sections; ms-swift (megatron sft) prints the Megatron-LM form.

tgs = gbs * seq_length / gpus / time_per_iter. Median over iterations after
warmup is reported; best (min time) is kept alongside.
"""
import argparse
import re
import statistics
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import benchlog
from common import ITER_RE, abspath_literal, load_json, load_run, save_json

KEYS = {
    "seq_len": ["seq_length", "max_length"],
    "gbs": ["global_batch_size"],
    "mbs": ["micro_batch_size"],
    "tp": ["tensor_model_parallel_size"],
    "pp": ["pipeline_model_parallel_size"],
    "cp": ["context_parallel_size"],
    "ep": ["expert_model_parallel_size"],
    "etp": ["expert_tensor_parallel_size"],
    "vpp": ["virtual_pipeline_model_parallel_size"],
    "dp": ["data_parallel_size"],
    "sp": ["sequence_parallel"],
    "dispatcher": ["moe_token_dispatcher_type"],
    "recompute": ["recompute_granularity"],
    "bf16": ["bf16"],
    "fp8": ["fp8"],
}
# Bridge YAML: only trust these sections (tokenizer:/dataset: repeat some keys with other meanings).
YAML_SECTIONS = {"model", "train"}
NULLS = {"none", "null", ""}


def parse_config(text):
    cfg = {}
    dotted = {m.group(1): m.group(2).strip() for m in re.finditer(r"^\s+(\w+) \.{3,} (.*)$", text, re.M)}
    yaml, section = {}, None
    for line in text.splitlines():
        if re.match(r"^\w+:\s*$", line):
            section = line[:-1].strip()
        elif section in YAML_SECTIONS and (m := re.match(r"^  (\w+): (.*)$", line)):
            yaml.setdefault(m.group(1), m.group(2).strip())
        elif line and not line.startswith(" "):
            section = None
    for field, names in KEYS.items():
        for n in names:
            v = dotted.get(n, yaml.get(n))
            if v is not None:
                cfg[field] = None if v.lower() in NULLS else v
                break
    for k in ("seq_len", "gbs", "mbs", "tp", "pp", "cp", "ep", "etp", "vpp", "dp"):
        if cfg.get(k) is not None:
            try:
                cfg[k] = int(cfg[k])
            except ValueError:
                cfg[k] = None
    if "sp" in cfg:
        cfg["sp"] = str(cfg["sp"]).lower() == "true"
    fp8 = cfg.pop("fp8", None)
    bf16 = str(cfg.pop("bf16", "")).lower() == "true"
    cfg["precision"] = f"FP8({fp8})" if fp8 else ("BF16" if bf16 else None)
    return cfg


def parse_iters(text, warmup):
    rows = []
    for m in ITER_RE.finditer(text):
        it = int(m.group(1))
        if it > warmup:
            rows.append((it, float(m.group(3)), float(m.group(4)) if m.group(4) else None))
    return rows


def summarize(case, cfg, rows):
    gpus = case["gpus"]
    seq, gbs = cfg.get("seq_len") or case.get("seq_len"), cfg.get("gbs") or case.get("gbs")
    times = [r[1] / 1000 for r in rows]
    tflops = [r[2] for r in rows if r[2] is not None]
    t_med, t_best = statistics.median(times), min(times)

    def tgs(t):
        return round(gbs * seq / gpus / t) if gbs and seq else None

    return {
        "iters_used": len(rows), "last_iter": rows[-1][0],
        "time_s": round(t_med, 3), "time_s_best": round(t_best, 3),
        "tflops": round(statistics.median(tflops), 1) if tflops else None,
        "tflops_best": round(max(tflops), 1) if tflops else None,
        "tgs": tgs(t_med), "tgs_best": tgs(t_best),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--case")
    a = ap.parse_args()
    run_dir, matrix, site = load_run(abspath_literal(a.run_dir))
    warmup = a.warmup if a.warmup is not None else site.get("warmup_iters", 5)
    n = sum(1 for c in matrix["cases"] if not a.case or c["id"] == a.case)
    plan = [f"解析 {n} 个用例的日志 {run_dir / 'logs'}（跳过前 {warmup} 步，取中位数，同时保留 best）",
            "写 metrics/<id>.json" + ("" if a.case else f" 和 {run_dir / 'results.json'}")]
    with benchlog.Step(site["runs_dir"], "parse", plan, run_dir=run_dir) as st:
        log_parse(st, run_dir, parse_all(a, run_dir, matrix, warmup), warmup)


def parse_all(a, run_dir, matrix, warmup):
    results = []
    for case in matrix["cases"]:
        cid = case["id"]
        if a.case and cid != a.case:
            continue
        log = run_dir / "logs" / f"{cid}.log"
        status_p = run_dir / "status" / f"{cid}.json"
        status = load_json(status_p) if status_p.exists() else {"state": "not_run"}
        rec = {"id": cid, "state": status["state"], "log": str(log)}
        if log.exists():
            text = log.read_text(errors="ignore")
            cfg = parse_config(text)
            rows = parse_iters(text, warmup)
            par = {k: cfg.get(k) or case.get("parallel", {}).get(k) for k in ("tp", "pp", "cp", "ep", "etp")}
            rec.update({k: case.get(k) for k in ("framework", "model", "stage", "nodes", "gpus", "cookbook_doc")})
            rec.update(cfg, **par)
            rec["seq_len"] = cfg.get("seq_len") or case.get("seq_len")
            rec["gbs"] = cfg.get("gbs") or case.get("gbs")
            rec["mbs"] = cfg.get("mbs") or case.get("mbs")
            if rows:
                rec.update(summarize(case, cfg, rows))
            else:
                rec["error"] = f"no iteration lines after warmup={warmup}"
        else:
            rec["error"] = "log missing"
        save_json(run_dir / "metrics" / f"{cid}.json", rec)
        results.append(rec)
        print(f"{cid:<50} {rec['state']:<8} tflops={rec.get('tflops')} time={rec.get('time_s')}s "
              f"tgs={rec.get('tgs')} {rec.get('error', '')}")

    if not a.case:
        save_json(run_dir / "results.json", results)
        print(f"-> {run_dir / 'results.json'}")
    return results


def log_parse(st, run_dir, results, warmup):
    ok = [r for r in results if r.get("tgs")]
    results_lines = [f"解析 {len(results)} 个用例，得到性能数据 {len(ok)} 个（warmup={warmup}，取中位数）"]
    results_lines += [f"{r['id']}：tflops={r.get('tflops')} time={r.get('time_s')}s tgs={r.get('tgs')}" for r in ok]
    issues = []
    for r in results:
        if r.get("tgs"):
            continue
        if r.get("error") == "log missing":
            issues.append((r["id"], "没有日志（用例未运行）", "执行 `run_cases.py <run_dir> --case " + r["id"] + "`"))
        elif r["state"] != "done":
            issues.append((r["id"], f"用例状态 {r['state']}，没有可用的迭代数据", "先看 run 步骤里该用例的问题与处理方案"))
        else:
            issues.append((r["id"], r.get("error", "无性能数据"),
                           f"迭代数不超过 warmup 时用 `--warmup` 调小；或检查日志里的 iteration 行格式"))
    st.end(benchlog.status_of(len(ok), len(results) - len(ok)), results_lines, issues, [run_dir / "results.json"])


if __name__ == "__main__":
    main()
