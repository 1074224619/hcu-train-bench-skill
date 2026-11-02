#!/usr/bin/env python3
"""Turn parsed results into dcu-train-cookbook perf-benchmarks rows.

Usage: export_cookbook.py <run_dir> [--apply] [--case ID] [--stat median|best]

Default is a preview: writes <run_dir>/cookbook/ (rows per model doc, renamed
logs, CSV rows) and prints what would change. --apply copies logs into
results/<dir>/<platform>/logs/, inserts rows before the first </tbody> of the
model doc (the full-param pretrain/sft table) and appends all_experiments.csv.
Git commit/push is left to the operator.
"""
import argparse
import csv
import io
import re
import shutil
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import benchlog
from common import FRAMEWORK_SHORT, abspath_literal, load_json, load_run

RESULT_DIR = {"megatron-lm": "megatron", "megatron-bridge": "megatron", "ms-swift": "ms-swift"}
CSV_FIELDS = ["date", "run_id", "case", "platform", "framework", "model", "stage", "precision", "nodes",
              "gpus", "mbs", "gbs", "seq_len", "tp", "cp", "dp", "pp", "sp", "vpp", "ep", "etp",
              "dispatcher", "recompute", "tflops", "time_s", "tgs", "stat", "log"]


# Family prefixes as written in the cookbook model column; longest first.
FAMILY = [
    ("qwen35_vl", "Qwen3.5-VL"), ("qwen3_vl", "Qwen3-VL"), ("qwen25_vl", "Qwen2.5-VL"),
    ("qwen35", "Qwen3.5"), ("qwen38", "Qwen3.8"), ("qwen25", "Qwen2.5"), ("qwen1.5", "Qwen1.5"),
    ("qwen3", "Qwen3"), ("qwen2", "Qwen2"), ("llama2", "llama2"), ("llama3", "llama3"),
    ("deepseek_v4", "DeepSeek-V4"), ("deepseekv3", "DeekSeek V3"), ("glm52", "GLM-5.2"),
    ("glm53", "GLM-5.3"), ("glm5", "GLM-5"), ("minimax_m3", "MiniMax-M3"), ("mixtral", "Mixtral"),
]


def display_name(rec):
    """Cookbook model-column name, e.g. qwen35_text_9b -> Qwen3.5-9B. Override with catalog display_name."""
    if rec.get("display_name"):
        return rec["display_name"]
    model = re.sub(r"_(mock|test|text|ultraep)(?=_|$)", "", rec["model"])
    head, rest = model, ""
    for key, name in FAMILY:
        if model.startswith(key):
            head, rest = name, model[len(key):].strip("_")
            break
    parts = [p.upper() if re.fullmatch(r"\d+(\.\d+)?[bt]|a\d+b|e\d+b|\d+x\d+b", p, re.I) else p.capitalize()
             for p in rest.split("_") if p]
    name = "-".join([head] + parts)
    return name + ("(sft)" if rec.get("stage") == "sft" else "")


def log_name(rec):
    par = "".join(f"{k}{rec[k]}" for k in ("tp", "pp", "ep", "cp") if rec.get(k))
    mbs = f"-mbs{rec['mbs']}" if rec.get("mbs") and rec["mbs"] != 1 else ""
    return (f"{display_name(rec)}-dcu{rec['gpus']}-b{rec['gbs']}-s{rec['seq_len']}{mbs}"
            f"-{par}-{FRAMEWORK_SHORT[rec['framework']]}.log")


def cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else ""
    return str(v)


def html_row(rec, stat):
    sfx = "_best" if stat == "best" else ""
    dp = rec.get("dp") or (rec["gpus"] // max(1, (rec.get("tp") or 1) * (rec.get("pp") or 1) * (rec.get("cp") or 1)))
    tds = lambda vals: "".join(f"<td>{cell(v)}</td>" for v in vals)
    other = "mock data" if "_mock" in rec["model"] else ""
    return "\n".join([
        "    <tr>",
        f"      <td>{display_name(rec)}</td>",
        "      " + tds(["", rec.get("precision") or "BF16", rec["nodes"], rec["gpus"]]),  # framework col left empty
        "      " + tds([rec.get("mbs"), rec.get("gbs"), rec.get("seq_len")]),
        "      " + tds([rec.get("tp"), rec.get("cp"), dp, rec.get("pp"), rec.get("sp"), rec.get("vpp"),
                        rec.get("ep") if (rec.get("ep") or 1) > 1 else None, rec.get("etp") if (rec.get("ep") or 1) > 1 else None]),
        "      " + tds([rec.get("dispatcher") if (rec.get("ep") or 1) > 1 else None, None, rec.get("recompute"), other]),
        "      " + tds([rec.get("tflops" + sfx), rec.get("time_s" + sfx), rec.get("tgs" + sfx)]),
        "      " + tds([None] * 4),
        "      <td></td>",
        f'      <td align="center"><a href="./logs/{log_name(rec)}">log</a></td>',
        "    </tr>",
    ])


def csv_row(rec, run_id, platform, stat):
    sfx = "_best" if stat == "best" else ""
    row = {k: cell(rec.get(k)) for k in CSV_FIELDS}
    row.update(date=run_id[:8], run_id=run_id, case=rec["id"], platform=platform, model=display_name(rec),
               tflops=cell(rec.get("tflops" + sfx)), time_s=cell(rec.get("time_s" + sfx)),
               tgs=cell(rec.get("tgs" + sfx)), stat=stat, log=log_name(rec))
    return row


ROW_RE = re.compile(r"[ \t]*<tr>.*?</tr>\n?", re.S)
# Column indices within a row's <td> list (0 = model name): precision..other (2..19) identify the
# benchmark config (nodes/gpus/parallel/batch/etc); tflops/time_s/tgs (20..22) are the metrics to compare.
CONFIG_COLS = slice(2, 20)
TFLOPS_COL, TGS_COL = 20, 22


def row_cells(row):
    return [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]


def model_cell(row):
    cells = row_cells(row)
    return cells[0] if cells else None


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def is_better(new_cells, old_cells):
    """True if new_cells' tgs (preferred) or tflops beats old_cells'; True too when the old value
    can't be read as a number (nothing to lose by overwriting an unparsable/missing metric)."""
    new_tgs, old_tgs = to_float(new_cells[TGS_COL]), to_float(old_cells[TGS_COL])
    if new_tgs is not None and old_tgs is not None:
        return new_tgs > old_tgs
    new_tflops, old_tflops = to_float(new_cells[TFLOPS_COL]), to_float(old_cells[TFLOPS_COL])
    if new_tflops is not None and old_tflops is not None:
        return new_tflops > old_tflops
    return True


def place_row(text, name, row_html):
    """For the same model + same benchmark config (nodes/gpus/parallel/batch/...), only keep the
    result with the better tgs (falling back to tflops): replaces a worse existing row, or skips the
    new one if the existing result is already as good or better. Otherwise fills an all-empty
    placeholder row for `name`, else inserts after its last row, else appends before the first
    </tbody>. Returns (text, how)."""
    lo = text.index("<tbody>")
    hi = text.index("</tbody>", lo)
    rows = [m for m in ROW_RE.finditer(text, lo, hi) if model_cell(m.group(0)) == name]
    new_cells = row_cells(row_html)
    for m in rows:
        old_cells = row_cells(m.group(0))
        if old_cells[1:] == [""] * (len(old_cells) - 1):
            continue  # empty placeholder row, nothing to compare against
        if old_cells[CONFIG_COLS] == new_cells[CONFIG_COLS]:
            if not is_better(new_cells, old_cells):
                return text, (f"跳过：已有结果不差于本次（tflops={old_cells[TFLOPS_COL]} "
                               f"tgs={old_cells[TGS_COL]} vs 本次 tflops={new_cells[TFLOPS_COL]} "
                               f"tgs={new_cells[TGS_COL]}）")
            return (text[:m.start()] + row_html + "\n" + text[m.end():],
                    f"替换已有行（tflops {old_cells[TFLOPS_COL]}->{new_cells[TFLOPS_COL]}, "
                    f"tgs {old_cells[TGS_COL]}->{new_cells[TGS_COL]}）")
    for m in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", m.group(0), re.S)[1:]
        if all(not c.strip() for c in cells):
            keep = re.search(r"<td[^>]*>.*?</td>", m.group(0), re.S).group(0)  # keep modelscope link
            row = re.sub(r"<td>[^<]*</td>", keep, row_html, count=1)
            return text[:m.start()] + row + "\n" + text[m.end():], "filled placeholder"
    at = rows[-1].end() if rows else text.rindex("\n", lo, hi) + 1
    return text[:at] + row_html + "\n" + text[at:], "inserted after model rows" if rows else "appended"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--case")
    ap.add_argument("--stat", choices=["median", "best"], default="median")
    a = ap.parse_args()
    run_dir, matrix, site = load_run(abspath_literal(a.run_dir))
    platform, run_id = site["platform"], matrix["meta"]["run_id"]
    bench = Path(site["cookbook_root"]) / "docs/perf-benchmarks"
    plan = [f"读取 {run_dir / 'results.json'} 中已完成且有性能数据的用例" + (f"（只要 {a.case}）" if a.case else "")
            + f"，统计方式 {a.stat}",
            f"生成预览到 {run_dir / 'cookbook'}"]
    if a.apply:
        plan += [f"改动 cookbook 工作区 {bench}：复制日志到 results/<框架>/{platform}/logs/、在模型文档表格里填/插行、"
                 "追加 all_experiments.csv（不提交、不推送）"]
    with benchlog.Step(site["runs_dir"], "export", plan, run_dir=run_dir) as st:
        export(a, run_dir, platform, run_id, bench, st)


def export(a, run_dir, platform, run_id, bench, st):
    recs = [r for r in load_json(run_dir / "results.json") if r.get("tgs") and r["state"] == "done"]
    if a.case:
        recs = [r for r in recs if r["id"] == a.case]
    stage = run_dir / "cookbook"
    stage.mkdir(exist_ok=True)

    by_doc, csv_rows, issues, done_all = {}, [], [], []
    for rec in recs:
        if not rec.get("cookbook_doc"):
            print(f"[warn] {rec['id']}: no cookbook_doc mapping, set it in catalog.json; skipped")
            issues.append((rec["id"], "没有对应的 cookbook 文档（cookbook_doc 为空）",
                           "在 catalog.json 给该用例填 cookbook_doc，或在 scan_examples.COOKBOOK_DOCS 加映射"))
            continue
        doc = bench / "results" / RESULT_DIR[rec["framework"]] / platform / rec["cookbook_doc"]
        by_doc.setdefault(doc, []).append(rec)
        csv_rows.append(csv_row(rec, run_id, platform, a.stat))

    for doc, items in by_doc.items():
        rows_md = "\n".join(html_row(r, a.stat) for r in items)
        (stage / doc.name).write_text(rows_md + "\n")
        logs_dir = doc.parent / "logs"
        print(f"== {doc.relative_to(bench)}  (+{len(items)} rows{'' if doc.exists() else ', NEW FILE needed'})")
        for r in items:
            print(f"   {r['id']}: tflops={r.get('tflops')} time={r.get('time_s')}s tgs={r.get('tgs')} -> logs/{log_name(r)}")
        if doc.exists():
            preview_text = doc.read_text()
            for r in items:
                _, how = place_row(preview_text, display_name(r), html_row(r, a.stat))
                if how.startswith("跳过"):
                    print(f"   [{'skip' if a.apply else 'would skip'}] {r['id']}: {how}")
        if not a.apply:
            continue
        if not doc.exists():
            print(f"   [skip] create {doc.name} from a sibling template first")
            issues.append((", ".join(r["id"] for r in items), f"cookbook 中不存在 {doc.relative_to(bench)}",
                           "按同目录模板新建该文档后重跑 `--apply`"))
            continue
        logs_dir.mkdir(parents=True, exist_ok=True)
        text, done = doc.read_text(), []
        for r in items:
            text, how = place_row(text, display_name(r), html_row(r, a.stat))
            if how.startswith("跳过"):
                issues.append((r["id"], how, "确认是否需要人工用更差的结果覆盖已有记录"))
                continue
            shutil.copy2(r["log"], logs_dir / log_name(r))
            done.append(f"{r['id']}: {how}")
        doc.write_text(text)
        if done:
            print("   " + "\n   ".join(done))
        done_all += [f"{doc.relative_to(bench)}：{d}" for d in done]

    buf = io.StringIO()
    csv.DictWriter(buf, CSV_FIELDS).writerows(csv_rows)
    (stage / "all_experiments.rows.csv").write_text(buf.getvalue())
    if a.apply and csv_rows:
        csv_path = bench / "all_experiments.csv"
        new = not csv_path.exists() or csv_path.stat().st_size == 0
        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, CSV_FIELDS)
            if new:
                w.writeheader()
            w.writerows(csv_rows)
    print(f"\npreview -> {stage}" + ("" if a.apply else "   (re-run with --apply to write the cookbook)"))

    skipped = [r for r in load_json(run_dir / "results.json") if not (r.get("tgs") and r["state"] == "done")]
    for r in skipped:
        issues.append((r["id"], "没有可用性能数据，不写入", "见 parse 步骤的处理方案"))
    results = [f"可写入 {len(recs)} 条，涉及 {len(by_doc)} 个文档（统计方式 {a.stat}）"]
    if a.apply:
        results += done_all + [f"追加 all_experiments.csv {len(csv_rows)} 行"]
        results.append("下一步：在 cookbook 仓库 `git diff` 给用户确认，再建分支提交")
    else:
        results.append(f"预览模式，未改动 cookbook；预览写在 {stage}，确认后加 --apply")
    status = ("成功" if recs and not issues else "部分写入" if recs else "失败") if a.apply else "预演"
    st.end(status, results, issues, [stage])


if __name__ == "__main__":
    main()
