#!/usr/bin/env python3
"""Classify nodes from one hcu-envcheck idle-check run.

Usage: screen_nodes.py --site site.json [--idle-dir <envcheck/idle-check_*>] [--no-ssh-recheck]

Reads <idle-dir>/results.tsv and <node>.log (default: newest idle-check run) and writes
  <runs_dir>/hostfile        nodes reachable over ssh (site.json: hostfile; containers go here)
  <runs_dir>/idle_hosts      reachable and every card idle (benchmarks only run here)
  <runs_dir>/node_status.md  one row per node with the reason
Nodes are taken from results.tsv, so hostfile.all (the full list) is never modified.
"""
import argparse
import re
import subprocess
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import benchlog
from common import load_json

SSH_ERR = re.compile(r"Permission denied|Connection (refused|timed out|closed)|Could not resolve hostname|"
                     r"No route to host|Host key verification failed|ssh: connect to host")
TAG = re.compile(r"^\[(IDLE|BUSY|CARD|OWNER|UNKNOWN|ERROR)\]\s*(.*)$")


def ssh_ok(host):
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, "true"],
                           capture_output=True, text=True, timeout=20)
        return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1]
    except subprocess.TimeoutExpired:
        return False, "ssh timeout"


def classify(host, rc, log_text, recheck):
    """Returns dict(ssh, idle, cards, owners, pids, reason)."""
    tags = {}
    cards, owners = {}, []
    pids = sorted(set(re.findall(r"^PID:\s*(\d+)", log_text, re.M)))
    for line in log_text.splitlines():
        m = TAG.match(line.strip())
        if not m:
            continue
        tag, rest = m.groups()
        tags.setdefault(tag, rest)
        if tag == "CARD":
            kv = dict(x.split("=", 1) for x in rest.split() if "=" in x)
            cards[kv.get("card")] = f"vram={kv.get('vram')} util={kv.get('util')}"
        elif tag == "OWNER":
            name = re.search(r"container=(\S+)", rest)
            if name and name.group(1) not in owners:
                owners.append(name.group(1))
        elif tag == "UNKNOWN" and "未识别" not in owners:
            owners.append("未识别")
    st = {"ssh": "通", "idle": "否", "cards": "-", "owners": "-", "pids": pids, "reason": ""}
    ssh_err = SSH_ERR.search(log_text)
    if "IDLE" in tags and rc == 0:
        st.update(idle="是", cards="0")
    elif "BUSY" in tags:
        st.update(cards=f"{len(cards)}（{'; '.join(f'卡{k} {v}' for k, v in sorted(cards.items()))}）",
                  owners=", ".join(owners) or "未识别")
    elif rc == 255 or ssh_err:
        err = (log_text[log_text.rfind("\n", 0, ssh_err.start()) + 1:].split("\n", 1)[0].strip()
               if ssh_err else f"rc={rc}")
        if recheck:
            ok, msg = ssh_ok(host)
            if ok:
                st.update(reason=f"idle 时 ssh 失败（{err}），复核已通，需重跑 idle")
                return st
            err = msg or err
        st.update(ssh="不通", idle="-", reason=err)
    else:
        st.update(reason="检测失败: " + tags.get("ERROR", f"rc={rc}，日志无结论"))
    return st


def issue_of(host, s):
    """(对象, 问题, 处理方案) for a node that is not usable, else None."""
    if s["ssh"] == "不通":
        prob, fix = benchlog.explain(s["reason"], host)
        return host, f"ssh 不通，不建容器、不跑测试；{prob}", fix
    if s["idle"] == "是":
        return None
    if s["reason"].startswith("idle 时 ssh 失败"):
        return host, s["reason"], "重跑 `prepare_nodes.sh site.json idle`"
    if s["reason"]:
        return host, s["reason"], "查看该节点的 idle-check 日志；修复后重跑 `idle`"
    n = s["cards"].split("（")[0]
    if s["owners"] not in ("-", "未识别"):
        return (host, f"{n} 张卡被容器 {s['owners']} 占用，不跑测试（仍会建容器）",
                "联系容器所有者；若是自己的旧容器，经同意后用 `idle --release` 释放，再重跑 `idle`")
    pid = ",".join(s["pids"]) or "?"
    return (host, f"{n} 张卡被宿主机进程 PID {pid} 占用（不属于任何容器），不跑测试（仍会建容器）",
            f"在节点上 `ps -fp {pid}` 确认归属；停掉后重跑 `idle`。hcu-envcheck 的 --release 不处理宿主机进程")


def log_screen(st, runs, idle_dir, rows, reach, idle):
    issues = [i for i in (issue_of(h, s) for h, s in rows) if i]
    if not idle:
        issues.append(("全部节点", "没有空闲节点，无法跑测试", "按上表处理后重跑 `idle`"))
    results = [f"共 {len(rows)} 个节点（来自 {idle_dir.name}）",
               f"可 ssh {len(reach)} 个 -> hostfile（建容器用）：{', '.join(reach) or '无'}",
               f"空闲 {len(idle)} 个 -> idle_hosts（跑测试用）：{', '.join(idle) or '无'}"]
    status = "成功" if len(idle) == len(rows) else ("失败" if not idle else "部分可用")
    st.end(status, results, issues, [idle_dir, runs / "node_status.md"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, required=True)
    ap.add_argument("--idle-dir", type=Path)
    ap.add_argument("--no-ssh-recheck", action="store_true")
    a = ap.parse_args()

    site = load_json(a.site)
    runs = Path(site["runs_dir"])
    plan = [f"读取 idle-check 结果：{a.idle_dir or '最新一次完整的 envcheck/idle-check_*'}",
            "逐个节点判定 ssh / 空闲 / 占用" + ("" if a.no_ssh_recheck else "，ssh 失败的节点再 ssh 复核一次"),
            f"写 {site['hostfile']}、{runs / 'idle_hosts'}、{runs / 'node_status.md'}"]
    with benchlog.Step(runs, "screen", plan) as st:
        screen(a, site, runs, st)


def screen(a, site, runs, st):
    def n_rows(d):
        f = d / "results.tsv"
        return sum(1 for l in f.read_text().splitlines()[1:] if l.strip()) if f.is_file() else 0

    # Newest complete run; an interrupted idle-check leaves results.tsv with only the header.
    complete = sorted((d for d in (runs / "envcheck").glob("idle-check_*") if n_rows(d)), key=lambda d: d.name)
    idle_dir = a.idle_dir or (complete[-1] if complete else None)
    if idle_dir is None or not n_rows(idle_dir):
        msg = f"idle-check 结果为空或不完整: {idle_dir or runs / 'envcheck/idle-check_*'}"
        st.end("未执行", [msg + "，未改动 hostfile/idle_hosts"],
               [("idle-check", "结果为空（多半是运行被中断）", "重跑 `prepare_nodes.sh <site.json> idle`")],
               [idle_dir] if idle_dir else [])
        raise SystemExit(msg + "，未改动 hostfile/idle_hosts")

    rows = []
    for line in (idle_dir / "results.tsv").read_text().splitlines()[1:]:
        if not line.strip():
            continue
        host, rc = line.split("\t")[:2]
        log = idle_dir / f"{host}.log"
        text = log.read_text(errors="replace") if log.is_file() else ""
        rows.append((host, classify(host, int(rc), text, not a.no_ssh_recheck)))

    reach = [h for h, s in rows if s["ssh"] == "通"]
    idle = [h for h, s in rows if s["idle"] == "是"]
    Path(site["hostfile"]).write_text("".join(h + "\n" for h in reach))
    (runs / "idle_hosts").write_text("".join(h + "\n" for h in idle))

    stamp = re.search(r"(\d{8}_\d{6})", idle_dir.name)
    md = ["# 节点状态", "", f"检查时间：{stamp.group(1) if stamp else '?'}（{idle_dir}）", "",
          "| 节点 | ssh | 空闲 | 占用卡数 | 占用容器 | 备注 |", "|---|---|---|---|---|---|"]
    md += [f"| {h} | {s['ssh']} | {s['idle']} | {s['cards']} | {s['owners']} | {s['reason'] or '-'} |" for h, s in rows]
    (runs / "node_status.md").write_text("\n".join(md) + "\n")

    log_screen(st, runs, idle_dir, rows, reach, idle)
    print("\n".join(md))
    print(f"\n可 ssh {len(reach)}/{len(rows)} -> {site['hostfile']}")
    print(f"空闲   {len(idle)}/{len(rows)} -> {runs / 'idle_hosts'}")
    bad = [(h, s["reason"]) for h, s in rows if s["ssh"] == "不通"]
    if bad:
        print("ssh 不通（需 ssh-copy-id 或找管理员）:")
        for h, why in bad:
            print(f"  {h}: {why}")


if __name__ == "__main__":
    main()
