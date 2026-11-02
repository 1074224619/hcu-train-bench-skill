#!/usr/bin/env python3
"""Run matrix cases wave by wave; cases inside a wave run concurrently on disjoint hosts.

Usage: run_cases.py <run_dir> [--wave N] [--case ID] [--dry-run] [--rerun]

Per case:
  hostfiles/<id>      host slice for the launcher (run_dir must be visible in containers)
  logs/<id>.log       full launcher output
  status/<id>.json    state: running | done | stopped | failed | timeout, rc, timings,
                      vram_peak_percent (per host, from polling `hy-smi --showmemuse` while it runs)

The launcher runs on hosts[0]; its own mpirun reaches the other hosts. Once
`max_iters` iteration lines are seen the job's process group is terminated, so
no framework script has to be edited to shorten training.
"""
import argparse
import shlex
import subprocess
import threading
import time
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import benchlog
from common import ITER_RE, MEMUSE_CARD_RE, VRAM_POLL_INTERVAL_S, abspath_literal, find_case, load_run, save_json

TERM_GRACE_S = 60


def render(value, ctx):
    return str(value).format(**ctx)


def build_command(case, site, run_dir, hostfile):
    root = Path(site["frameworks_root"])
    launcher = root / case["launcher"]
    ctx = {"hostfile": hostfile, "nodes": case["nodes"], "gpus_per_node": site["gpus_per_node"],
           "ssh_port": site["container"]["ssh_port"], "run_dir": run_dir}
    env = " ".join(f"{k}={shlex.quote(render(v, ctx))}" for k, v in case.get("env", {}).items())
    args = " ".join(shlex.quote(render(a, ctx)) for a in case.get("args", []))
    # setsid makes bash the group leader, so $$ is the PGID used to stop the job.
    inner = (f"echo BENCH_PGID=$$; cd {shlex.quote(str(launcher.parent))} && "
             f"exec env {env} bash {shlex.quote(launcher.name)} {args}")
    payload = f"setsid --wait bash -c {shlex.quote(inner)}"
    head = case["hosts"][0]
    if site.get("launch_via", "docker-exec") == "local":
        return ["bash", "-c", payload], None
    exec_prefix = f"docker exec {shlex.quote(site['container']['name'])}"
    return ["ssh", "-o", "BatchMode=yes", head, f"{exec_prefix} bash -c {shlex.quote(payload)}"], exec_prefix


def kill_group(site, case, pgid, exec_prefix, sig):
    cmd = f"kill -{sig} -- -{pgid}"
    if exec_prefix is None:
        subprocess.run(["bash", "-c", cmd], check=False)
    else:
        subprocess.run(["ssh", "-o", "BatchMode=yes", case["hosts"][0], f"{exec_prefix} {cmd}"], check=False)


def memuse_cmd(exec_prefix, host):
    """Command to query `hy-smi --showmemuse --json` on one node, same transport (local/ssh+docker exec) as the case itself."""
    if exec_prefix is None:
        return ["bash", "-c", "hy-smi --showmemuse --json"]
    return ["ssh", "-o", "BatchMode=yes", host, f"{exec_prefix} hy-smi --showmemuse --json"]


def poll_vram_peak(case, exec_prefix, peaks, stop_event):
    """Background loop: every VRAM_POLL_INTERVAL_S, sample each case host's `hy-smi --showmemuse`
    and keep the running max (%) per host in `peaks`. Best-effort: an unreachable host or
    unparseable output just contributes no sample that tick."""
    while not stop_event.is_set():
        for h in case["hosts"]:
            try:
                out = subprocess.run(memuse_cmd(exec_prefix, h), capture_output=True, text=True, timeout=10).stdout
                vals = [float(v) for v in MEMUSE_CARD_RE.findall(out)]
            except Exception:
                vals = []
            if vals:
                peaks[h] = max(peaks.get(h, 0.0), max(vals))
        stop_event.wait(VRAM_POLL_INTERVAL_S)


def run_one(case, site, run_dir, dry_run):
    cid = case["id"]
    hostfile = run_dir / "hostfiles" / cid
    log = run_dir / "logs" / f"{cid}.log"
    status_path = run_dir / "status" / f"{cid}.json"
    for d in (hostfile.parent, log.parent):
        d.mkdir(parents=True, exist_ok=True)
    hostfile.write_text("\n".join(case["hosts"]) + "\n")

    cmd, exec_prefix = build_command(case, site, str(run_dir), str(hostfile))
    if dry_run:
        print(f"[dry-run] {cid}: {shlex.join(cmd)}")
        return {"id": cid, "state": "dry-run", "cmd": shlex.join(cmd)}

    max_iters, timeout_min = site.get("max_iters", 20), site.get("case_timeout_min", 60)
    plan = [f"在 {case['hosts'][0]} 启动 launcher，节点 {'+'.join(case['hosts'])}（{case['gpus']} 卡，wave {case['wave']}）",
            f"命令：`{shlex.join(cmd)}`",
            f"到第 {max_iters} 步自动停止；超过 {timeout_min} 分钟记 timeout",
            f"日志 {log}，状态 {status_path}"]
    with benchlog.Step(site["runs_dir"], "case", plan, run_dir=run_dir, target=cid) as st:
        status = execute(case, site, cmd, exec_prefix, log, status_path, max_iters, timeout_min * 60)
        issue = case_issue(status, run_dir, max_iters)
        st.end("成功" if not issue else {"timeout": "超时"}.get(status["state"], "失败"),
               [f"{status['state']}：rc={status['rc']}，到第 {status['last_iter']} 步（目标 {max_iters}），"
                f"用时 {status['duration_s']}s"], [issue] if issue else [], [log, status_path])
    return status


def execute(case, site, cmd, exec_prefix, log, status_path, max_iters, timeout_s):
    cid = case["id"]
    status = {"id": cid, "state": "running", "hosts": case["hosts"], "cmd": cmd, "start": time.time()}
    save_json(status_path, status)
    print(f"[start] {cid} on {','.join(case['hosts'])} -> {log}", flush=True)

    pgid, last_iter, reason = None, 0, None
    vram_peaks, vram_stop = {}, threading.Event()
    vram_thread = threading.Thread(target=poll_vram_peak, args=(case, exec_prefix, vram_peaks, vram_stop), daemon=True)
    vram_thread.start()
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        timer = threading.Timer(timeout_s, lambda: proc.poll() is None and stop("timeout"))

        def stop(why):
            nonlocal reason
            reason = reason or why
            if pgid:
                kill_group(site, case, pgid, exec_prefix, "TERM")
                killer = threading.Timer(TERM_GRACE_S, lambda: proc.poll() is None and kill_group(
                    site, case, pgid, exec_prefix, "KILL"))
                killer.daemon = True
                killer.start()
            else:
                proc.terminate()

        timer.daemon = True
        timer.start()
        for line in proc.stdout:
            fh.write(line)
            if pgid is None and line.startswith("BENCH_PGID="):
                pgid = line.strip().split("=", 1)[1]
            m = ITER_RE.search(line)
            if m:
                last_iter = int(m.group(1))
                if last_iter >= max_iters and reason is None:
                    stop("max_iters")
        rc = proc.wait()
        timer.cancel()
    vram_stop.set()
    vram_thread.join(timeout=VRAM_POLL_INTERVAL_S + 5)

    state = {"max_iters": "done", "timeout": "timeout"}.get(reason, "done" if rc == 0 else "failed")
    status.update(state=state, rc=rc, reason=reason, last_iter=last_iter, end=time.time())
    status["duration_s"] = round(status["end"] - status["start"])
    status["vram_peak_percent"] = {h: round(v, 1) for h, v in vram_peaks.items()}
    save_json(status_path, status)
    peak_str = "，".join(f"{h}={v}%" for h, v in status["vram_peak_percent"].items())
    print(f"[{state}] {cid} rc={rc} last_iter={last_iter} ({status['duration_s']}s)"
          + (f" vram_peak: {peak_str}" if peak_str else ""), flush=True)
    return status


def case_issue(st, run_dir, max_iters):
    """(对象, 问题, 处理方案) for a case that did not reach max_iters, else None."""
    if st["state"] in ("done", "dry-run") and (st["state"] == "dry-run" or st.get("reason") == "max_iters"
                                              or st.get("last_iter", 0) >= max_iters):
        return None
    log = run_dir / "logs" / f"{st['id']}.log"
    tail = "\n".join(log.read_text(errors="replace").splitlines()[-400:]) if log.exists() else ""
    host = st.get("hosts", ["<节点>"])[0]
    if st["state"] == "timeout":
        prob, fix = (f"超时，最后到第 {st.get('last_iter', 0)} 步",
                     "看日志是否卡在初始化/通信；确实慢就调大 site.json 的 case_timeout_min")
        d = benchlog.diagnose(tail, host)
        if d:
            prob, fix = f"超时；{d[0]}：`{d[2]}`", d[1]
    elif st["state"] == "done":
        prob, fix = (f"进程正常退出但只跑到第 {st.get('last_iter', 0)} 步（目标 {max_iters}）",
                     "检查脚本的 train_iters 是否小于 max_iters，或日志里 iteration 行格式是否变化")
    else:
        prob, fix = benchlog.explain(tail, host, "查看用例日志末尾定位原因，修复后用 `--case` 单独重跑")
        prob = f"rc={st.get('rc')}，第 {st.get('last_iter', 0)} 步时退出；{prob}"
    return st["id"], prob, fix


def log_run(st, site, a, run_dir, statuses, interrupted):
    max_iters = site.get("max_iters", 20)
    real = [s for s in statuses if s and s["state"] != "dry-run"]
    issues = [i for i in (case_issue(s, run_dir, max_iters) for s in statuses if s) if i]
    results = []
    if a.dry_run:
        results.append(f"预演模式，只生成命令，未在节点上执行：{len(statuses)} 个用例")
        results += [f"{s['id']}：`{s['cmd']}`" for s in statuses if s]
        status = "预演"
    else:
        ok = len(real) - len(issues)
        results.append(f"执行 {len(real)} 个用例：完成 {ok}，失败/超时 {len(issues)}（目标 {max_iters} 步）")
        for s in real:
            peak = s.get("vram_peak_percent") or {}
            peak_str = "，显存峰值 " + "，".join(f"{h}={v}%" for h, v in peak.items()) if peak else ""
            results.append(f"{s['id']}：{s['state']}，到第 {s.get('last_iter', 0)} 步，用时 {s.get('duration_s', '?')}s，"
                            f"节点 {'+'.join(s['hosts'])}{peak_str}")
        if not real:
            results.append("没有需要执行的用例（已完成的会跳过，加 --rerun 强制重跑）")
        status = benchlog.status_of(ok, len(issues)) if real else "无操作"
    if interrupted:
        status = "中断"
        results.insert(0, "被 Ctrl-C 中断；下面只统计已结束的用例，重跑同一命令会跳过已完成的")
    st.end(status, results, issues, [run_dir / "logs", run_dir / "status"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--wave", type=int)
    ap.add_argument("--case")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="also rerun cases already done")
    a = ap.parse_args()
    run_dir, matrix, site = load_run(abspath_literal(a.run_dir))

    cases = [find_case(matrix, a.case)] if a.case else matrix["cases"]
    if a.wave is not None:
        cases = [c for c in cases if c["wave"] == a.wave]
    if not a.rerun and not a.case:
        def finished(c):
            p = run_dir / "status" / f"{c['id']}.json"
            return p.exists() and '"state": "done"' in p.read_text()
        cases = [c for c in cases if not finished(c)]

    waves = sorted({c["wave"] for c in cases})
    plan = [f"共 {len(cases)} 个用例，分 {len(waves)} 个 wave：" + "；".join(
                f"wave {w} " + "，".join(c["id"] for c in cases if c["wave"] == w) for w in waves)
            if cases else "没有需要执行的用例（已完成的会跳过，加 --rerun 强制重跑）"]
    plan.append("预演：只生成命令，不在节点上执行" if a.dry_run else
                f"wave 依次执行，同一 wave 内并发；每个用例到第 {site.get('max_iters', 20)} 步停止，单独记开始/结果")
    with benchlog.Step(site["runs_dir"], "run", plan, run_dir=run_dir) as st:
        run_waves(st, a, site, run_dir, cases)


def run_waves(st, a, site, run_dir, cases):
    statuses, interrupted = [], False
    try:
        for wave in sorted({c["wave"] for c in cases}):
            batch = [c for c in cases if c["wave"] == wave]
            print(f"=== wave {wave}: {', '.join(c['id'] for c in batch)}", flush=True)
            out = [None] * len(batch)

            def work(i, c):
                out[i] = run_one(c, site, run_dir, a.dry_run)

            threads = [threading.Thread(target=work, args=(i, c)) for i, c in enumerate(batch)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            statuses += out
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        # Also on Ctrl-C: record what finished so far.
        log_run(st, site, a, run_dir, statuses, interrupted)


if __name__ == "__main__":
    main()
