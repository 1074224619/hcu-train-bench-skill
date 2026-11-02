#!/usr/bin/env python3
"""在指定节点的容器里跑一个模型，跑到第 N 个 iteration 自动停止，保存日志并写中文结果。

Usage:
  run_model.py --hosts NODE[,NODE...] --container NAME --launcher <框架 example 下的脚本绝对路径>
               [--arg A ...] [--env KEY=VAL ...] [--max-iters 20] [--timeout-min 60]
               [--out DIR] [--local] [--dry-run] [--site site.json]

  --arg 可用占位符：{hostfile} {nodes}（按 --hosts 生成的 hostfile 和节点数）
  --local  不经 ssh/docker exec，直接在当前机器（例如已在容器里）运行
  --site   给出时日志同时写入 <runs_dir>/bench_log.md

输出（默认 <skill>/runs/<时间>_<launcher名>/）：
  hostfile        节点列表
  cmd.txt         实际执行的命令
  train.log       launcher 全部输出
  status.json     state(done|failed|timeout|stopped)、rc、last_iter、耗时、每步 tflops/耗时、
                  各节点显存占用峰值（vram_peak_percent，hy-smi --showmemuse 后台轮询取最大值）
  result.md       中文结论：状态、性能、显存占用峰值、末尾报错和可能原因
  run_log.md      执行前一条「开始」（命令、将执行的操作），执行后一条结果（状态、问题、耗时、明细）
只依赖 Python 标准库，不依赖 skills/common 和其他子 skill。
"""
import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ITER_RE = re.compile(r"iteration\s+(\d+)/\s*(\d+)\s*\|.*?elapsed time per iteration \(ms\):\s*([\d.]+)"
                     r"(?:.*?throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+))?")
# `hy-smi --showmemuse --json` prints one block per card, e.g.
# {"card0":{"HCU memory use (%)":"31"},"card1":{"HCU memory use (%)":"0"}}; match the
# percentage inside each card's block regardless of exact key wording/card index.
MEMUSE_CARD_RE = re.compile(r'"(?:card|hcu)[^"]*"\s*:\s*\{[^}]*?"[^"]*"\s*:\s*"?([\d.]+)"?', re.I)
VRAM_POLL_INTERVAL_S = 5
HINTS = [  # (正则, 可能原因)
    (r"Permission denied \(publickey", "ssh 免密未配置（节点或容器之间）"),
    (r"Connection timed out|No route to host|Connection refused", "ssh 端口不通：检查 mpirun 的 plm_rsh_args 端口是否等于容器 sshd 端口"),
    (r"No such container|is not running", "容器不存在或未运行"),
    (r"No such file or directory|FileNotFoundError|not found", "路径不存在：数据集/模型/tokenizer/env 脚本路径写死或不在挂载内"),
    (r"/path/to/", "脚本里还是占位路径 /path/to/..."),
    (r"out of memory|OutOfMemoryError", "显存不足 (OOM)：调大 tp/pp、减小 mbs、开 recompute"),
    (r"ModuleNotFoundError", "容器缺少 Python 包"),
    (r"NCCL error|RCCL|nccl\w*Error|collective operation timeout", "集合通信失败/超时"),
    (r"[Aa]ddress already in use", "端口被占用：换 MASTER_PORT/PORT 或清理残留进程"),
]
STATE_ZH = {"done": "成功", "failed": "失败", "timeout": "超时", "stopped": "中断"}


def log_entry(a, out, status, lines):
    """追加一条中文日志到 <out>/run_log.md（给了 --site 时也写 <runs_dir>/bench_log.md），格式同 common/benchlog.py。"""
    entry = "\n".join([f"## [{time.strftime('%Y-%m-%d %H:%M:%S')}] 3 跑单个模型 {Path(a.launcher).name} —— {status}", "",
                       f"- 命令：`{shlex.join(['run_model.py'] + sys.argv[1:])}`", *lines]) + "\n\n"
    targets = [(out / "run_log.md", f"# 跑模型日志 {out.name}\n\n")]
    if a.site:
        runs = Path(json.loads(Path(a.site).read_text())["runs_dir"])
        targets.append((runs / "bench_log.md", "# hcu-bench 运行日志\n\n每个步骤执行前写一条「开始」（命令、将执行的操作），"
                        "执行后写一条结果（状态、结果、问题与处理方案、耗时、明细路径）。最新的在最后。\n\n"))
    for path, header in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(header if new else "")
            fh.write(entry)


def build_cmd(a, out):
    hosts = a.hosts.split(",")
    hostfile = out / "hostfile"
    hostfile.write_text("\n".join(hosts) + "\n")
    ctx = {"hostfile": str(hostfile), "nodes": len(hosts)}
    launcher = Path(abspath_literal(a.launcher))
    env = " ".join(shlex.quote(e.format(**ctx)) for e in a.env)
    args = " ".join(shlex.quote(x.format(**ctx)) for x in a.arg)
    inner = (f"echo RUN_PGID=$$; cd {shlex.quote(str(launcher.parent))} && "
             f"exec env {env} bash {shlex.quote(launcher.name)} {args}")
    payload = f"setsid --wait bash -c {shlex.quote(inner)}"
    if getattr(a, 'local'):
        return ["bash", "-c", payload], None
    prefix = f"docker exec {shlex.quote(a.container)}"
    return ["ssh", "-o", "BatchMode=yes", hosts[0], f"{prefix} bash -c {shlex.quote(payload)}"], prefix


def kill(a, prefix, pgid, sig):
    cmd = f"kill -{sig} -- -{pgid}"
    if prefix is None:
        subprocess.run(["bash", "-c", cmd], check=False)
    else:
        subprocess.run(["ssh", "-o", "BatchMode=yes", a.hosts.split(",")[0], f"{prefix} {cmd}"], check=False)


def memuse_cmd(a, host):
    """Command to query `hy-smi --showmemuse --json` on one node, same transport (local/ssh+docker exec) as the job itself."""
    if getattr(a, "local"):
        return ["bash", "-c", "hy-smi --showmemuse --json"]
    return ["ssh", "-o", "BatchMode=yes", host, f"docker exec {shlex.quote(a.container)} hy-smi --showmemuse --json"]


def poll_vram_peak(a, hosts, peaks, stop_event):
    """Background loop: every VRAM_POLL_INTERVAL_S, sample each host's `hy-smi --showmemuse`
    and keep the running max (%) per host in `peaks`. Best-effort: a host that's briefly
    unreachable or whose output doesn't parse just contributes no sample that tick."""
    while not stop_event.is_set():
        for h in hosts:
            try:
                out = subprocess.run(memuse_cmd(a, h), capture_output=True, text=True, timeout=10).stdout
                vals = [float(v) for v in MEMUSE_CARD_RE.findall(out)]
            except Exception:
                vals = []
            if vals:
                peaks[h] = max(peaks.get(h, 0.0), max(vals))
        stop_event.wait(VRAM_POLL_INTERVAL_S)


def run(a, cmd, prefix, out):
    log = out / "train.log"
    st = {"state": "running", "start": time.time(), "iters": []}
    pgid, reason = None, None
    vram_peaks, vram_stop = {}, threading.Event()
    vram_thread = threading.Thread(target=poll_vram_peak, args=(a, a.hosts.split(","), vram_peaks, vram_stop), daemon=True)
    vram_thread.start()
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        def stop(why):
            nonlocal reason
            reason = reason or why
            if pgid:
                kill(a, prefix, pgid, "TERM")
                t = threading.Timer(60, lambda: proc.poll() is None and kill(a, prefix, pgid, "KILL"))
                t.daemon = True
                t.start()
            else:
                proc.terminate()

        timer = threading.Timer(a.timeout_min * 60, lambda: proc.poll() is None and stop("timeout"))
        timer.daemon = True
        timer.start()
        try:
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                print(line, end="", flush=True)
                if pgid is None and line.startswith("RUN_PGID="):
                    pgid = line.strip().split("=", 1)[1]
                m = ITER_RE.search(line)
                if m:
                    st["iters"].append([int(m.group(1)), float(m.group(3)), float(m.group(4)) if m.group(4) else None])
                    if int(m.group(1)) >= a.max_iters and reason is None:
                        stop("max_iters")
        except KeyboardInterrupt:
            stop("stopped")
        rc = proc.wait()
        timer.cancel()
    vram_stop.set()
    vram_thread.join(timeout=VRAM_POLL_INTERVAL_S + 5)
    st["rc"], st["reason"] = rc, reason
    st["last_iter"] = st["iters"][-1][0] if st["iters"] else 0
    st["state"] = {"max_iters": "done", "timeout": "timeout", "stopped": "stopped"}.get(
        reason, "done" if rc == 0 and st["last_iter"] >= a.max_iters else "failed")
    st["duration_s"] = round(time.time() - st["start"])
    st["vram_peak_percent"] = {h: round(v, 1) for h, v in vram_peaks.items()}
    return st


def write_result(a, out, st):
    """写 result.md；返回 (结论行, 问题) 给执行后的日志用。"""
    tail = (out / "train.log").read_text(errors="replace").splitlines()[-400:]
    summary = [f"状态：{st['state']}（rc={st['rc']}，到第 {st['last_iter']} 步，目标 {a.max_iters} 步，用时 {st['duration_s']}s）"]
    stable = [r for r in st["iters"] if r[0] > a.warmup]
    if stable:
        t = statistics.median(r[1] for r in stable) / 1000
        tf = [r[2] for r in stable if r[2] is not None]
        summary.append(f"性能（跳过前 {a.warmup} 步，中位数）：time/iter={t:.3f}s"
                       + (f"，tflops={statistics.median(tf):.1f}" if tf else ""))
    elif st["iters"]:
        summary.append(f"迭代数不超过 warmup={a.warmup}，未统计性能")
    peaks = st.get("vram_peak_percent") or {}
    if peaks:
        summary.append("显存占用峰值（hy-smi --showmemuse）：" + "，".join(f"{h}={v}%" for h, v in peaks.items()))
    lines = [f"# 跑模型结果：{Path(a.launcher).name}", "", f"- {summary[0]}", f"- 节点：{a.hosts}", "- 命令：见 cmd.txt", ""]
    lines += [f"- {s}" for s in summary[1:]] + ([""] if summary[1:] else [])
    problem = ""
    if st["state"] != "done":
        text = "\n".join(tail)
        hints = [h for rx, h in HINTS if re.search(rx, text)]
        errs = [l for l in tail if re.search(r"(?i)error|exception|failed|denied|killed|Traceback", l)][-8:]
        problem = "可能原因：" + ("；".join(hints) if hints else "未识别，看下面的报错行和 train.log")
        lines += ["## 问题", "", problem, "", "末尾报错行：", "", "```", *(errs or tail[-15:]), "```"]
    (out / "result.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return summary, problem


def abspath_literal(path):
    """Like os.path.abspath but joins relative paths against $PWD (bash's logical,
    non-dereferenced cwd) instead of os.getcwd() (a kernel getcwd() syscall that
    follows any symlink in the chain). Some sandboxes keep this script's own working
    directory as a live symlink to a rotating target; a container's bind mounts are
    fixed at create time to the literal path used then, so re-resolving through
    getcwd() later can silently produce a path that was never mounted, causing
    "No such file or directory" inside the container even though the file exists
    on the login node.
    """
    p = str(path)
    if not os.path.isabs(p):
        p = os.path.join(os.environ.get("PWD") or os.getcwd(), p)
    return os.path.normpath(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", required=True, help="逗号分隔，第一个节点上启动 launcher")
    ap.add_argument("--container", default="hcu-bench")
    ap.add_argument("--launcher", required=True)
    ap.add_argument("--arg", action="append", default=[])
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL，传给 launcher")
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timeout-min", type=int, default=60)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--site", type=Path, help="site.json；给出时日志同时写入 <runs_dir>/bench_log.md")
    a = ap.parse_args()

    # 用 abspath_literal 而不是 Path.resolve()/os.path.abspath：两者都经过 os.getcwd()，
    # 会把路径里任何符号链接都解析掉。本机 /public/home/wangxj3 是指向 /public2/... 的符号链接，
    # 而这个沙箱里脚本自己的工作目录也可能是一个会变的符号链接；解析后的路径可能和建容器时
    # 实际挂进去的路径不一致，容器内就会报 "No such file or directory"。
    # 用 sys.argv[0] 而不是 __file__：当脚本以相对路径启动时，Python 在脚本真正开始执行前
    # 就已经把 __file__ 转成绝对路径了（内部走的也是会解析符号链接的 getcwd()），这一步比
    # 我们自己的代码更早发生，事后已经无法分辨原始相对路径；sys.argv[0] 保留的是命令行上
    # 原样输入的字符串，没有被提前转换。
    here = Path(abspath_literal(sys.argv[0])).parent.parent
    # 日志/输出目录默认放到 BenchAgent/runs/（site.json 的 runs_dir），不要放在 skills/ 下面，
    # 否则和 skill 本身的代码混在一起、显得很乱；只有没给 --site 时才退回 skills/hcu-run-model/runs/
    # （没有别的地方可以参照）。
    if a.site:
        default_root = Path(json.loads(Path(a.site).read_text())["runs_dir"]) / "hcu-run-model"
    else:
        default_root = here / "runs"
    out = Path(abspath_literal(a.out or default_root /
               f"{time.strftime('%Y%m%d_%H%M%S')}_{Path(a.launcher).stem}"))
    out.mkdir(parents=True, exist_ok=True)
    cmd, prefix = build_cmd(a, out)
    (out / "cmd.txt").write_text(shlex.join(cmd) + "\n")
    print(f"输出目录：{out}\n命令：{shlex.join(cmd)}\n", flush=True)
    where = "本机" if getattr(a, 'local') else f" {a.hosts.split(',')[0]} 的容器 {a.container} "
    log_entry(a, out, "开始", ["- 将执行：", f"  - 在{where}里启动 {a.launcher}，节点 {a.hosts}",
                               f"  - 命令：`{shlex.join(cmd)}`",
                               "  - 预演：只生成命令写入 cmd.txt，不执行" if a.dry_run else
                               f"  - 到第 {a.max_iters} 步自动停止；超过 {a.timeout_min} 分钟记 timeout",
                               f"  - 输出目录 {out}"])
    t0 = time.time()
    if a.dry_run:
        log_entry(a, out, "预演", ["- 结果：", "  - 只生成命令，未执行", "- 问题：无", f"- 明细：`{out / 'cmd.txt'}`"])
        return
    try:
        st = run(a, cmd, prefix, out)
        (out / "status.json").write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n")
        summary, problem = write_result(a, out, st)
    except BaseException as e:
        log_entry(a, out, "异常", ["- 结果：", f"  - 脚本中途退出：{type(e).__name__}: {e}",
                                   "- 问题：看终端输出的 Traceback 和 train.log", f"- 耗时：{time.time() - t0:.0f}s",
                                   f"- 明细：`{out / 'train.log'}`"])
        raise
    log_entry(a, out, STATE_ZH.get(st["state"], st["state"]),
              ["- 结果：", *(f"  - {l}" for l in summary), f"- 问题：{problem or '无'}", f"- 耗时：{st['duration_s']}s",
               f"- 明细：`{out / 'result.md'}`；`{out / 'train.log'}`"])


if __name__ == "__main__":
    main()
