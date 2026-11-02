#!/usr/bin/env python3
"""中文步骤日志：每一步执行前、执行后各追加一条到 <runs_dir>/bench_log.md；测试相关步骤同时写 <run_dir>/run_log.md。

执行前一条「开始」：时间、步骤、命令、将执行的操作。
执行后一条结果：时间、步骤、状态、命令、结果、问题与处理方案（表格）、耗时、明细路径。
Python 脚本用 Step（with 块退出时若没写结果，会按异常/中断补写一条）；prepare_nodes.sh 和手动操作通过命令行：

  benchlog.py begin --site S --step create --cmd "..." [--plan "..." ...] [--hosts F]
  benchlog.py envcheck --site S --step create --rc N --terminal F --since T --cmd "..." [--hosts F] [--dry-run]
  benchlog.py note --site S --step healthy --cmd "..." --status 成功 --result "..." [--since T] \
      [--issue '对象|问题|处理方案' ...] [--detail PATH ...]

AI 自己做的操作（step 默认 ai，标题「AI 操作 <target>」）：
  benchlog.py exec --site S --target "核对 qwen3 数据路径" --plan "为什么做、做什么" -- ssh bw67 ls /x   # 自动写前后两条
  benchlog.py begin/note --site S --target "改 site.json 的 mounts" ...                                  # 非命令操作手动写
"""
import subprocess
import argparse
import re
import shlex
import sys
import threading
import time
from pathlib import Path

from common import load_json, read_hosts

STEP_TITLES = {
    "scan": "0 扫描用例", "paths": "0 补齐路径", "idle": "1.1 空闲检测", "screen": "1.1 节点筛查",
    "status": "1.2 容器状态", "create": "1.2 建容器", "recreate": "1.2 重建容器", "delete": "1.2 删除容器",
    "check": "1.3 平台/资源检查", "rccl": "1.3 单机 RCCL", "healthy": "1.4 健康节点",
    "matrix": "2 生成测试列表", "run": "3 执行", "case": "3 执行用例", "parse": "4 解析", "export": "5 写入 cookbook",
    "ai": "AI 操作",
}
LOG_HEADER = ("# hcu-bench 运行日志\n\n每个步骤执行前写一条「开始」（命令、将执行的操作），执行后写一条结果"
              "（状态、结果、问题与处理方案、耗时、明细路径）。最新的在最后。\n\n")
_LOCK = threading.Lock()  # run_cases.py 的并发用例会同时写日志

# (正则, 问题, 处理方案)；按顺序取第一个命中的。{0}.. 是正则分组，{host} 是节点名。
DIAGNOSES = [
    (r"Permission denied(?: \(publickey[^)]*\))?\.?$|Permission denied \(publickey", "ssh 免密未配置",
     "在登录节点执行 `ssh-copy-id {host}` 或联系管理员；修复后重跑 `idle`"),
    (r"Connection timed out|No route to host|port 22: Connection refused", "节点网络不通或已宕机",
     "联系管理员确认节点状态；重跑 `idle` 会把它排除出 hostfile"),
    (r"Could not resolve hostname", "主机名无法解析", "检查 hostfile.all 的节点名拼写或 /etc/hosts"),
    (r"Host key verification failed|REMOTE HOST IDENTIFICATION HAS CHANGED", "ssh 主机指纹不匹配",
     "确认节点没有被替换后执行 `ssh-keygen -R {host}` 再重试"),
    (r"Mounting (\S+) is not allowed", "集群禁止挂载 `{0}`",
     "把 site.json 的 `container.mounts` 改成允许挂载的子目录（如 `/public/Users/<用户名>/`）后重跑 `create`"),
    (r"container name .* is already in use", "同名容器已存在",
     "先 `status` 查看；确认旧容器可删除并经用户同意后执行 `recreate --yes`"),
    (r"pull access denied|manifest unknown|repository does not exist|No such image", "镜像不存在或无拉取权限",
     "检查 site.json 的 `container.image`，或在节点上 `docker login` 镜像仓库"),
    (r"Cannot connect to the Docker daemon|permission denied while trying to connect to the Docker daemon",
     "docker 未启动或当前用户无 docker 权限", "联系管理员启动 docker 或把用户加入 docker 组"),
    (r"port is already allocated|[Aa]ddress already in use", "端口被占用",
     "建容器时改 site.json 的 `container.ssh_port`；训练时清理残留进程或换 PORT"),
    (r"No such container", "容器不存在", "先执行 `create --yes`"),
    (r"/path/to/\S*", "脚本里仍是占位路径 `{m}`", "在 site.json 填写该模型的数据集/模型路径"),
    (r"model directory not found: (\S+)", "模型目录不存在 `{0}`",
     "检查 site.json 的模型路径，并确认它在 `container.mounts` 内"),
    (r"(?:CUDA|HIP|hip) out of memory|OutOfMemoryError", "显存不足 (OOM)",
     "增大 tp/pp、减小 mbs 或开 recompute，或改跑卡数更多的用例"),
    (r"ModuleNotFoundError: No module named '([^']+)'", "容器缺少 Python 包 `{0}`",
     "在镜像中安装该包，或换用包含它的镜像"),
    (r"nccl(?:Internal|System|Unhandled|Remote)Error|NCCL error|RCCL error", "集合通信 (RCCL) 失败",
     "对相关节点跑 `rccl`；排除故障节点后重新生成测试列表"),
    (r"Watchdog caught collective operation timeout|ProcessGroup\w*.*[Tt]imeout", "集合通信超时",
     "检查节点间网络/RDMA，对相关节点跑 `rccl`"),
    (r"cannot (?:access|open|stat) '([^']+)': No such file or directory", "文件不存在 `{0}`",
     "检查路径是否正确、是否在 `container.mounts` 内"),
    (r"(?:FileNotFoundError|No such file or directory)[^\d\n]*?(/[^\s'\":]+)", "文件不存在 `{0}`",
     "检查路径是否正确、是否在 `container.mounts` 内"),
]
_DIAG = [(re.compile(p, re.M), prob, fix) for p, prob, fix in DIAGNOSES]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
NOISE = re.compile(r"^Warning: Permanently added")
ERRORISH = re.compile(r"(?i)error|exception|failed|denied|killed|abort")


def strip_ansi(text):
    return ANSI.sub("", text)


def short(line, n=160):
    line = strip_ansi(line).strip()
    return line if len(line) <= n else line[:n] + "…"


def diagnose(text, host="<节点>"):
    """Returns (问题, 处理方案, 证据行) for the first matching pattern, or None."""
    text = strip_ansi(text)
    for rx, prob, fix in _DIAG:
        m = rx.search(text)
        if m:
            line = text[text.rfind("\n", 0, m.start()) + 1:].split("\n", 1)[0]
            return prob.format(*m.groups(), m=m.group(0)), fix.format(host=host), short(line)
    return None


def last_error(text):
    lines = [l for l in strip_ansi(text).splitlines() if l.strip() and not NOISE.match(l)]
    hits = [l for l in lines if ERRORISH.search(l)]
    return short((hits or lines or [""])[-1])


def explain(text, host="<节点>", fallback_fix="查看明细日志定位原因"):
    """Always returns (问题, 处理方案): a known diagnosis, else the last error-like line."""
    d = diagnose(text, host)
    if d:
        return f"{d[0]}：`{d[2]}`", d[1]
    line = last_error(text)
    return (f"未识别的错误：`{line}`" if line else "未识别的错误（无输出）"), fallback_fix


def _cell(v):
    return str(v).replace("|", "\\|").replace("\n", " ")


def cmdline():
    """当前脚本的命令行（脚本名 + 参数），用作日志里的「命令」。"""
    return shlex.join([Path(sys.argv[0]).name] + sys.argv[1:])


def _head(step, target, status):
    title = STEP_TITLES.get(step, step) + (f" {target}" if target else "")
    return [f"## [{time.strftime('%Y-%m-%d %H:%M:%S')}] {title} —— {status}", ""]


def render_begin(step, cmd, plan, target=""):
    out = _head(step, target, "开始")
    if cmd:
        out.append(f"- 命令：`{cmd}`")
    if plan:
        out.append("- 将执行：")
        out += [f"  - {p}" for p in plan]
    return "\n".join(out) + "\n\n"


def render(step, cmd, status, results, issues, details, target="", elapsed=None):
    out = _head(step, target, status)
    if cmd:
        out.append(f"- 命令：`{cmd}`")
    if results:
        out.append("- 结果：")
        out += [f"  - {r}" for r in results]
    if issues:
        out += ["- 问题与处理方案：", "", "  | 对象 | 问题 | 处理方案 |", "  |---|---|---|"]
        out += [f"  | {_cell(o)} | {_cell(p)} | {_cell(f)} |" for o, p, f in issues]
        out.append("")
    else:
        out.append("- 问题：无")
    if elapsed is not None:
        out.append(f"- 耗时：{elapsed:.0f}s")
    if details:
        out.append("- 明细：" + "；".join(f"`{d}`" for d in details))
    return "\n".join(out) + "\n\n"


def _append(path, header, entry):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        new = not path.exists()
        with open(path, "a", encoding="utf-8") as fh:
            if new:
                fh.write(header)
            fh.write(entry)


def _emit(runs_dir, run_dir, entry):
    _append(Path(runs_dir) / "bench_log.md", LOG_HEADER, entry)
    if run_dir:
        _append(Path(run_dir) / "run_log.md", f"# 测试运行日志 {Path(run_dir).name}\n\n", entry)
    return entry


def begin(runs_dir, step, cmd, plan=(), run_dir=None, target=""):
    """执行前：追加一条「开始」记录。Returns the entry text."""
    return _emit(runs_dir, run_dir, render_begin(step, cmd, list(plan), target))


def write(runs_dir, step, cmd, status, results=(), issues=(), details=(), run_dir=None, target="", elapsed=None):
    """执行后：追加一条结果记录到 <runs_dir>/bench_log.md（和 <run_dir>/run_log.md）。Returns the entry text."""
    entry = render(step, cmd, status, list(results), list(issues), [str(d) for d in details], target, elapsed)
    return _emit(runs_dir, run_dir, entry)


class Step:
    """一步操作：进入 with 时写「开始」，end() 写结果；with 退出时还没写结果（异常、Ctrl-C、提前退出）就补写一条。

    runs_dir 为 None 时不写日志（例如 scan_examples.py 没给 --site）。
    """

    def __init__(self, runs_dir, step, plan=(), cmd=None, run_dir=None, target=""):
        self.runs_dir, self.step, self.plan, self.run_dir, self.target = runs_dir, step, list(plan), run_dir, target
        self.cmd = cmdline() if cmd is None else cmd
        self.t0, self.ended = time.time(), False

    def __enter__(self):
        if self.runs_dir:
            begin(self.runs_dir, self.step, self.cmd, self.plan, self.run_dir, self.target)
        return self

    def end(self, status, results=(), issues=(), details=()):
        self.ended = True
        if self.runs_dir:
            write(self.runs_dir, self.step, self.cmd, status, results, issues, details, self.run_dir,
                  self.target, time.time() - self.t0)

    def __exit__(self, et, ev, tb):
        if self.ended:
            return False
        if et is None or (et is SystemExit and ev.code in (0, None)):
            self.end("完成", ["脚本正常结束，没有写入更多结果"])
        elif et is KeyboardInterrupt:
            self.end("中断", ["被 Ctrl-C 中断，未完成"], [("-", "用户中断", "确认现场后重跑同一命令")])
        else:
            msg = str(ev) if et is SystemExit else f"{et.__name__}: {ev}"
            prob, fix = explain(msg, fallback_fix="看终端输出的 Traceback 定位原因，修复后重跑")
            self.end("失败" if et is SystemExit else "异常", [f"脚本中途退出：{short(msg)}"], [("脚本", prob, fix)])
        return False


def benchlog_status(rc, n_ok, n_bad):
    return "成功" if rc == 0 and not n_bad else status_of(n_ok, max(n_bad, 1))


def status_of(n_ok, n_bad):
    return "成功" if not n_bad else ("失败" if not n_ok else "部分失败")


# ---------------------------------------------------------------- hcu-envcheck 输出解析
GROUP_RE = re.compile(r"^\[(PASS|FAIL|ERROR|WARN)\]\s+nodes=(\S+)\s+(.*)$")
KV_RE = re.compile(r"(\w+)=(.*?)(?=\s+\w+=|$)")
NEXT_STEP = {
    "status": "缺容器的节点执行 `create`（先预演再 `--yes`）",
    "create": "重跑 `create --yes`；成功后执行 `check`",
    "recreate": "重跑 `recreate --yes`（需用户同意）",
    "check": "失败节点会在 `healthy` 中排除；如需保留，先按上表修复再重跑 `check`",
    "rccl": "失败节点会在 `healthy` 中排除；如需保留，先按上表修复再重跑 `rccl`",
}


def _blocks(detail_log):
    """lifecycle-details.log -> {node: (status, block text, stage)} for the last stage that ran on each node."""
    out, node, buf = {}, None, []
    for line in strip_ansi(Path(detail_log).read_text(errors="replace")).splitlines() + ["[DETAIL] node=_"]:
        m = re.match(r"\[DETAIL\] node=(\S+)", line)
        if m:
            if node:
                st = re.search(r"\[NODE-RESULT\].*?stage=([\w-]+) status=(\w+)", "\n".join(buf))
                out[node] = (st.group(2) if st else "?", "\n".join(buf), st.group(1) if st else "?")
            node, buf = m.group(1), []
        elif node:
            buf.append(line)
    return out


def _stage_dir(out_dir, op, since):
    dirs = [d for d in Path(out_dir).glob(f"per-node-container_{op}_stages_*") if d.stat().st_mtime >= since - 1]
    return max(dirs, key=lambda d: d.stat().st_mtime, default=None)


def _failures(stage_dir):
    """failures.tsv rows -> [(nodes, 问题, 处理方案)]."""
    path = stage_dir / "failures.tsv"
    if not path.is_file():
        return []
    lines = path.read_text(errors="replace").splitlines()
    head = lines[0].split("\t") if lines else []
    issues = []
    for line in lines[1:]:
        row = dict(zip(head, line.split("\t")))
        ev = Path(row.get("evidence", ""))
        text = row.get("detail", "")
        if ev.is_file():
            text += "\n" + "\n".join(ev.read_text(errors="replace").splitlines()[-200:])
        prob, fix = explain(text, row.get("nodes", "<节点>"))
        issues.append((row.get("nodes") or row.get("group", "?"),
                       f"[{row.get('item', '')}/{row.get('reason_code', '')}] {prob}", fix))
    return issues


def from_envcheck(step, terminal_text, rc, out_dir, since, dry_run=False):
    """Returns (status, results, issues, details) for one prepare_nodes.sh envcheck step."""
    text = strip_ansi(terminal_text)
    results, issues, details = [], [], []
    result = re.search(r"^RESULT\s+(\S+)", text, re.M)
    nodes = re.search(r"^NODES\s+(.*)$", text, re.M)
    if result:
        results.append(f"hcu-envcheck RESULT={result.group(1)}" + (f"，{nodes.group(1)}" if nodes else ""))
    if dry_run:
        results.append("预演模式：只打印将执行的命令，未改动任何节点")
        return "预演", results, [], details

    if step == "idle":
        # idle-check exits non-zero whenever any node isn't both reachable and fully
        # idle; busy cards are normal cluster churn, not a step failure, so don't let
        # the generic fallback below flag them (its "failed" substring match fires on
        # the SUMMARY line's own "busy_or_failed" field name). Only a genuine ssh/
        # detection error (caught by diagnose()) should count as a real failure here.
        m = re.search(r"^\[SUMMARY\]\s+total=(\d+)\s+idle=(\d+)\s+busy_or_failed=(\d+)", text, re.M)
        if m:
            total, idle_n, bad_n = (int(x) for x in m.groups())
            results.append(f"共 {total} 个节点，空闲 {idle_n} 个，占用/不可用 {bad_n} 个（详见随后的节点筛查）")
            d = diagnose(text)
            if d:
                issues.append(("-", f"{d[0]}：`{d[2]}`", d[1]))
                return "失败", results, issues, details
            return ("成功" if bad_n == 0 else "部分可用"), results, issues, details

    log = re.search(r"^LOG\s+(\S+)", text, re.M)
    blocks = _blocks(log.group(1)) if log and Path(log.group(1)).is_file() else {}
    if log:
        details.append(log.group(1))
    for line in text.splitlines():
        m = GROUP_RE.match(line.strip())
        if not m:
            continue
        level, who, rest = m.groups()
        kv = dict(KV_RE.findall(rest))
        if level == "PASS":
            results.append(f"通过 stage={kv.get('stage', '-')}：{who}")
            continue
        if level not in ("FAIL", "ERROR"):
            continue
        reason = kv.get("reason", rest)
        members = list(blocks) if who == "ALL" else who.split(",")
        per_node = [n for n in members if n in blocks and blocks[n][0] != "PASS"]
        for n in per_node:
            prob, fix = explain(blocks[n][1], n)
            issues.append((n, f"stage={kv.get('stage', '-')} {reason}；{prob}", fix))
        if not per_node:
            prob, fix = explain(reason, who)
            issues.append((who, f"{kv.get('code', level)}：{prob}", fix))
        if step in ("create", "recreate") and kv.get("stage") == "preflight":
            issues.append(("全部节点", "预检有节点失败，hcu-envcheck 不会在任何节点建容器",
                           "先 `idle` 刷新 hostfile（排除不通节点），再重跑 `create`"))
    if step in ("check", "rccl"):
        sd = _stage_dir(out_dir, "rccl" if step == "rccl" else "platform", since) or \
            _stage_dir(out_dir, "resource", since) or _stage_dir(out_dir, "*", since)
        if sd:
            details.append(sd / "summary.md")
            issues += _failures(sd)
    if rc != 0 and not issues:
        prob, fix = explain(text)
        issues.append(("-", f"退出码 {rc}；{prob}", fix))
    seen, uniq = set(), []
    for i in issues:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    if uniq and step in NEXT_STEP:
        uniq.append(("下一步", "-", NEXT_STEP[step]))
    if blocks:  # per-node final state from lifecycle-details.log
        final = {"create": "create", "recreate": "create"}.get(step)
        n_ok = sum(1 for st, _, sg in blocks.values() if st == "PASS" and (not final or sg != "preflight"))
        results.append(f"节点最终结果：成功 {n_ok}，失败 {len(blocks) - n_ok}")
        status = benchlog_status(rc, n_ok, len(blocks) - n_ok)
    else:
        status = "成功" if rc == 0 and not issues else "失败"
    return status, results, uniq, details


def run_logged(runs, a):
    """exec 模式：写「开始」→ 执行命令（输出照常打印）→ 按退出码写结果。Returns the command's exit code."""
    argv = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not argv:
        raise SystemExit("exec 需要在 -- 之后给出命令")
    cmd = argv[0] if len(argv) == 1 else shlex.join(argv)
    run = ["bash", "-c", argv[0]] if len(argv) == 1 else argv
    with Step(runs, a.step, a.plan, cmd=a.cmd or cmd, run_dir=a.run_dir, target=a.target) as st:
        tail = []
        proc = subprocess.Popen(run, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail = (tail + [line.rstrip("\n")])[-200:]
        rc = proc.wait()
        shown = [f"`{short(l)}`" for l in tail if l.strip()][-5:]
        results = [f"退出码 {rc}"] + ([f"输出：{l}" for l in shown] if shown else ["无输出"])
        issues = [] if rc == 0 else [(a.target or "-", *explain("\n".join(tail)))]
        st.end("成功" if rc == 0 else "失败", results, issues, a.detail)
    return rc


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("begin")
    b.add_argument("--plan", action="append", default=[], help="将执行的操作，可多次")
    b.add_argument("--hosts", type=Path)
    x = sub.add_parser("exec", help="AI 执行一条命令：执行前写「开始」，执行后按退出码写结果；返回命令的退出码")
    x.add_argument("--plan", action="append", default=[], help="为什么做、做什么，可多次")
    x.add_argument("--detail", action="append", default=[])
    x.add_argument("command", nargs=argparse.REMAINDER, help="-- 之后的命令；只有一个参数时用 bash -c 执行")
    e = sub.add_parser("envcheck")
    e.add_argument("--terminal", type=Path, required=True)
    e.add_argument("--rc", type=int, required=True)
    e.add_argument("--since", type=float, default=0)
    e.add_argument("--hosts", type=Path)
    e.add_argument("--dry-run", action="store_true")
    n = sub.add_parser("note")
    n.add_argument("--status", required=True)
    n.add_argument("--result", action="append", default=[])
    n.add_argument("--issue", action="append", default=[], help="对象|问题|处理方案")
    n.add_argument("--detail", action="append", default=[])
    n.add_argument("--since", type=float, default=0, help="开始时间（epoch 秒），用于计算耗时")
    for p in (b, e, n, x):
        p.add_argument("--site", type=Path, required=True)
        p.add_argument("--step", default="ai", help="步骤名（STEP_TITLES 的键或自由文本），AI 操作用默认的 ai")
        p.add_argument("--target", default="", help="这一步做的事，接在标题后面")
        p.add_argument("--cmd", default="")
        p.add_argument("--run-dir", type=Path, help="测试相关时给出，同时写 <run_dir>/run_log.md")
    a = ap.parse_args()
    site = load_json(a.site)
    runs = Path(site["runs_dir"])
    if a.mode == "begin":
        plan = list(a.plan)
        if a.hosts and a.hosts.is_file():
            hosts = read_hosts(a.hosts)
            plan.insert(0, f"目标节点 {len(hosts)} 个（{a.hosts.name}）：{', '.join(hosts)}")
        print(begin(runs, a.step, a.cmd, plan, a.run_dir, a.target), end="")
        return
    if a.mode == "exec":
        raise SystemExit(run_logged(runs, a))
    if a.mode == "envcheck":
        status, results, issues, details = from_envcheck(
            a.step, a.terminal.read_text(errors="replace"), a.rc, runs / "envcheck", a.since, a.dry_run)
        if a.hosts and a.hosts.is_file():
            hosts = read_hosts(a.hosts)
            results.insert(0, f"目标节点 {len(hosts)} 个（{a.hosts.name}）：{', '.join(hosts)}")
        details.append(a.terminal)
    else:
        status, results, details = a.status, a.result, a.detail
        issues = [tuple((i.split("|", 2) + ["", ""])[:3]) for i in a.issue]
    elapsed = time.time() - a.since if a.since else None
    print(write(runs, a.step, a.cmd, status, results, issues, details, a.run_dir, a.target, elapsed), end="")


if __name__ == "__main__":
    main()
