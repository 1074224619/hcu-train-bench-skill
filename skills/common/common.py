"""Shared helpers for hcu-bench sub-skill scripts (stdlib only)."""
import json
import math
import os
import re
from pathlib import Path

ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+)\s*\|.*?"
    r"elapsed time per iteration \(ms\):\s*([\d.]+)"
    r"(?:.*?throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+))?"
    r"(?:.*?global batch size:\s*(\d+))?"
)

FRAMEWORK_SHORT = {"megatron-lm": "lm", "megatron-bridge": "bridge", "ms-swift": "swift"}

# `hy-smi --showmemuse --json` prints one block per card, e.g.
# {"card0":{"HCU memory use (%)":"31"},"card1":{"HCU memory use (%)":"0"}}; match the
# percentage inside each card's block regardless of exact key wording/card index.
MEMUSE_CARD_RE = re.compile(r'"(?:card|hcu)[^"]*"\s*:\s*\{[^}]*?"[^"]*"\s*:\s*"?([\d.]+)"?', re.I)
VRAM_POLL_INTERVAL_S = 5


def abspath_literal(path):
    """Make `path` absolute without dereferencing symlinks (unlike Path.resolve() /
    os.path.abspath(), which both end up calling the kernel's getcwd() and so follow
    any symlink in the chain).

    Some sandboxes keep the project's own working directory as a live symlink (e.g.
    the path this code is running from may itself be `.../sandbox-cwd3`, pointing at
    a different, rotating target). Containers are built with bind mounts resolved at
    create time against the literal configured path (site.json's runs_dir /
    frameworks_root, or whatever cwd a script was invoked from); if a later script
    re-resolves a relative path through os.getcwd(), it can silently land on a
    *different* real directory than the one actually mounted, producing "No such
    file or directory" inside the container even though the file exists on the
    login node. Joining against $PWD (bash's logical, non-dereferenced cwd, which
    is exported to child processes) instead keeps whatever literal path was used
    to invoke the command.
    """
    p = os.fspath(path)
    if not os.path.isabs(p):
        p = os.path.join(os.environ.get("PWD") or os.getcwd(), p)
    return Path(os.path.normpath(p))


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def read_hosts(hostfile):
    """One host per line; ignores blanks, comments and any 'slots=' suffix."""
    hosts = []
    for line in Path(hostfile).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            hosts.append(line.split()[0])
    return hosts


def lcm(a, b):
    return a * b // math.gcd(a, b)


def min_world_size(tp=1, pp=1, cp=1, ep=1, etp=1):
    """Smallest world size satisfying Megatron's dense (tp*pp*cp) and MoE (etp*ep*pp) grids."""
    return lcm(tp * pp * cp, etp * ep * pp)


def round_up(n, step):
    return int(math.ceil(n / step) * step)


def load_run(run_dir):
    run_dir = Path(run_dir)
    matrix = load_json(run_dir / "matrix.json")
    site = load_json(matrix["meta"]["site"])
    return run_dir, matrix, site


def find_case(matrix, case_id):
    for case in matrix["cases"]:
        if case["id"] == case_id:
            return case
    raise SystemExit(f"case not found: {case_id}")
