#!/usr/bin/env python3
"""按一次运行记录（或手工输入）给出显存/切分调优建议。

Usage:
  tune_plan.py <run_dir>                    # 读 run_model.py 的 status.json / train.log / cmd.txt
  tune_plan.py --state failed --oom-at optimizer --vram-peak 97 --framework megatron-bridge
  tune_plan.py --state done --vram-peak 43

只读运行记录、不改任何配置；输出中文结论、按优先级排列的建议、参数入口和并行约束。
只依赖 Python 标准库。
"""
import argparse
import json
import re
import sys
from pathlib import Path

STATE_ZH = {"done": "成功", "failed": "失败", "timeout": "超时", "stopped": "中断"}
OOM_RE = re.compile(r"out of memory|OutOfMemoryError", re.I)
OOM_CONTEXTS = [
    (re.compile(r"get_grad_norm|optimizer\.step|clip_grads|grad_norm", re.I), "optimizer"),
    (re.compile(r"ParamAndGradBuffer|grad(?:ient)?[ _]buffer|\bDDP\b|distributed[^\n]*grad", re.I), "ddp"),
    (re.compile(r"activation|TransformerLayer|\battention\b|\bmlp\b|\bbackward\b|\bforward\b", re.I), "activation"),
]
OOM_ZH = {"optimizer": "优化器更新/梯度裁剪阶段（参数+梯度+优化器状态）",
          "ddp": "DDP 梯度缓冲阶段（参数量决定）",
          "activation": "激活值阶段（前向/反向计算）",
          "unknown": "未识别的堆栈阶段"}

KNOBS = {
    "megatron-bridge": [
        ("切分", "launcher 环境变量 TP / PP / EP / ETP / CP（minimax 支持；其他 launcher 视脚本而定）"),
        ("批次", "qwen35_vl: MICRO_BATCH_SIZE / GLOBAL_BATCH_SIZE / SEQ_LENGTH 环境变量；minimax 需改 recipe"),
        ("recompute", "recipe: cfg.model.recompute_granularity=selective|full；"
                      "cfg.model.recompute_modules=['moe_act']（MoE 激活）"),
        ("offload", "cfg.model.fine_grained_activation_offloading=true + cfg.model.offload_modules；"
                    "cfg.optimizer.optimizer_cpu_offload=true"),
        ("分布式优化器", "cfg.ddp.use_distributed_optimizer=true"),
        ("改 recipe 的位置", "frameworks/Megatron-bridge-das/hcu_megatron_bridge/.../recipes/<model>/<platform>/"),
    ],
    "megatron-lm": [
        ("切分/批次", "examples/<family>/train_*.sh 里的 TP/PP/CP/MBS/GBS/SEQ_LEN（写死，修改前需用户确认）"),
        ("recompute", "--recompute-granularity selective|full；--recompute-method block|uniform "
                      "--recompute-num-layers N；MoE 可加 --recompute-modules moe_act"),
        ("activation offload", "MCore 的 fine-grained activation offloading（flag 名以所用版本 --help 为准；需用户确认）"),
        ("offload", "--optimizer-cpu-offload --optimizer-offload-fraction 0.5"),
        ("分布式优化器", "--use-distributed-optimizer"),
    ],
}

CONSTRAINTS = [
    "world_size = TP × PP × CP × DP，必须等于节点数 × 每节点卡数",
    "EP 必须整除 DP；TP 必须整除 attention head 数；PP 不超过层数",
    "GBS 必须能被 MBS × DP 整除",
    "一次只改一项；先 dryrun 或 1 iter，再跑 10 iters 对比 time/iter 与显存峰值",
    "修改 frameworks/ 下的脚本或 recipe 前先让用户确认",
]


def detect_oom(text):
    """按堆栈帧判断 OOM 位置：optimizer / ddp / activation / unknown；没有 OOM 返回空串。"""
    if not OOM_RE.search(text):
        return ""
    frames = [l for l in text.splitlines() if re.search(r'File "[^"]+", line \d+', l)]
    ctx = "\n".join(frames) or text[-8000:]
    for rx, kind in OOM_CONTEXTS:
        if rx.search(ctx):
            return kind
    return "unknown"


def last_errors(text, n=3):
    lines = [l.strip() for l in text.splitlines()
             if re.search(r"(?i)error|exception|failed|killed|traceback|out of memory", l)]
    return lines[-n:]


def detect_framework(cmd):
    low = cmd.lower()
    if "megatron-bridge" in low:
        return "megatron-bridge"
    if "megatron-lm" in low:
        return "megatron-lm"
    return ""


def oom_recommendations(kind):
    if kind == "optimizer":
        return [
            "先加优化器 offload 并确认分布式优化器：LM 用 `--optimizer-cpu-offload "
            "--optimizer-offload-fraction 0.5`，bridge 用 `cfg.optimizer.optimizer_cpu_offload=true` "
            "+ `cfg.ddp.use_distributed_optimizer=true`。",
            "加大模型切分：PP/TP/EP 任一提高一档（PP≤层数，EP 整除 DP，TP 整除 head 数）；仍不够就减少层数。",
            "减小 MBS/SEQ_LEN 只降峰值，优化器状态常驻占用不变，放在最后。",
            "recompute 对这里通常无效（它只省激活值）；如果已经开着，可以关掉换吞吐。",
        ]
    if kind == "ddp":
        return [
            "梯度缓冲大小随参数量增长：优先加大 PP/TP/EP 或减少层数。",
            "开分布式优化器（LM `--use-distributed-optimizer`；bridge `cfg.ddp.use_distributed_optimizer=true`）"
            "降低每卡梯度/优化器冗余。",
            "recompute 帮助有限；减小 MBS 只影响激活峰值。",
        ]
    if kind == "activation":
        return [
            "先开 selective recompute；仍 OOM 再改 full；MoE 可同时加 `moe_act` 模块。",
            "开 activation offload（bridge：`cfg.model.fine_grained_activation_offloading=true` + "
            "`cfg.model.offload_modules`）。",
            "减小 MBS / SEQ_LEN，降低单次前向的激活峰值。",
            "最后再加大 PP/TP/EP（切分对激活和参数都有效，但会牺牲吞吐）。",
        ]
    return [
        "先看 result.md / train.log 里 OOM 的堆栈：optimizer/DDP 阶段按参数类处理（切分+优化器 offload），"
        "forward/backward 阶段按激活类处理（recompute/activation offload）。",
        "保底方案：加大 PP/TP/EP + 优化器 offload；激活类再叠加 recompute。",
        "也可用 `--oom-at optimizer|ddp|activation` 指定位置后重跑本工具，拿具体建议。",
    ]


def headroom_recommendations(peak):
    if peak < 40:
        return [
            "显存富余明显，可以换吞吐：若开了 recompute/offload，先逐步关闭（先关 activation offload，"
            "再关 recompute），观察峰值和 s/iter。",
            "切分降一档：优先 PP（如 2→1，需层数允许），其次 EP（8→4，需整除 DP），最后 TP（需整除 head 数）。",
            "GBS 不变时提高 MBS，减少 micro-batch 数和通信开销。",
        ]
    if peak < 55:
        return [
            "还有余量（目标 55–75%）：可先关 activation offload 或降一档切分，再观察峰值。",
            "每次只改一项；降切分后若峰值超过 75%，回退。",
        ]
    if peak <= 75:
        return ["显存处于合适区间（55–75%），保持当前切分。",
                "若要提速，优先在 GBS 约束内调大 MBS，而不是继续降切分。"]
    if peak < 90:
        return ["余量偏小：不要减切分；训练波动可能 OOM。",
                "建议加 recompute / activation offload，或加一档切分。"]
    return ["接近显存上限：先加 recompute/offload 或加一档切分，再跑长一点确认稳定。",
            "不要在这个峰值下减少切分。"]


def main():
    ap = argparse.ArgumentParser(description="按运行记录给出显存/切分调优建议（手工调优接口）")
    ap.add_argument("run_dir", nargs="?", type=Path, help="run_model.py 的输出目录")
    ap.add_argument("--state", choices=sorted(STATE_ZH), help="覆盖运行状态")
    ap.add_argument("--oom-at", choices=["auto", "optimizer", "ddp", "activation", "unknown", "none"],
                    help="OOM 位置；none=确认不是显存问题；默认 auto 从 train.log 识别")
    ap.add_argument("--vram-peak", type=float, help="显存峰值百分比（0–100；<=1 视为比例）")
    ap.add_argument("--framework", choices=["auto", "megatron-bridge", "megatron-lm"], default="auto")
    a = ap.parse_args()

    status, text, cmd = {}, "", ""
    if a.run_dir:
        if not a.run_dir.is_dir():
            ap.error(f"运行目录不存在：{a.run_dir}")
        sp = a.run_dir / "status.json"
        if sp.exists():
            status = json.loads(sp.read_text())
        lp = a.run_dir / "train.log"
        if lp.exists():
            text = lp.read_text(errors="replace")
        cp = a.run_dir / "cmd.txt"
        if cp.exists():
            cmd = cp.read_text(errors="replace")

    state = a.state or status.get("state")
    if not state and not a.vram_peak and a.oom_at in (None, "auto"):
        ap.error("请给出 run_dir，或至少传 --state/--vram-peak/--oom-at <具体位置|none> 之一（见 --help）")

    peaks = status.get("vram_peak_percent") or {}
    peak = a.vram_peak
    if peak is None and peaks:
        peak = max(peaks.values())
    if peak is not None and peak <= 1:
        peak = peak * 100

    if a.oom_at and a.oom_at != "auto":
        oom = "" if a.oom_at == "none" else a.oom_at
    else:
        oom = detect_oom(text)
        if not oom and state in ("failed", "timeout", "stopped"):
            oom = ""  # 没识别到 OOM，按非显存问题处理

    framework = "" if a.framework == "auto" else a.framework
    if not framework:
        framework = detect_framework(cmd)

    print("== 运行概况 ==")
    if state:
        print(f"状态：{state}（{STATE_ZH.get(state, state)}）")
    if status.get("last_iter") is not None:
        print(f"迭代：到第 {status['last_iter']} 步，用时 {status.get('duration_s', '?')}s")
    if peaks:
        detail = "，".join(f"{h}={v}%" for h, v in peaks.items())
        print(f"显存峰值：{detail}（取最高 {max(peaks.values()):.0f}%）")
        vals = list(peaks.values())
        if len(vals) > 1 and max(vals) - min(vals) >= 15:
            print(f"注意：节点间峰值差 {max(vals) - min(vals):.0f} 个百分点，"
                  "可能是 PP/EP 各 stage 不均衡或某张卡异常；用 hy-smi --showmemuse 逐卡对比")
    elif peak is not None:
        print(f"显存峰值：{peak:.0f}%（手工输入）")
    else:
        print("显存峰值：未知（run_model.py 会轮询 hy-smi --showmemuse 记录；也可用 --vram-peak 传入）")
    if framework:
        print(f"框架：{framework}")
    if oom:
        print(f"OOM 位置：{OOM_ZH.get(oom, oom)}")

    print("\n== 结论与建议（按优先级） ==")
    recs = []
    if oom:
        print(f"这是显存不足，且发生在{OOM_ZH.get(oom, oom)}；按下面的顺序调，先别急着加 recompute。")
        recs = oom_recommendations(oom)
    elif state == "failed":
        print("没有识别到 OOM，先按报错排查，不是显存调优问题。")
        errs = last_errors(text)
        if errs:
            print("末尾报错：")
            recs = [f"`{e}`" for e in errs]
    elif peak is not None:
        print(f"显存峰值 {peak:.0f}%。" + ("显存富余，可以适当减少切分/关重计算换吞吐。"
                                          if peak < 55 else ""))
        recs = headroom_recommendations(peak)
    else:
        print("信息不足：给出 --vram-peak 或 run_dir 才能判断该加还是该减切分。")

    for i, r in enumerate(recs, 1):
        print(f"{i}. {r}")

    print("\n== 参数入口 ==")
    frameworks = [framework] if framework in KNOBS else ["megatron-bridge", "megatron-lm"]
    for fw in frameworks:
        print(f"[{fw}]")
        for name, where in KNOBS[fw]:
            print(f"  {name:<10} {where}")

    print("\n== 并行约束与调试纪律 ==")
    for c in CONSTRAINTS:
        print(f"- {c}")
    print("\n本工具只读运行记录；改完配置后用 run_model.py 重跑，对比 result.md 里的 s/iter 和显存峰值。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
