#!/bin/bash
# Prepare benchmark nodes with hcu-envcheck and produce <runs_dir>/healthy_hosts.
#
# Usage: prepare_nodes.sh <site.json> <step> [--yes] [extra hcu-cluster-run args...]
#
# Steps (run in order; each prints what hcu-cluster-run reports):
#   idle      idle-check on hostfile.all (read-only), then screen
#   screen    classify the newest idle-check run -> hostfile (ssh ok), idle_hosts, node_status.md
#   status    container-status for the bench container
#   create    container-create   (dry-run unless --yes)
#   recreate  container-recreate (dry-run unless --yes; destroys the old container)
#   check     platform,resource inside containers on idle_hosts
#   rccl      single-node RCCL per idle host (group-size 1; needs site.json's rccl_baseline
#             or a command-line --baseline-file, else every collective fails in prepare phase)
#   rccl2     two-node RCCL: pairs idle hosts in order (group-size 2) to test cross-node
#             RDMA/RoCE + container ssh before attempting real multi-node training; any odd
#             leftover host runs alone with group-size 1 (same baseline/env-script as rccl)
#   healthy   write healthy_hosts = idle_hosts minus failed nodes of the given check/rccl runs
#
# Containers are managed on every reachable node (hostfile); benchmarks only use idle_hosts.
#
# create/recreate/delete are never executed without --yes; confirm with the user first.
# Every step appends two Chinese entries to <runs_dir>/bench_log.md: 「开始」 before it runs
# (command, what will be done) and the result after it (status, problems, fixes, elapsed);
# raw terminal output is kept in <runs_dir>/envcheck/terminal/.
set -euo pipefail

SITE=${1:?用法: prepare_nodes.sh <site.json> <步骤> [--yes] [其他参数...]}
STEP=${2:?缺少步骤参数: idle|screen|status|create|recreate|check|rccl|rccl2|healthy}
shift 2
YES=0
EXTRA=()
for a in "$@"; do
    if [[ $a == --yes ]]; then YES=1; else EXTRA+=("$a"); fi
done

j() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'d': d}))" "$SITE" "$1"; }
ROOT=$(j 'd["frameworks_root"]')
RUNS=$(j 'd["runs_dir"]')
HOSTFILE=$(j 'd["hostfile"]')
GPN=$(j 'd["gpus_per_node"]')
CNAME=$(j 'd["container"]["name"]')
IMAGE=$(j 'd["container"]["image"]')
PORT=$(j 'd["container"]["ssh_port"]')
mapfile -t MOUNTS < <(j '"\n".join(d["container"]["mounts"])')
# Cluster env script sourced inside the container by check/rccl; differs per cluster.
ENV_SCRIPT=$(j 'd.get("env_script", "")')
# RCCL performance baseline (np=8 busbw thresholds) used by the rccl step; without
# it rccl_perf_test.sh fails every collective in its prepare phase.
RCCL_BASELINE=$(j 'd.get("rccl_baseline", "")')
# Optional: host dir copied (docker cp, not a mount) into each container's
# /root/.ssh after create/recreate. Default /root/.ssh needs a root keypair
# already in authorized_keys on every node; pointing this at e.g. a
# shared-home .ssh avoids that per-node setup.
SSH_DIR=$(j 'd.get("ssh_dir", "")')

HERE=$(cd "$(dirname "$0")" && pwd)
COMMON=$(cd "${HERE}/../../common" && pwd)
# idle scans the full node list so nodes that recover are picked up again.
ALL=${RUNS}/hostfile.all
[[ -f ${ALL} ]] || ALL=${HOSTFILE}
ENVCHECK=${ROOT}/cluster-manager-das/hcu-envcheck
[[ -z ${SSH_DIR} ]] || export HCU_SSH_DIR=${SSH_DIR}
RUN=(bash "${ENVCHECK}/bin/hcu-cluster-run" per-node-container)
OUT=${RUNS}/envcheck
mkdir -p "${OUT}"

VOLS=()
for m in "${MOUNTS[@]}"; do VOLS+=(-v "$m"); done
# hcu-envcheck must see itself at the same path inside the container.
VOLS+=(-v "${ENVCHECK}:${ENVCHECK}")
# So is the env script; mount its directory (not the file) so edits stay visible.
ENV_DIR=""
ENV_ARGS=()
if [[ -n ${ENV_SCRIPT} ]]; then
    [[ -f ${ENV_SCRIPT} ]] || { echo "site.json 的 env_script 不存在: ${ENV_SCRIPT}"; exit 2; }
    ENV_DIR=$(dirname "${ENV_SCRIPT}")
    VOLS+=(-v "${ENV_DIR}:${ENV_DIR}:ro")
    # An explicit --env-script on the command line wins over site.json.
    [[ " ${EXTRA[*]} " == *" --env-script "* ]] || ENV_ARGS=(--env-script "${ENV_SCRIPT}")
fi
# Same for the RCCL baseline: rccl_perf_test.sh reads it inside the container,
# so its directory must be mounted at create/recreate time too.
RCCL_ARGS=()
if [[ -n ${RCCL_BASELINE} ]]; then
    [[ -f ${RCCL_BASELINE} ]] || { echo "site.json 的 rccl_baseline 不存在: ${RCCL_BASELINE}"; exit 2; }
    BASELINE_DIR=$(dirname "${RCCL_BASELINE}")
    [[ ${BASELINE_DIR} == "${ENV_DIR}" ]] || VOLS+=(-v "${BASELINE_DIR}:${BASELINE_DIR}:ro")
    # An explicit --baseline-file on the command line wins over site.json.
    [[ " ${EXTRA[*]} " == *" --baseline-file "* ]] || RCCL_ARGS=(--baseline-file "${RCCL_BASELINE}")
fi
# rccl_perf_test.sh does its own idle re-check right before launching and refuses
# to run on busy cards; we already picked idle_hosts via `idle`/`screen`, so skip
# its redundant check. There's no opt-out flag on their side, so cards that went
# busy in the gap between `idle` and `rccl` will just run instead of failing loud.
[[ " ${EXTRA[*]} " == *" --skip-idle-check "* ]] || RCCL_ARGS+=(--skip-idle-check)
# Default per-item x group budget is 300s; give slower/larger runs more headroom.
[[ " ${EXTRA[*]} " == *" --timeout "* ]] || RCCL_ARGS+=(--timeout 600)

TERM_DIR=${OUT}/terminal
mkdir -p "${TERM_DIR}"
YES_FLAG=""
(( YES )) && YES_FLAG=" --yes"
CMDLINE="prepare_nodes.sh ${SITE} ${STEP}${YES_FLAG}${EXTRA[*]:+ ${EXTRA[*]}}"

# log_begin <step> <hosts-file|-> <plan...>: the 「开始」 entry written before a step runs.
# OPEN_STEP stays set until the result entry is written; on_exit fills it in if the script dies first.
OPEN_STEP=""
STEP_T0=0
BAD=""
log_begin() {
    local step=$1 hosts=$2; shift 2
    local args=(--site "${SITE}" --step "${step}" --cmd "${CMDLINE}")
    [[ ${hosts} != - ]] && args+=(--hosts "${hosts}")
    for p in "$@"; do args+=(--plan "$p"); done
    python3 "${COMMON}/benchlog.py" begin "${args[@]}" | sed 's/^/[日志] /'
    OPEN_STEP=${step}
    STEP_T0=$(date +%s)
}
on_exit() {
    local rc=$?
    [[ -n ${BAD} ]] && rm -f "${BAD}"
    [[ -n ${OPEN_STEP} ]] || return 0
    local why="未知原因"
    case ${rc} in
        130) why="rc=130，被 Ctrl-C 中断" ;;
        1) why="rc=1，脚本内某条命令失败（常见于 \`healthy\` 现算 export-plan 时底层 hcu-cluster-run 调用失败，比如 ssh/网络抖动）" ;;
        *) why="rc=${rc}" ;;
    esac
    python3 "${COMMON}/benchlog.py" note --site "${SITE}" --step "${OPEN_STEP}" --cmd "${CMDLINE}" \
        --since "${STEP_T0}" --status 异常 --result "脚本在这一步中途退出（${why}），没有写入结果" \
        --issue "-|脚本中途退出（${why}）|看上面的终端输出定位原因；没有输出就在同一条命令前加 bash -x 重跑一次看具体卡在哪条命令" | sed 's/^/[日志] /'
}
trap on_exit EXIT

declare -A DESC=(
    [idle]="idle-check：检查每个节点能否 ssh、卡是否空闲（只读，不改动节点）"
    [status]="container-status：查看容器 ${CNAME} 是否存在、是否在运行"
    [create]="container-create：在每个节点建容器 ${CNAME}（镜像 ${IMAGE}，sshd 端口 ${PORT}）"
    [recreate]="container-recreate：删除旧容器 ${CNAME} 并重建（镜像 ${IMAGE}，sshd 端口 ${PORT}）"
    [check]="platform,resource：在容器 ${CNAME} 内做平台和资源检查（环境脚本 ${ENV_SCRIPT:-未设置}）"
    [rccl]="rccl：每个节点单机 RCCL 测试（group-size 1，每节点 ${GPN} 卡，环境脚本 ${ENV_SCRIPT:-未设置}，基线 ${RCCL_BASELINE:-未设置}）"
    [rccl2]="rccl2：每 2 个节点一组双机 RCCL 测试（group-size 2，每节点 ${GPN} 卡，容器间 ssh 端口 ${PORT}，环境脚本 ${ENV_SCRIPT:-未设置}，基线 ${RCCL_BASELINE:-未设置}）；节点数为奇数时最后一个单独跑单机 RCCL（group-size 1）"
)

# logged <hosts-file> <hcu-cluster-run args...>: write 「开始」, run and tee the output, then write the result.
# Keeps the hcu-envcheck exit code.
logged() {
    local hosts=$1; shift
    local term rc dry=() plan=("${DESC[${STEP}]:-hcu-cluster-run $1}")
    term=${TERM_DIR}/$(date +%Y%m%d_%H%M%S)_${STEP}.log
    if [[ " $* " == *" --dry-run "* ]]; then
        dry=(--dry-run)
        plan+=("预演模式：只打印将执行的命令，不改动任何节点")
    fi
    plan+=("终端输出保存到 ${term}")
    log_begin "${STEP}" "${hosts}" "${plan[@]}"
    set +e
    "${RUN[@]}" "$@" 2>&1 | tee "${term}"
    rc=${PIPESTATUS[0]}
    set -e
    python3 "${COMMON}/benchlog.py" envcheck --site "${SITE}" --step "${STEP}" --rc "${rc}" \
        --terminal "${term}" --since "${STEP_T0}" --hosts "${hosts}" --cmd "${CMDLINE}" "${dry[@]}" \
        | sed 's/^/[日志] /'
    OPEN_STEP=""
    return "${rc}"
}

lifecycle() {
    local op=$1
    if (( YES )); then
        logged "${HOSTFILE}" "$op" -f "${HOSTFILE}" --container "${CNAME}" -i "${IMAGE}" --port "${PORT}" \
            "${VOLS[@]}" -o "${OUT}" --yes "${EXTRA[@]}"
    else
        echo ">>> 预演模式，不会实际执行；确认无误后加 --yes 重新运行"
        logged "${HOSTFILE}" "$op" -f "${HOSTFILE}" --container "${CNAME}" -i "${IMAGE}" --port "${PORT}" \
            "${VOLS[@]}" --dry-run "${EXTRA[@]}"
    fi
}

IDLE=${RUNS}/idle_hosts
# check/rccl/healthy only make sense on idle nodes (busy cards fail the precheck).
need_idle() {
    [[ -s ${IDLE} ]] && return
    log_begin "${STEP}" - "检查 ${IDLE}（${STEP} 只在空闲节点上执行）" >/dev/null
    echo "${IDLE} 不存在或为空，先运行 idle"
    python3 "${COMMON}/benchlog.py" note --site "${SITE}" --step "${STEP}" --cmd "${CMDLINE}" --status 未执行 \
        --since "${STEP_T0}" --issue "idle_hosts|文件不存在或为空|先运行 \`prepare_nodes.sh ${SITE} idle\`" >/dev/null
    OPEN_STEP=""
    exit 2
}

case ${STEP} in
    idle)
        # idle-check exits non-zero when any node is busy or unreachable; screen reports it.
        logged "${ALL}" idle-check -f "${ALL}" -o "${OUT}" "${EXTRA[@]}" || true
        python3 "${HERE}/screen_nodes.py" --site "${SITE}"
        ;;
    screen)   python3 "${HERE}/screen_nodes.py" --site "${SITE}" "${EXTRA[@]}" ;;
    status)   logged "${HOSTFILE}" container-status -f "${HOSTFILE}" --container "${CNAME}" -i "${IMAGE}" "${EXTRA[@]}" ;;
    create)   lifecycle container-create ;;
    recreate) lifecycle container-recreate ;;
    check)    need_idle; logged "${IDLE}" platform,resource -f "${IDLE}" --container "${CNAME}" -i "${IMAGE}" \
                  -o "${OUT}" "${ENV_ARGS[@]}" "${EXTRA[@]}" ;;
    rccl)     need_idle; logged "${IDLE}" rccl -f "${IDLE}" --container "${CNAME}" -i "${IMAGE}" \
                  --group-size 1 --gpus-per-node "${GPN}" -o "${OUT}" "${ENV_ARGS[@]}" "${RCCL_ARGS[@]}" "${EXTRA[@]}" ;;
    rccl2)
        # Pair idle_hosts in order for 2-node RCCL (cross-node RDMA/RoCE + container ssh);
        # a hostfile with an odd tail node gets that node tested alone at group-size 1,
        # same baseline/env-script/skip-idle-check as the single-node `rccl` step.
        need_idle
        mapfile -t RCCL2_NODES < <(awk 'NF{print $1}' "${IDLE}")
        RCCL2_N=${#RCCL2_NODES[@]}
        RCCL2_PAIRED_N=$(( RCCL2_N / 2 * 2 ))
        if (( RCCL2_PAIRED_N >= 2 )); then
            RCCL2_PAIRED=${OUT}/rccl2_paired_hosts
            printf '%s\n' "${RCCL2_NODES[@]:0:${RCCL2_PAIRED_N}}" > "${RCCL2_PAIRED}"
            logged "${RCCL2_PAIRED}" rccl -f "${RCCL2_PAIRED}" --container "${CNAME}" -i "${IMAGE}" \
                --group-size 2 --gpus-per-node "${GPN}" --container-ssh-port "${PORT}" \
                -o "${OUT}" "${ENV_ARGS[@]}" "${RCCL_ARGS[@]}" "${EXTRA[@]}"
        fi
        if (( RCCL2_N % 2 )); then
            RCCL2_LEFT=${OUT}/rccl2_leftover_host
            printf '%s\n' "${RCCL2_NODES[-1]}" > "${RCCL2_LEFT}"
            logged "${RCCL2_LEFT}" rccl -f "${RCCL2_LEFT}" --container "${CNAME}" -i "${IMAGE}" \
                --group-size 1 --gpus-per-node "${GPN}" -o "${OUT}" "${ENV_ARGS[@]}" "${RCCL_ARGS[@]}" "${EXTRA[@]}"
        fi
        ;;
    healthy)
        # Pass the verify_results run dirs to aggregate, e.g. healthy <dir1> <dir2>.
        need_idle
        log_begin healthy "${IDLE}" "汇总 check/rccl 结果目录：${EXTRA[*]:-（未传入）}" \
            "没有导出过的目录先用 hcu-cluster-run --report-only 导出 failed-nodes.hostfile" \
            "写 ${RUNS}/healthy_hosts = idle_hosts 去掉失败节点"
        BAD=$(mktemp)
        for run_dir in "${EXTRA[@]}"; do
            exp=${OUT}/export-$(basename "${run_dir}")
            op=$(basename "${run_dir}" | sed -E 's/^per-node-container_([^_]+)_stages_.*/\1/')
            [[ -d ${exp} ]] || "${RUN[@]}" "${op}" -f "${IDLE}" --container "${CNAME}" -i "${IMAGE}" \
                --report-only "${run_dir}" --export-plan "${exp}"
            [[ -f ${exp}/failed-nodes.hostfile ]] && awk '{print $1}' "${exp}/failed-nodes.hostfile" >> "${BAD}"
        done
        grep -vxF -f <(sort -u "${BAD}"; echo '__none__') <(awk 'NF{print $1}' "${RUNS}/idle_hosts") > "${RUNS}/healthy_hosts" || true
        echo "健康节点: $(wc -l < "${RUNS}/healthy_hosts") / $(awk 'NF' "${RUNS}/idle_hosts" | wc -l)（空闲节点），已写入 ${RUNS}/healthy_hosts"
        sort -u "${BAD}" | sed 's/^/  已排除: /'
        NOTE=(--status "$( [[ -s ${RUNS}/healthy_hosts ]] && echo 成功 || echo 失败)"
              --result "空闲节点 $(awk 'NF' "${IDLE}" | wc -l) 个，健康 $(wc -l < "${RUNS}/healthy_hosts") 个：$(paste -sd, "${RUNS}/healthy_hosts")"
              --detail "${RUNS}/healthy_hosts")
        (( ${#EXTRA[@]} )) || NOTE+=(--issue "-|没有传入 check/rccl 结果目录，未排除任何节点|healthy 后面跟 check、rccl 的 per-node-container_*_stages_* 目录")
        while read -r h; do
            NOTE+=(--issue "${h}|check/rccl 未通过，已排除|原因见上面 check/rccl 步骤的日志；修复后重跑 check/rccl 再 healthy")
        done < <(sort -u "${BAD}")
        [[ -s ${RUNS}/healthy_hosts ]] || NOTE+=(--issue "全部节点|没有健康节点，无法生成测试列表|按 check/rccl 日志处理后重跑")
        for run_dir in "${EXTRA[@]}"; do NOTE+=(--detail "${run_dir}"); done
        python3 "${COMMON}/benchlog.py" note --site "${SITE}" --step healthy --cmd "${CMDLINE}" \
            --since "${STEP_T0}" "${NOTE[@]}" | sed 's/^/[日志] /'
        OPEN_STEP=""
        ;;
    *) echo "未知步骤: ${STEP}（可选: idle|screen|status|create|recreate|check|rccl|rccl2|healthy）"; exit 2 ;;
esac
