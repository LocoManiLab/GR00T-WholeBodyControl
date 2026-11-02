#!/usr/bin/env bash
# robot_deploy.sh （机器人侧）
# 包装现有 gear_sonic_deploy/deploy.sh，把【主机】的地址作为 --zmq-host 传入，
# 让 C++ 部署端连到运行 PICO streamer 的主机。网络方式不限：Tailscale / CPE 局域网 / 网线均可。
# 不修改 deploy.sh 本身，只做参数拼接。
#
# 用法：
#   bash remote_teleop/robot_deploy.sh --host <HOST_IP> real   # 显式指定主机 IP（推荐）
#   bash remote_teleop/robot_deploy.sh real                        # 用默认主机地址
#   bash remote_teleop/robot_deploy.sh real --cp policy/checkpoints/custom/model
#   OPERATOR_HOST=100.x.y.z bash remote_teleop/robot_deploy.sh real
#   bash remote_teleop/robot_deploy.sh <HOST_IP> real               # 第1参数即主机地址（旧式写法，仍兼容）
#
# 主机地址解析优先级：--host 参数 > 第1个看起来像地址的参数 > 环境变量 OPERATOR_HOST > default: operator-host (Tailscale MagicDNS name)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="$REPO_ROOT/gear_sonic_deploy"

# Default operator-host address: its Tailscale MagicDNS name (see remote_teleop/README.md).
OPERATOR_HOST="${OPERATOR_HOST:-operator-host}"

# 解析 --host 参数，其余参数原样透传给 deploy.sh
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            [[ $# -ge 2 ]] || { echo "[ERROR] --host 需要一个 IP/主机名参数" >&2; exit 1; }
            OPERATOR_HOST="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}

# 兼容旧式写法：若第一个参数不是 deploy.sh 的已知关键字/选项，则把它当作主机地址
if [[ $# -ge 1 && "$1" != "sim" && "$1" != "real" && "$1" != -* ]]; then
    OPERATOR_HOST="$1"
    shift
fi

echo "[INFO] 主机(streamer) 地址 --zmq-host = $OPERATOR_HOST"

# 连通性快速自检（非致命）：直接探测 streamer 的 TCP 端口，任何网络方式都适用
if timeout 3 bash -c "echo > /dev/tcp/$OPERATOR_HOST/5556" 2>/dev/null; then
    echo "[OK] $OPERATOR_HOST:5556 可连通（streamer 在线）"
else
    echo "[WARN] 连不上 $OPERATOR_HOST:5556，请确认主机已启动 host_streamer.sh 且 IP 正确"
fi

cd "$DEPLOY_DIR"
# setup_env.sh 是上游部署脚本，设计为在非 nounset 环境下 source；
# 这里临时关闭 -u 再 source，避免它引用未定义的 CMAKE_PREFIX_PATH 等变量时触发 unbound variable。
# shellcheck disable=SC1091
set +u
source scripts/setup_env.sh
set -u

# 默认 target 为 real（若用户没传 sim/real/IP，则补 real）
HAS_TARGET=false
for a in "$@"; do
    [[ "$a" == "sim" || "$a" == "real" ]] && HAS_TARGET=true
done

if $HAS_TARGET; then
    exec ./deploy.sh --input-type zmq_manager --zmq-host "$OPERATOR_HOST" "$@"
else
    exec ./deploy.sh --input-type zmq_manager --zmq-host "$OPERATOR_HOST" "$@" real
fi
