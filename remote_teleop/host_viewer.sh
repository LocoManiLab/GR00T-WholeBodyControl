#!/usr/bin/env bash
# host_viewer.sh （主机侧，视频回传）
# 在 .venv_camera 中启动相机 viewer（ZMQ SUB），连机器人相机服务端。
# 网络方式不限：Tailscale / CPE 局域网 / 网线均可。
# 不修改任何仓库代码，仅调用 gear_sonic/scripts/run_camera_viewer.py。
#
# 用法：
#   bash remote_teleop/host_viewer.sh --robot <ROBOT_IP>    # 显式指定机器人 IP（推荐）
#   bash remote_teleop/host_viewer.sh                        # 用默认机器人地址
#   bash remote_teleop/host_viewer.sh 100.x.y.z              # 第1参数即机器人地址（旧式写法，仍兼容）
#   ROBOT_HOST=100.x.y.z bash remote_teleop/host_viewer.sh
#
# 机器人地址解析优先级：--robot 参数 > 第1参数 > 环境变量 ROBOT_HOST > default: g1-robot (Tailscale MagicDNS name)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Default robot address: the robot's Tailscale MagicDNS name (see remote_teleop/README.md).
ROBOT_HOST="${ROBOT_HOST:-g1-robot}"

# 解析 --robot 参数，其余参数原样透传给 run_camera_viewer.py
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot)
            [[ $# -ge 2 ]] || { echo "[ERROR] --robot 需要一个 IP/主机名参数" >&2; exit 1; }
            ROBOT_HOST="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}

# 兼容旧式写法：第1个非选项参数当作机器人地址
if [[ $# -ge 1 && "$1" != -* ]]; then
    ROBOT_HOST="$1"; shift
fi

echo "[INFO] 连接机器人相机 --camera-host = $ROBOT_HOST :5555"

# 连通性快速自检（非致命）：直接探测相机服务端的 TCP 端口，任何网络方式都适用
if timeout 3 bash -c "echo > /dev/tcp/$ROBOT_HOST/5555" 2>/dev/null; then
    echo "[OK] $ROBOT_HOST:5555 可连通（相机服务端在线）"
else
    echo "[WARN] 连不上 $ROBOT_HOST:5555，请确认机器人已启动 robot_camera.sh 且 IP 正确"
fi

# shellcheck disable=SC1091
source .venv_camera/bin/activate

exec python gear_sonic/scripts/run_camera_viewer.py \
    --camera-host "$ROBOT_HOST" --camera-port 5555 "$@"
