#!/usr/bin/env bash
# robot_camera.sh （机器人侧，视频源）
# 在 .venv_camera 中启动相机服务端（ZMQ PUB，bind tcp://*:PORT）。
# 绑定在所有接口上，因此天然在 tailscale0 上可达，主机 viewer 可经 Tailscale 连入。
# 不修改任何仓库代码，仅调用 gear_sonic/camera/composed_camera.py。
#
# 用法：
#   bash remote_teleop/robot_camera.sh                 # 默认端口 5555
#   bash remote_teleop/robot_camera.sh --port 5555 --ego_view_camera oak
#   端口及相机相关参数原样透传给 composed_camera.py（tyro CLI）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv_camera/bin/activate

echo "[INFO] 启动相机服务端 (ZMQ PUB, bind tcp://*:5555 默认) …"
echo "[INFO] 主机侧用 host_viewer.sh 连本机的 Tailscale 名/IP 即可看到画面。"

exec python gear_sonic/camera/composed_camera.py --run-as-server "$@"
