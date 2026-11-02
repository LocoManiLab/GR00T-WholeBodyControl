#!/usr/bin/env bash
# host_streamer.sh （主机侧，PICO → ZMQ）
# 在 .venv_teleop 中启动 PICO streamer（ZMQ PUB，bind tcp://*:5556）。
# PICO 仍通过【本地 WiFi】连本机，不走公网：把下面打印的 LAN IP 填进 PICO 的 XRoboToolKit app。
# 机器人 robot_deploy.sh 则通过 Tailscale 连本机（用打印的 Tailscale 名/IP）。
# 不修改任何仓库代码，仅调用 pico_manager_thread_server.py。
#
# 用法：
#   bash remote_teleop/host_streamer.sh                       # 无可视化
#   bash remote_teleop/host_streamer.sh --vis_vr3pt --vis_smpl
#   额外参数原样透传给 pico_manager_thread_server.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 本机所有非回环 IPv4（含 CPE/局域网/Tailscale），ZMQ 绑定所有接口，机器人连哪个都行
ALL_IPS="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -vE '^(127\.|169\.254\.)' | grep -v '^$' || true)"
LAN_IP="$(echo "$ALL_IPS" | grep -vE '^100\.' | head -1 || true)"
TS_IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"

echo "============================================================"
echo "  PICO app 里填这个【本地 LAN IP】:   ${LAN_IP:-<未检测到，请手动用 ip addr 查看>}"
echo "  机器人 robot_deploy.sh --host <选下面一个能通的地址>:"
echo "$ALL_IPS" | sed 's/^/    /'
[[ -n "$TS_IP" ]] && echo "    （其中 $TS_IP 是 Tailscale，其余为本地/CPE 网段）"
echo "============================================================"

# shellcheck disable=SC1091
source .venv_teleop/bin/activate

echo "[INFO] 启动 PICO streamer (ZMQ PUB, bind tcp://*:5556) …"
exec python gear_sonic/scripts/pico_manager_thread_server.py --manager "$@"
