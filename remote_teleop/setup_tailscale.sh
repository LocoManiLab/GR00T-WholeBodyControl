#!/usr/bin/env bash
# setup_tailscale.sh
# 在机器人(Jetson/arm64)或本地主机(x86_64)上安装并加入 Tailscale 虚拟局域网。
# 幂等：已安装则跳过安装，直接 tailscale up。
#
# 用法：
#   # 机器人（无头，用 auth key）：
#   TS_AUTHKEY=tskey-xxxx bash remote_teleop/setup_tailscale.sh --hostname g1-robot
#
#   # 主机（交互浏览器登录）：
#   bash remote_teleop/setup_tailscale.sh --hostname operator-host
#
#   # 也可显式传 key：
#   bash remote_teleop/setup_tailscale.sh --hostname g1-robot --authkey tskey-xxxx

set -euo pipefail

HOSTNAME_ARG=""
AUTHKEY="${TS_AUTHKEY:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hostname)
            [[ $# -ge 2 ]] || { echo "[ERROR] --hostname 需要一个参数" >&2; exit 1; }
            HOSTNAME_ARG="$2"; shift 2 ;;
        --authkey)
            [[ $# -ge 2 ]] || { echo "[ERROR] --authkey 需要一个参数" >&2; exit 1; }
            AUTHKEY="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,16p' "$0"; exit 0 ;;
        *)
            echo "[ERROR] 未知参数: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$HOSTNAME_ARG" ]]; then
    echo "[ERROR] 必须用 --hostname 指定本机在 tailnet 中的名字（如 g1-robot / operator-host）" >&2
    exit 1
fi

ARCH="$(uname -m)"
echo "[OK] 架构: $ARCH"

# ── 1. 安装 Tailscale（官方安装器自动识别 arm64 / x86_64）─────────────────────
if ! command -v tailscale &>/dev/null; then
    echo "[INFO] 未检测到 tailscale，使用官方安装器安装 …"
    curl -fsSL https://tailscale.com/install.sh | sh
else
    echo "[OK] tailscale 已安装: $(tailscale version | head -1)"
fi

# 确保后台守护进程在运行（官方安装器通常会用 systemd 自动拉起）
if command -v systemctl &>/dev/null; then
    sudo systemctl enable --now tailscaled 2>/dev/null || true
fi

# ── 2. 加入 tailnet ───────────────────────────────────────────────────────────
UP_ARGS=(--hostname "$HOSTNAME_ARG" --accept-dns=true)
if [[ -n "$AUTHKEY" ]]; then
    echo "[INFO] 使用 auth key 无头入网 …"
    UP_ARGS+=(--authkey "$AUTHKEY")
else
    echo "[INFO] 未提供 auth key —— 将打印登录 URL，请在浏览器中授权 …"
fi

sudo tailscale up "${UP_ARGS[@]}"

# ── 3. 防火墙：放行 tailscale0 入站，避免对端连不上 5556 / 5555 ──────────────
if command -v ufw &>/dev/null && sudo ufw status 2>/dev/null | grep -q "Status: active"; then
    echo "[INFO] 检测到 ufw 已启用，放行 tailscale0 入站流量 …"
    sudo ufw allow in on tailscale0 || echo "[WARN] ufw 放行失败，请手动检查"
fi

# ── 4. 打印本机信息，方便记录对端地址 ─────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Tailscale 已就绪"
echo "  本机 tailnet 名 (MagicDNS): $HOSTNAME_ARG"
echo "  本机 Tailscale IP:          $(tailscale ip -4 2>/dev/null | head -1)"
echo "============================================================"
echo "当前 tailnet 节点："
tailscale status || true
echo ""
echo "提示：在对端用上面的【名字】或【100.x IP】作为连接地址。"
