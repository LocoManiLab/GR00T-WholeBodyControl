#!/bin/bash
# run_deploy_cu12.sh — 绕过 deploy.sh 直接运行已编译的 g1_deploy_onnx_ref
# 原因: deploy.sh 内部 source scripts/setup_env.sh 会把 CUDA 探测回系统 CUDA13.2,
#       导致 CUDA error 35。本脚本强制使用 CUDA12(匹配驱动570) + 本地 TensorRT10。
#
# 命令行【完全兼容 deploy.sh】,可作为 deploy.sh 的直接替代:
#   ./run_deploy_cu12.sh [--input-type T] [--zmq-host H] [--cp PATH] [--obs-config F]
#                        [--planner F] [--motion-data D] [--output-type T] [sim|real|<iface>]
# 示例:
#   ./run_deploy_cu12.sh --input-type zmq_manager --zmq-host localhost real
#   ./run_deploy_cu12.sh --input-type zmq_manager --zmq-host localhost sim   # sim→lo 回环
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── CUDA12 + TensorRT 运行时环境(关键:覆盖系统 CUDA13) ──────────────────────
GR00T_DEPLOY_CUDA="${GR00T_DEPLOY_CUDA:-$HOME/miniconda3/envs/gr00t_deploy_cu12}"
GR00T_DEPLOY_TRT="${GR00T_DEPLOY_TRT:-$HOME/.local/TensorRT-10.9.0.34}"
# /opt/ros/humble/lib: unitree_sdk2 链接了 ROS2 typesupport 库,运行时需要
export LD_LIBRARY_PATH="$GR00T_DEPLOY_TRT/lib:$HOME/.local/onnxruntime/lib:$GR00T_DEPLOY_CUDA/lib:/opt/ros/humble/lib:${LD_LIBRARY_PATH:-}"

# ── 默认值(对齐 deploy.sh) ───────────────────────────────────────────────────
CHECKPOINT="policy/low_latency/model"      # decoder/encoder 由此派生
OBS_CONFIG="policy/low_latency/observation_config.yaml"
PLANNER="planner/target_vel/V2/planner_sonic.onnx"
MOTION_DATA="reference/example/"
INPUT_TYPE="zmq_manager"
OUTPUT_TYPE=""
ZMQ_HOST="localhost"
MODE="real"

# ── 解析 deploy.sh 风格参数 ──────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --input-type)  INPUT_TYPE="$2"; shift 2 ;;
        --zmq-host)    ZMQ_HOST="$2"; shift 2 ;;
        --cp|--checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --obs-config)  OBS_CONFIG="$2"; shift 2 ;;
        --planner)     PLANNER="$2"; shift 2 ;;
        --motion-data) MOTION_DATA="$2"; shift 2 ;;
        --output-type) OUTPUT_TYPE="$2"; shift 2 ;;
        sim|real)      MODE="$1"; shift ;;
        *)             MODE="$1"; shift ;;   # 也允许直接传接口名(如 lo / enpXsY)
    esac
done

DECODER="${CHECKPOINT}_decoder.onnx"
ENCODER="${CHECKPOINT}_encoder.onnx"

# ── 模式 → DDS 网络接口 ──────────────────────────────────────────────────────
case "$MODE" in
    sim)  IFACE="lo" ;;                       # 仿真走回环
    real)
        IFACE=$(ip -4 addr show 2>/dev/null | awk '/inet 192\.168\.123\./{print $NF; exit}')
        if [ -z "$IFACE" ]; then
            echo "[ERROR] 未找到 192.168.123.x 接口(G1 内网)。请用网线连 G1,或显式指定接口名。" >&2
            exit 1
        fi
        echo "[OK] 自动探测到 G1 网络接口: $IFACE" ;;
    *)    IFACE="$MODE" ;;                     # 显式接口名
esac

echo "════════════════════════════════════════════════════"
echo "  g1_deploy_onnx_ref (CUDA12 runtime)"
echo "  Mode/Iface : $MODE -> $IFACE"
echo "  Decoder    : $DECODER"
echo "  Motion     : $MOTION_DATA"
echo "  Input/ZMQ  : $INPUT_TYPE @ $ZMQ_HOST"
echo "════════════════════════════════════════════════════"

# ── 安全确认(对齐 deploy.sh): 真机模式会让 G1 真正动起来,启动前等手动放行。
#    在 launch_data_collection.py 的 tmux pane 0 里,这会停住等你确认其余 pane
#    (相机/遥操)就绪后再按 Enter 放行。设 DEPLOY_AUTO_YES=1 可跳过(非交互场景)。
if [ "${DEPLOY_AUTO_YES:-0}" != "1" ]; then
    if [ "$MODE" = "real" ] || { [ "$MODE" != "sim" ] && [ "$IFACE" != "lo" ]; }; then
        echo -e "\033[1;33m⚠️  警告: 即将启动【真实机器人】控制系统 (接口 $IFACE)!\033[0m"
    else
        echo "📋 即将启动仿真/本机控制系统。"
    fi
    read -p "$(echo -e '\033[0;32m确认启动部署? [Y/n]: \033[0m')" _confirm
    if ! { [[ "$_confirm" =~ ^[Yy]$ ]] || [ -z "$_confirm" ]; }; then
        echo "已取消。"; exit 0
    fi
fi
echo "🚀 启动部署..."

CMD=( ./target/release/g1_deploy_onnx_ref "$IFACE" "$DECODER" "$MOTION_DATA"
      --obs-config "$OBS_CONFIG"
      --encoder-file "$ENCODER"
      --planner-file "$PLANNER"
      --input-type "$INPUT_TYPE"
      --zmq-host "$ZMQ_HOST" )
[ -n "$OUTPUT_TYPE" ] && CMD+=( --output-type "$OUTPUT_TYPE" )

exec "${CMD[@]}"
