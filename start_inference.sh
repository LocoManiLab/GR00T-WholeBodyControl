#!/usr/bin/env bash
# One-shot launcher: VLA inference stack + GR00T policy server + camera viewer.
#
# Usage:
#   bash start_inference.sh [OPTIONS]
#
# Options:
#   -p, --prompt      Language prompt for the VLA policy
#                     (default: "pick up the apple")
#   -m, --model-path  Path to the fine-tuned GR00T model checkpoint
#                     (default: /path/to/your/finetuned_model)
#   -t, --trt-engine-path
#                     Path to the TensorRT engines directory
#   -a, --action-horizon
#                     Action horizon — must match the value used during training
#                     (default: 64, or ACTION_HORIZON env var)
#   --low-latency     Use the low-latency SONIC deploy policy
#                     (policy/low_latency/model + observation_config.yaml)
#   --deploy-checkpoint PATH
#                     SONIC deploy checkpoint prefix passed to the C++ deploy
#                     (e.g. policy/sonic_v1_1/model). Overrides --low-latency.
#   --deploy-obs-config FILE
#                     Observation config for that checkpoint
#                     (e.g. policy/sonic_v1_1/observation_config.yaml)
#   -h, --help        Show this help message
#
# Examples:
#   bash start_inference.sh
#   bash start_inference.sh --prompt "place the cup on the table"
#   bash start_inference.sh --model-path ~/models/gr00t_finetune_v2
#   bash start_inference.sh -p "open the drawer" -m /data/checkpoints/gr00t_v3
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
PROMPT="pick up the apple"
MODEL_PATH="/path/to/your/finetuned_model"
TRT_ENGINE_PATH="${TRT_ENGINE_PATH:-}"   # optional: path to TRT engines dir
TRT_MODE="${TRT_MODE:-n17_full_pipeline}"
# Action horizon: MUST match the value used during training (e.g. 16, 40, 64).
# Mismatch causes the robot to freeze on the last action of the short window
# and stutter for the entire inference latency before each new chunk.
ACTION_HORIZON="${ACTION_HORIZON:-64}"
MAX_LATENCY_SKIP="${MAX_LATENCY_SKIP:-3}"
TOKEN_EMA_ALPHA="${TOKEN_EMA_ALPHA:-1.0}"
LOW_LATENCY="${LOW_LATENCY:-0}"
# Explicit SONIC deploy assets (env or flags). Unset -> the C++ deploy's own
# default (run_deploy_cu12.sh: policy/low_latency), or --low-latency's preset.
DEPLOY_CHECKPOINT="${DEPLOY_CHECKPOINT:-}"
DEPLOY_OBS_CONFIG="${DEPLOY_OBS_CONFIG:-}"
# Inference rate: default 2.5Hz (PyTorch); auto-raised to 10Hz when TRT is enabled
# Override with INFERENCE_RATE=5.0 env var
INFERENCE_RATE="${INFERENCE_RATE:-}"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--prompt)
            PROMPT="$2"; shift 2 ;;
        -m|--model-path)
            MODEL_PATH="$2"; shift 2 ;;
        -t|--trt-engine-path)
            TRT_ENGINE_PATH="$2"; shift 2 ;;
        -a|--action-horizon)
            ACTION_HORIZON="$2"; shift 2 ;;
        --low-latency)
            LOW_LATENCY=1; shift ;;
        --no-low-latency)
            LOW_LATENCY=0; shift ;;
        --deploy-checkpoint)
            DEPLOY_CHECKPOINT="$2"; shift 2 ;;
        --deploy-obs-config)
            DEPLOY_OBS_CONFIG="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "Unknown option: $1  (use --help for usage)"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="sonic_inference"
# Machine-specific roots: override via env vars (LocoManiLab injects them from env.local.yaml;
# see src/locomanilab/models/gr00t.py deploy). Defaults are conventional install locations.
GROOT_REPO_ROOT="${GROOT_REPO_ROOT:-$REPO_ROOT/../Isaac-GR00T}"
GROOT_TRT_ROOT="${GROOT_TRT_ROOT:-$HOME/.local/TensorRT-10.9.0.34}"
GROOT_CUDA12_ROOT="${GROOT_CUDA12_ROOT:-$HOME/miniconda3/envs/gr00t_deploy_cu12}"

cd "$REPO_ROOT"

echo "================================================================"
echo "  SONIC Inference Launcher"
echo "  Prompt     : ${PROMPT}"
echo "  Model path : ${MODEL_PATH}"
echo "  TRT engine : ${TRT_ENGINE_PATH:-disabled}"
echo "  Horizon    : ${ACTION_HORIZON}"
echo "  Low latency: ${LOW_LATENCY}"
echo "  Deploy ckpt: ${DEPLOY_CHECKPOINT:-<deploy default>}"
echo "  Obs config : ${DEPLOY_OBS_CONFIG:-<deploy default>}"
echo "================================================================"
echo ""

# Kill any pre-existing session
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Run the Python launcher in background.
# It creates and populates the tmux session, then calls "tmux attach" which
# exits silently with no controlling terminal — the session stays alive.
# Auto-set inference rate: 10Hz with TRT, 2.5Hz without (override via INFERENCE_RATE=)
if [[ -z "${INFERENCE_RATE}" ]]; then
    [[ -n "${TRT_ENGINE_PATH}" ]] && INFERENCE_RATE="10.0" || INFERENCE_RATE="2.5"
fi

LOW_LATENCY_FLAG=()
case "${LOW_LATENCY}" in
    1|true|TRUE|yes|YES|on|ON)
        LOW_LATENCY_FLAG=(--low-latency) ;;
    0|false|FALSE|no|NO|off|OFF)
        ;;
    *)
        echo "ERROR: LOW_LATENCY must be 0/1, true/false, yes/no, or on/off."
        exit 1 ;;
esac

DEPLOY_FLAGS=()
[[ -n "${DEPLOY_CHECKPOINT}" ]] && DEPLOY_FLAGS+=(--deploy-checkpoint "${DEPLOY_CHECKPOINT}")
[[ -n "${DEPLOY_OBS_CONFIG}" ]]  && DEPLOY_FLAGS+=(--deploy-obs-config "${DEPLOY_OBS_CONFIG}")

python gear_sonic/scripts/launch_inference.py \
    --prompt "${PROMPT}" \
    --camera-host 192.168.123.164 \
    --deploy-output-type all \
    --action-horizon "${ACTION_HORIZON}" \
    --max-latency-skip "${MAX_LATENCY_SKIP}" \
    --token-ema-alpha "${TOKEN_EMA_ALPHA}" \
    --inference-rate "${INFERENCE_RATE}" \
    "${LOW_LATENCY_FLAG[@]}" \
    "${DEPLOY_FLAGS[@]}" &

# Wait for the tmux session to appear (up to 30 s)
echo "Waiting for tmux session '${SESSION}'..."
for i in $(seq 1 30); do
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "  Session '${SESSION}' ready (${i}s)."
        break
    fi
    sleep 1
done

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: session '${SESSION}' was not created within 30 s. Aborting."
    exit 1
fi

# Give the launcher time to finish populating all four panes
sleep 8

# ── Window: gr00t-server ──────────────────────────────────────────────────────
TRT_FLAGS=""
if [[ -n "${TRT_ENGINE_PATH}" ]]; then
    TRT_FLAGS="--trt-engine-path ${TRT_ENGINE_PATH} --trt-mode ${TRT_MODE}"
fi

tmux new-window -t "$SESSION" -n "gr00t-server"
tmux send-keys -t "${SESSION}:gr00t-server" \
    "cd ${GROOT_REPO_ROOT} && env LD_LIBRARY_PATH=${GROOT_TRT_ROOT}/lib:${GROOT_CUDA12_ROOT}/lib:/opt/ros/humble/lib:\${LD_LIBRARY_PATH:-} \
uv run python gr00t/eval/run_gr00t_server.py \
--model-path ${MODEL_PATH} \
--embodiment-tag UNITREE_G1_SONIC \
--device cuda:0 \
--port 5550 \
${TRT_FLAGS}" C-m

# ── Window: camera-viewer ─────────────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "camera-viewer"
tmux send-keys -t "${SESSION}:camera-viewer" \
    "cd ${REPO_ROOT} && source .venv_data_collection/bin/activate && \
python gear_sonic/scripts/run_camera_viewer.py \
--camera-host 192.168.123.164 \
--camera-port 5555" C-m

# Focus the main inference window
tmux select-window -t "${SESSION}:inference"

echo ""
echo "================================================================"
echo "  sonic_inference session ready"
echo ""
echo "  inference     — VLA inference stack (4 panes)"
echo "  gr00t-server  — Isaac-GR00T policy server"
echo "  camera-viewer — live camera feed"
echo ""
echo "  Switch windows : Ctrl+b  n / p"
echo "  Detach         : Ctrl+b  d"
echo "  Kill session   : Ctrl+\\"
echo "================================================================"
echo ""

tmux attach -t "$SESSION"
