#!/usr/bin/env bash
# One-shot launcher for FastWAM inference as the policy backend
# (drop-in replacement for start_inference_va.sh which uses lingbot-va).
#
# Same client stack (gear_sonic) + same Isaac-GR00T ZMQ wire protocol, but the
# policy server is <FastWAM>/deployment/run_fastwam_server.py wrapping a fine-tuned
# FastWAM model.
#
# Usage:
#   bash start_inference_fastwam.sh [OPTIONS]
#
# Options:
#   -p, --prompt          Language prompt for the policy
#   -d, --run-dir         FastWAM training run dir (config.yaml + dataset_stats.json + checkpoints/)
#   -c, --checkpoint      weights/step_XXXXX.pt (default: newest in <run-dir>/checkpoints/weights)
#   -r, --fastwam-repo    Path to the FastWAM repo (its src/ is added to PYTHONPATH)
#   -m, --base-model-path Base Wan2.2-TI2V-5B dir (VAE+T5+tokenizer); overrides the run
#                         config's model_id path (use for all-local runs)
#       --action-space    motion_token | smpl  (default: motion_token)
#       --t5-cache DIR    Training-format text-embed cache dir (<run>/cache_t5,
#                         fetched by fetch_fastwam_ckpt.sh). T5 is NOT loaded;
#                         the prompt must exactly match a cached training prompt.
#       --deploy-checkpoint PATH
#                         SONIC deploy checkpoint prefix for the C++ deploy
#                         (e.g. policy/sonic_v1_1/model). Default: the
#                         low_latency preset (--low-latency).
#       --deploy-obs-config FILE
#                         Observation config for that checkpoint
#                         (e.g. policy/sonic_v1_1/observation_config.yaml)
#   -h, --help            Show this help message
#
# Environment overrides:
#   FW_CONDA_ENV        conda env with FastWAM installed            (default: fastwam)
#   POLICY_HOST         host the client connects to                 (default: localhost)
#   FW_PORT             ZMQ port for the policy server              (default: 5550)
#   SERVER_REMOTE       1 -> do NOT launch the server locally; just point the
#                       client at POLICY_HOST:FW_PORT (server started elsewhere,
#                       e.g. on the remote H100). (default: 0 -> launch locally)
#   NUM_INFERENCE_STEPS flow-matching denoising steps               (default: 10)
#   UPSAMPLE_FACTOR     predicted tokens are 100ms apart; x5 -> 50Hz (default: 5)
#   UPSAMPLE_MODE       interp | repeat                             (default: interp)
#   HANDS_MODE          hold | zeros  (FastWAM doesn't predict hands)(default: hold)
#   ACTION_SPACE        motion_token | smpl                         (default: motion_token)
#   ACTION_HORIZON_PRED predicted tokens per chunk (default: auto = num_frames-1 = 32)
#   EVICT_T5            1 -> free text encoder after encoding prompt (default: 1)
#   TEXT_CFG_SCALE      text classifier-free guidance scale         (default: 1.0)
#   SEED                inference seed (default: unset / random)
#   INFERENCE_RATE      client VLA forward-pass frequency (Hz)      (default: 2.5)
#   MAX_LATENCY_SKIP    client latency-compensation cap             (default: 8)
#   TOKEN_EMA_ALPHA     chunk-boundary EMA smoothing                (default: 0.4)
#   STARTUP_RAMP_HOLD   pubs to hold anchor (skip chunk-0 LF0)      (default: 20)
#   STARTUP_RAMP_STEPS  pubs to ramp anchor->model after hold       (default: 60)
#
# The client publishes one action per 50Hz tick, so ACTION_HORIZON (client) must
# equal the server's emitted chunk length = ACTION_HORIZON_PRED * UPSAMPLE_FACTOR.
# This script computes it automatically.
#
# Examples:
#   bash start_inference_fastwam.sh -p "Push the bottle into the drawer and squat and close drawer"
#   SERVER_REMOTE=1 POLICY_HOST=<POLICY_SERVER_IP> bash start_inference_fastwam.sh -p "..."
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
PROMPT="Push the bottle into the drawer and squat and close drawer"
FW_RUN_DIR="${FW_RUN_DIR:-}"   # required: pass -d/--run-dir or set FW_RUN_DIR
FW_CHECKPOINT="${FW_CHECKPOINT:-}"
FW_REPO="${FW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/FastWAM}"
FW_BASE_MODEL_PATH="${FW_BASE_MODEL_PATH:-}"
FW_PROMPT_EMBEDS="${FW_PROMPT_EMBEDS:-}"   # precomputed prompt-embeds .pt (skips T5; use on 24GB GPUs)
FW_T5_CACHE="${FW_T5_CACHE:-}"             # training-format cache_t5 dir (skips T5; prompt = cache key)
# SONIC controller the C++ deploy tracks the policy with. Unset -> --low-latency
# preset (policy/low_latency). Same pair launch_data_collection.py takes.
DEPLOY_CHECKPOINT="${DEPLOY_CHECKPOINT:-}"
DEPLOY_OBS_CONFIG="${DEPLOY_OBS_CONFIG:-}"

FW_CONDA_ENV="${FW_CONDA_ENV:-fastwam}"
POLICY_HOST="${POLICY_HOST:-localhost}"
FW_PORT="${FW_PORT:-5550}"
SERVER_REMOTE="${SERVER_REMOTE:-0}"

NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
UPSAMPLE_FACTOR="${UPSAMPLE_FACTOR:-5}"
UPSAMPLE_MODE="${UPSAMPLE_MODE:-interp}"
HANDS_MODE="${HANDS_MODE:-hold}"
ACTION_SPACE="${ACTION_SPACE:-motion_token}"
ACTION_HORIZON_PRED="${ACTION_HORIZON_PRED:-32}"   # num_frames(33) - 1
EVICT_T5="${EVICT_T5:-1}"
TEXT_CFG_SCALE="${TEXT_CFG_SCALE:-1.0}"
SEED="${SEED:-}"

# Client-side cadence (same knobs as start_inference_va.sh)
INFERENCE_RATE="${INFERENCE_RATE:-2.5}"
MAX_LATENCY_SKIP="${MAX_LATENCY_SKIP:-8}"
TOKEN_EMA_ALPHA="${TOKEN_EMA_ALPHA:-0.4}"
STARTUP_RAMP_HOLD="${STARTUP_RAMP_HOLD:-20}"
STARTUP_RAMP_STEPS="${STARTUP_RAMP_STEPS:-60}"
export STARTUP_RAMP_HOLD STARTUP_RAMP_STEPS

# Emitted chunk length the client consumes at 50Hz.
ACTION_HORIZON="$(( ACTION_HORIZON_PRED * UPSAMPLE_FACTOR ))"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--prompt)        PROMPT="$2"; shift 2 ;;
        -d|--run-dir)       FW_RUN_DIR="$2"; shift 2 ;;
        -c|--checkpoint)    FW_CHECKPOINT="$2"; shift 2 ;;
        -r|--fastwam-repo)  FW_REPO="$2"; shift 2 ;;
        -m|--base-model-path) FW_BASE_MODEL_PATH="$2"; shift 2 ;;
        --action-space)     ACTION_SPACE="$2"; shift 2 ;;
        --t5-cache)         FW_T5_CACHE="$2"; shift 2 ;;
        --deploy-checkpoint) DEPLOY_CHECKPOINT="$2"; shift 2 ;;
        --deploy-obs-config) DEPLOY_OBS_CONFIG="$2"; shift 2 ;;
        -h|--help)          sed -n '2,57p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)                  echo "Unknown option: $1  (use --help)"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The policy server lives in the FastWAM repo now (it only imports fastwam.*);
# this launcher keeps the parts that are genuinely this repo's: cameras,
# tmux, the inference main loop.
SERVER_PY="${FW_REPO}/deployment/run_fastwam_server.py"
SESSION="sonic_inference_fastwam"
cd "$REPO_ROOT"

[[ -n "${FW_RUN_DIR}" ]] || { echo "ERROR: FastWAM run dir not set (use -d/--run-dir or FW_RUN_DIR)"; exit 1; }
[[ -f "${SERVER_PY}" ]] || { echo "ERROR: server not found: ${SERVER_PY}"; exit 1; }

# Resolve `conda activate` for non-interactive shells.
CONDA_PROFILE="${CONDA_PROFILE:-}"
if [[ -z "${CONDA_PROFILE}" ]]; then
    for c in "$HOME/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/anaconda3/etc/profile.d/conda.sh" \
             "/opt/conda/etc/profile.d/conda.sh"; do
        if [[ -f "$c" ]]; then CONDA_PROFILE="$c"; break; fi
    done
fi

echo "================================================================"
echo "  SONIC Inference Launcher (FastWAM backend)"
echo "  Prompt           : ${PROMPT}"
echo "  Run dir          : ${FW_RUN_DIR}"
echo "  Checkpoint       : ${FW_CHECKPOINT:-<newest in run-dir>}"
echo "  FastWAM repo     : ${FW_REPO}"
echo "  Base model path  : ${FW_BASE_MODEL_PATH:-<from run config>}"
echo "  Conda env        : ${FW_CONDA_ENV}"
echo "  Policy host:port : ${POLICY_HOST}:${FW_PORT}"
echo "  Server location  : $([[ "${SERVER_REMOTE}" == "1" ]] && echo "REMOTE (not launched here)" || echo "LOCAL (tmux window)")"
echo "  Inference steps  : ${NUM_INFERENCE_STEPS}"
echo "  Action space     : ${ACTION_SPACE}"
echo "  SONIC controller : ${DEPLOY_CHECKPOINT:-policy/low_latency/model (--low-latency preset)}"
echo "  Chunk            : ${ACTION_HORIZON_PRED} tokens @10Hz x${UPSAMPLE_FACTOR} (${UPSAMPLE_MODE}) = ${ACTION_HORIZON} actions @50Hz"
echo "  Hands mode       : ${HANDS_MODE} (fallback when the checkpoint does not predict hands)"
if [[ -n "${FW_PROMPT_EMBEDS}" ]]; then
echo "  Prompt embeds    : ${FW_PROMPT_EMBEDS} (T5 NOT loaded -> ~13GB VRAM, fits 24GB)"
elif [[ -n "${FW_T5_CACHE}" ]]; then
echo "  T5 cache         : ${FW_T5_CACHE} (T5 NOT loaded -> ~13GB VRAM; prompt must match a cached training prompt)"
else
echo "  T5 evict         : ${EVICT_T5} (T5 loaded once at startup to encode the prompt, then freed)"
fi
echo "================================================================"
echo ""

# Kill any pre-existing session
tmux kill-session -t "$SESSION" 2>/dev/null || true

# ── Launch the gear_sonic client-side stack ───────────────────────────────────
# Explicit deploy assets replace the low_latency preset (launch_inference.py
# would otherwise also insist the low_latency files exist).
CONTROLLER_FLAGS=(--low-latency)
if [[ -n "${DEPLOY_CHECKPOINT}" || -n "${DEPLOY_OBS_CONFIG}" ]]; then
    CONTROLLER_FLAGS=()
    [[ -n "${DEPLOY_CHECKPOINT}" ]] && CONTROLLER_FLAGS+=(--deploy-checkpoint "${DEPLOY_CHECKPOINT}")
    [[ -n "${DEPLOY_OBS_CONFIG}" ]]  && CONTROLLER_FLAGS+=(--deploy-obs-config "${DEPLOY_OBS_CONFIG}")
fi
python gear_sonic/scripts/launch_inference.py \
    --prompt "${PROMPT}" \
    --camera-host 192.168.123.164 \
    "${CONTROLLER_FLAGS[@]}" \
    --policy-host "${POLICY_HOST}" \
    --policy-port "${FW_PORT}" \
    --action-horizon "${ACTION_HORIZON}" \
    --action-space "${ACTION_SPACE}" \
    --max-latency-skip "${MAX_LATENCY_SKIP}" \
    --token-ema-alpha "${TOKEN_EMA_ALPHA}" \
    --inference-rate "${INFERENCE_RATE}" &

echo "Waiting for tmux session..."
for i in $(seq 1 30); do
    if tmux has-session -t "sonic_inference" 2>/dev/null; then
        tmux rename-session -t "sonic_inference" "$SESSION" 2>/dev/null || true
        echo "  Session '${SESSION}' ready (${i}s)."; break
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "  Session '${SESSION}' ready (${i}s)."; break
    fi
    sleep 1
done

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session was not created within 30 s. Aborting."; exit 1
fi
sleep 8

# ── Window: fastwam-server (only when launching locally) ──────────────────────
if [[ "${SERVER_REMOTE}" != "1" ]]; then
    [[ -f "${CONDA_PROFILE}" ]] || {
        echo "ERROR: cannot locate conda.sh; set CONDA_PROFILE=/path/to/conda.sh"; exit 1; }

    CKPT_FLAG=""
    [[ -n "${FW_CHECKPOINT}" ]] && CKPT_FLAG="--checkpoint ${FW_CHECKPOINT}"
    BASE_FLAG=""
    [[ -n "${FW_BASE_MODEL_PATH}" ]] && BASE_FLAG="--base-model-path ${FW_BASE_MODEL_PATH}"
    EMBEDS_FLAG=""
    [[ -n "${FW_PROMPT_EMBEDS}" ]] && EMBEDS_FLAG="--prompt-embeds ${FW_PROMPT_EMBEDS}"
    T5CACHE_FLAG=""
    [[ -n "${FW_T5_CACHE}" ]] && T5CACHE_FLAG="--t5-cache ${FW_T5_CACHE}"
    EVICT_FLAG=""
    # --evict-t5 only matters when T5 is loaded (i.e. no precomputed embeds/cache).
    [[ "${EVICT_T5}" == "1" && -z "${FW_PROMPT_EMBEDS}" && -z "${FW_T5_CACHE}" ]] && EVICT_FLAG="--evict-t5"
    SEED_FLAG=""
    [[ -n "${SEED}" ]] && SEED_FLAG="--seed ${SEED}"

    PROMPT_ESC="${PROMPT//\\/\\\\}"; PROMPT_ESC="${PROMPT_ESC//\"/\\\"}"

    tmux new-window -t "$SESSION" -n "fastwam-server"
    tmux send-keys -t "${SESSION}:fastwam-server" \
        "source ${CONDA_PROFILE} && conda activate ${FW_CONDA_ENV} && \
cd ${REPO_ROOT} && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u ${SERVER_PY} \
    --run-dir ${FW_RUN_DIR} \
    --fastwam-repo ${FW_REPO} \
    ${CKPT_FLAG} ${BASE_FLAG} ${EMBEDS_FLAG} ${T5CACHE_FLAG} \
    --host 0.0.0.0 --port ${FW_PORT} \
    --num-inference-steps ${NUM_INFERENCE_STEPS} \
    --action-horizon ${ACTION_HORIZON_PRED} \
    --action-space ${ACTION_SPACE} \
    --upsample-factor ${UPSAMPLE_FACTOR} --upsample-mode ${UPSAMPLE_MODE} \
    --hands-mode ${HANDS_MODE} \
    --text-cfg-scale ${TEXT_CFG_SCALE} \
    ${SEED_FLAG} ${EVICT_FLAG} \
    --initial-prompt \"${PROMPT_ESC}\"" C-m
else
    echo "SERVER_REMOTE=1: start the server yourself on the model host, e.g.:"
    echo "  python ${SERVER_PY} --run-dir ${FW_RUN_DIR} --fastwam-repo ${FW_REPO} \\"
    echo "    --host 0.0.0.0 --port ${FW_PORT} --num-inference-steps ${NUM_INFERENCE_STEPS} \\"
    echo "    --action-horizon ${ACTION_HORIZON_PRED} --action-space ${ACTION_SPACE} --upsample-factor ${UPSAMPLE_FACTOR} \\"
    echo "    --upsample-mode ${UPSAMPLE_MODE} --hands-mode ${HANDS_MODE} ${EVICT_FLAG:+--evict-t5} \\"
    echo "    --initial-prompt \"${PROMPT}\""
    echo "Then make sure POLICY_HOST=${POLICY_HOST} is reachable from this machine."
fi

# ── Window: camera-viewer ─────────────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "camera-viewer"
tmux send-keys -t "${SESSION}:camera-viewer" \
    "cd ${REPO_ROOT} && source .venv_data_collection/bin/activate && \
python gear_sonic/scripts/run_camera_viewer.py \
--camera-host 192.168.123.164 --camera-port 5555" C-m

tmux select-window -t "${SESSION}:inference"

echo ""
echo "================================================================"
echo "  ${SESSION} session ready"
echo "    inference      — VLA inference stack (4 panes)"
[[ "${SERVER_REMOTE}" != "1" ]] && echo "    fastwam-server — FastWAM policy server"
echo "    camera-viewer  — live camera feed"
echo ""
echo "  Switch windows : Ctrl+b  n / p     Detach : Ctrl+b d"
echo "  Tail server log: tmux capture-pane -p -t ${SESSION}:fastwam-server"
echo "================================================================"
echo ""

tmux attach -t "$SESSION"
