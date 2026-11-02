#!/usr/bin/env bash
# One-shot launcher for VLA inference using lingbot-va as the policy backend,
# with FDM-grounded inference (Algorithm 2 of arXiv 2601.21998v2) enabled.
#
# Identical to start_inference_va.sh in every respect EXCEPT:
#   - the policy server module is wan_va.gr00t_compat_server_fdm (not ...server)
#   - tmux session is named sonic_inference_va_fdm so it can coexist with the
#     non-FDM stack
#   - new optional env knob: SIMULATE_ASYNC_LAG (default 0) — set to 1 to drop
#     one chunk of real-obs grounding between FDM commit and PREDICT, mimicking
#     the async-lag scenario the paper's FDM was designed for. In a true sync
#     deployment FDM is mostly redundant with real-obs grounding; this knob
#     surfaces FDM's effect for A/B testing.
#
# Usage:
#   bash start_inference_va_fdm.sh [OPTIONS]
#
# All other options + env vars match start_inference_va.sh — see that file for
# details.
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
PROMPT="pick up the apple"
LVA_BASE_PATH="${LVA_BASE_PATH:-$HOME/lingbot-va-base}"
LVA_TRANSFORMER_PATH="${LVA_TRANSFORMER_PATH:-$HOME/lingbot-va/checkpoint_step_5000/transformer}"
LVA_REPO="${LVA_REPO:-$HOME/lingbot-va}"

LVA_CONDA_ENV="${LVA_CONDA_ENV:-lingbot-va}"
LVA_CONFIG="${LVA_CONFIG:-custom}"
LVA_PORT="${LVA_PORT:-5550}"
EVICT_T5="${EVICT_T5:-1}"
CLOSED_LOOP="${CLOSED_LOOP:-1}"
OFFLOAD_T5_ONLY="${OFFLOAD_T5_ONLY:-0}"
ENABLE_OFFLOAD="${ENABLE_OFFLOAD:-0}"
NO_KV_WARMUP="${NO_KV_WARMUP:-0}"
SIMULATE_ASYNC_LAG="${SIMULATE_ASYNC_LAG:-0}"

INFERENCE_RATE="${INFERENCE_RATE:-2.5}"
ACTION_HORIZON="${ACTION_HORIZON:-80}"
MAX_LATENCY_SKIP="${MAX_LATENCY_SKIP:-8}"
TOKEN_EMA_ALPHA="${TOKEN_EMA_ALPHA:-0.4}"
STARTUP_RAMP_HOLD="${STARTUP_RAMP_HOLD:-20}"
STARTUP_RAMP_STEPS="${STARTUP_RAMP_STEPS:-60}"
export STARTUP_RAMP_HOLD STARTUP_RAMP_STEPS
MASTER_PORT="${MASTER_PORT:-29501}"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--prompt)            PROMPT="$2"; shift 2 ;;
        -b|--base-model-path)   LVA_BASE_PATH="$2"; shift 2 ;;
        -t|--transformer-path)  LVA_TRANSFORMER_PATH="$2"; shift 2 ;;
        -r|--lingbot-va-repo)   LVA_REPO="$2"; shift 2 ;;
        --simulate-async-lag)   SIMULATE_ASYNC_LAG=1; shift ;;
        -h|--help)
            sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $1  (use --help)"; exit 1 ;;
    esac
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
[[ -d "${LVA_BASE_PATH}" ]] || { echo "ERROR: base-model-path not found: ${LVA_BASE_PATH}"; exit 1; }
[[ -d "${LVA_TRANSFORMER_PATH}" ]] || { echo "ERROR: transformer-path not found: ${LVA_TRANSFORMER_PATH}"; exit 1; }
[[ -d "${LVA_REPO}/wan_va" ]] || { echo "ERROR: lingbot-va repo missing wan_va/: ${LVA_REPO}"; exit 1; }
[[ -f "${LVA_REPO}/wan_va/gr00t_compat_server_fdm.py" ]] || {
    echo "ERROR: FDM adapter not found: ${LVA_REPO}/wan_va/gr00t_compat_server_fdm.py"
    echo "       (this should have been created alongside this launcher)"
    exit 1;
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="sonic_inference_va_fdm"
cd "$REPO_ROOT"

CONDA_PROFILE="${CONDA_PROFILE:-}"
if [[ -z "${CONDA_PROFILE}" ]]; then
    for c in "$HOME/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/anaconda3/etc/profile.d/conda.sh" \
             "/opt/conda/etc/profile.d/conda.sh"; do
        if [[ -f "$c" ]]; then CONDA_PROFILE="$c"; break; fi
    done
fi
[[ -f "${CONDA_PROFILE}" ]] || {
    echo "ERROR: cannot locate conda.sh; set CONDA_PROFILE=/path/to/conda.sh"; exit 1;
}

echo "================================================================"
echo "  SONIC Inference Launcher (lingbot-va + FDM grounding)"
echo "  Prompt                : ${PROMPT}"
echo "  Base model            : ${LVA_BASE_PATH}"
echo "  Transformer ckpt      : ${LVA_TRANSFORMER_PATH}"
echo "  Lingbot-va repo       : ${LVA_REPO}"
echo "  Conda env             : ${LVA_CONDA_ENV}"
echo "  Policy port           : ${LVA_PORT}"
echo "  Inference rate        : ${INFERENCE_RATE} Hz"
echo "  KV cache loop         : $([[ "${CLOSED_LOOP}" == "0" ]] && echo "OPEN (legacy)" || echo "CLOSED")"
echo "  Simulate async lag    : $([[ "${SIMULATE_ASYNC_LAG}" == "1" ]] && echo "ON (FDM grounding is effective)" || echo "OFF (sync; FDM mostly redundant with real obs)")"
if   [[ "${EVICT_T5}"        == "1" ]]; then OFFLOAD_DESC="T5 one-shot then freed; VAE+Transformer on GPU"
elif [[ "${OFFLOAD_T5_ONLY}" == "1" ]]; then OFFLOAD_DESC="T5 on CPU, VAE on GPU"
elif [[ "${ENABLE_OFFLOAD}"  == "1" ]]; then OFFLOAD_DESC="T5+VAE on CPU"
else                                         OFFLOAD_DESC="all on GPU"; fi
echo "  Memory mode           : ${OFFLOAD_DESC}"
echo "================================================================"
echo ""

tmux kill-session -t "$SESSION" 2>/dev/null || true

python gear_sonic/scripts/launch_inference.py \
    --prompt "${PROMPT}" \
    --camera-host 192.168.123.164 \
    --policy-host localhost \
    --policy-port "${LVA_PORT}" \
    --action-horizon "${ACTION_HORIZON}" \
    --max-latency-skip "${MAX_LATENCY_SKIP}" \
    --token-ema-alpha "${TOKEN_EMA_ALPHA}" \
    --inference-rate "${INFERENCE_RATE}" &

echo "Waiting for tmux session '${SESSION}'..."
for i in $(seq 1 30); do
    if tmux has-session -t "sonic_inference" 2>/dev/null; then
        tmux rename-session -t "sonic_inference" "$SESSION" 2>/dev/null || true
        echo "  Session '${SESSION}' ready (${i}s)."
        break
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "  Session '${SESSION}' ready (${i}s)."
        break
    fi
    sleep 1
done

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session was not created within 30 s. Aborting."
    exit 1
fi

sleep 8

OFFLOAD_FLAG=""
if   [[ "${EVICT_T5}"        == "1" ]]; then OFFLOAD_FLAG="--evict-t5"
elif [[ "${OFFLOAD_T5_ONLY}" == "1" ]]; then OFFLOAD_FLAG="--offload-t5-only"
elif [[ "${ENABLE_OFFLOAD}"  == "1" ]]; then OFFLOAD_FLAG="--enable-offload"
fi
KV_FLAG=""
[[ "${NO_KV_WARMUP}" == "1" ]]   && KV_FLAG="--no-kv-warmup"
CL_FLAG=""
[[ "${CLOSED_LOOP}"  == "0" ]]   && CL_FLAG="--no-closed-loop"
FDM_LAG_FLAG=""
[[ "${SIMULATE_ASYNC_LAG}" == "1" ]] && FDM_LAG_FLAG="--simulate-async-lag"

PROMPT_ESC="${PROMPT//\\/\\\\}"
PROMPT_ESC="${PROMPT_ESC//\"/\\\"}"

tmux new-window -t "$SESSION" -n "lva-server-fdm"
tmux send-keys -t "${SESSION}:lva-server-fdm" \
    "source ${CONDA_PROFILE} && conda activate ${LVA_CONDA_ENV} && \
cd ${LVA_REPO} && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT} \
python -u -m wan_va.gr00t_compat_server_fdm \
    --config-name ${LVA_CONFIG} \
    --base-model-path ${LVA_BASE_PATH} \
    --transformer-path ${LVA_TRANSFORMER_PATH} \
    --host 0.0.0.0 \
    --port ${LVA_PORT} \
    ${OFFLOAD_FLAG} ${KV_FLAG} ${CL_FLAG} ${FDM_LAG_FLAG} \
    --initial-prompt \"${PROMPT_ESC}\"" C-m

tmux new-window -t "$SESSION" -n "camera-viewer"
tmux send-keys -t "${SESSION}:camera-viewer" \
    "cd ${REPO_ROOT} && source .venv_data_collection/bin/activate && \
python gear_sonic/scripts/run_camera_viewer.py \
--camera-host 192.168.123.164 \
--camera-port 5555" C-m

tmux select-window -t "${SESSION}:inference"

echo ""
echo "================================================================"
echo "  ${SESSION} session ready"
echo ""
echo "  inference       — VLA inference stack (4 panes)"
echo "  lva-server-fdm  — lingbot-va policy server (FDM-grounded)"
echo "  camera-viewer   — live camera feed"
echo ""
echo "  Switch windows : Ctrl+b  n / p"
echo "  Detach         : Ctrl+b  d"
echo ""
echo "  Tail server log: tmux capture-pane -p -t ${SESSION}:lva-server-fdm"
echo "================================================================"
echo ""

tmux attach -t "$SESSION"
