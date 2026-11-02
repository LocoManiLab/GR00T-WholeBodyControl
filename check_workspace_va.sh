#!/usr/bin/env bash
# check_workspace_va.sh — verify the workspace is ready to launch
# start_inference_va.sh, without actually touching the robot.
#
# Light mode (default):
#   - file/dir existence
#   - .venv_inference / .venv_data_collection imports work
#   - lingbot-va conda env imports work
#   - launcher --help paths execute
#
# Heavy mode (--with-model):
#   additionally start the lingbot-va policy server, wait for it to bind
#   tcp://*:5550, send a ZMQ ping, then kill it. Loads ~11 GB into VRAM
#   and takes ~30s on a 4090.
#
# Usage:
#   bash check_workspace_va.sh
#   bash check_workspace_va.sh --with-model
#   bash check_workspace_va.sh --help
set -u   # NOT -e: we want to continue past individual failures and count

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
ISAAC_GROOT_SRC="$WORKSPACE_ROOT/Isaac-GR00T"
ISAAC_GROOT_LINK="$REPO_ROOT/.isaac_groot"

LINGBOT_VA_REPO="${LINGBOT_VA_REPO:-$HOME/lingbot-va}"
LINGBOT_VA_ENV="${LINGBOT_VA_ENV:-lingbot-va}"
LINGBOT_VA_BASE="${LINGBOT_VA_BASE:-$HOME/lingbot-va-base}"
LINGBOT_VA_TRANSFORMER="${LINGBOT_VA_TRANSFORMER:-$HOME/lingbot-va/checkpoint_step_5000/transformer}"

WITH_MODEL=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-model) WITH_MODEL=1; shift ;;
        --help|-h)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Colors ────────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    R='\033[31m'; G='\033[32m'; Y='\033[33m'; B='\033[34m'; C='\033[36m'; D='\033[0m'
else
    R=''; G=''; Y=''; B=''; C=''; D=''
fi

PASS=0
FAIL=0
WARN=0
FAILED_ITEMS=()

pass() { PASS=$((PASS + 1)); printf "  ${G}[PASS]${D} %s\n" "$1"; }
fail() { FAIL=$((FAIL + 1)); FAILED_ITEMS+=("$1"); printf "  ${R}[FAIL]${D} %s\n" "$1"; [[ $# -gt 1 ]] && printf "         ${D}%s${D}\n" "$2"; }
warn() { WARN=$((WARN + 1)); printf "  ${Y}[WARN]${D} %s\n" "$1"; }
section() { printf "\n${B}== %s ==${D}\n" "$1"; }

# Run a quiet check; pass desc + command
check_exists() {
    local desc="$1" path="$2"
    if [[ -e "$path" || -L "$path" ]]; then pass "$desc ($path)"
    else fail "$desc missing" "$path"; fi
}

# Run a Python snippet inside a venv and report pass/fail
check_py_in_venv() {
    local desc="$1" venv_python="$2" snippet="$3"
    if [[ ! -x "$venv_python" ]]; then
        fail "$desc — venv python not found" "$venv_python"
        return
    fi
    local out
    if out="$("$venv_python" -c "$snippet" 2>&1)"; then
        pass "$desc"
    else
        fail "$desc" "$(echo "$out" | tail -3)"
    fi
}

# Run a Python snippet inside the lingbot-va conda env
check_py_in_conda() {
    local desc="$1" env="$2" snippet="$3"
    local out
    if out="$(conda run -n "$env" --no-capture-output python -c "$snippet" 2>&1)"; then
        pass "$desc"
    else
        fail "$desc" "$(echo "$out" | tail -3)"
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
section "1. Files & paths"
check_exists "Isaac-GR00T source"             "$ISAAC_GROOT_SRC"
check_exists "Isaac-GR00T has gr00t/"         "$ISAAC_GROOT_SRC/gr00t"
check_exists ".isaac_groot symlink"           "$ISAAC_GROOT_LINK"
if [[ -L "$ISAAC_GROOT_LINK" ]]; then
    target="$(readlink -f "$ISAAC_GROOT_LINK")"
    if [[ "$target" == "$ISAAC_GROOT_SRC" ]]; then
        pass ".isaac_groot resolves to local Isaac-GR00T"
    else
        fail ".isaac_groot points elsewhere" "-> $target"
    fi
fi
check_exists ".venv_inference"                "$REPO_ROOT/.venv_inference/bin/python"
check_exists ".venv_data_collection"          "$REPO_ROOT/.venv_data_collection/bin/python"
check_exists "C++ build directory"            "$REPO_ROOT/gear_sonic_deploy/build"
if [[ -d "$REPO_ROOT/gear_sonic_deploy/build" ]]; then
    # Quick sanity: at least one ELF binary should be present.
    if find "$REPO_ROOT/gear_sonic_deploy/build" -maxdepth 3 -type f -executable 2>/dev/null | head -1 | grep -q .; then
        pass "C++ build has executables"
    else
        warn "C++ build dir exists but no executables found"
    fi
fi
check_exists "start_inference_va.sh"          "$REPO_ROOT/start_inference_va.sh"
check_exists "lingbot-va adapter"             "$LINGBOT_VA_REPO/wan_va/gr00t_compat_server.py"
check_exists "lingbot-va base model"          "$LINGBOT_VA_BASE"
check_exists "lingbot-va transformer ckpt"    "$LINGBOT_VA_TRANSFORMER"

# pyproject path sanity
PYPROJ="$REPO_ROOT/gear_sonic/pyproject.toml"
if grep -qF "file://${ISAAC_GROOT_LINK}" "$PYPROJ" 2>/dev/null; then
    pass "pyproject.toml gr00t @ file:// points at local .isaac_groot"
elif grep -qF 'file:///${PROJECT_ROOT}/.isaac_groot' "$PYPROJ" 2>/dev/null; then
    pass "pyproject.toml gr00t @ file:// uses the repo-relative \${PROJECT_ROOT}/.isaac_groot (uv)"
else
    warn "pyproject.toml gr00t path not recognized — manual check advised"
fi

# ──────────────────────────────────────────────────────────────────────────────
section "2. .venv_inference imports (run_vla_inference.py deps)"

VENV_INF_PY="$REPO_ROOT/.venv_inference/bin/python"
check_py_in_venv "import gr00t.policy.server_client.PolicyClient" "$VENV_INF_PY" \
    "from gr00t.policy.server_client import PolicyClient"
check_py_in_venv "import gear_sonic"                             "$VENV_INF_PY" \
    "import gear_sonic"
check_py_in_venv "import gear_sonic.scripts.run_vla_inference"   "$VENV_INF_PY" \
    "import importlib; importlib.import_module('gear_sonic.scripts.run_vla_inference') if False else None
import sys
sys.path.insert(0, '$REPO_ROOT')
import importlib.util
spec = importlib.util.spec_from_file_location('rva', '$REPO_ROOT/gear_sonic/scripts/run_vla_inference.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"
check_py_in_venv "import tyro, zmq, msgpack_numpy, cv2"          "$VENV_INF_PY" \
    "import tyro, zmq, msgpack_numpy, cv2"
check_py_in_venv "torch CUDA available"                          "$VENV_INF_PY" \
    "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"

# ──────────────────────────────────────────────────────────────────────────────
section "3. .venv_data_collection imports (camera-viewer deps)"

VENV_DC_PY="$REPO_ROOT/.venv_data_collection/bin/python"
check_py_in_venv "import gear_sonic"                             "$VENV_DC_PY" \
    "import gear_sonic"
check_py_in_venv "import tyro, zmq, cv2"                         "$VENV_DC_PY" \
    "import tyro, zmq, cv2"

# ──────────────────────────────────────────────────────────────────────────────
section "4. lingbot-va conda env"

CONDA_BASE="$(conda info --base 2>/dev/null)"
if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "$LINGBOT_VA_ENV"; then
        pass "conda env '$LINGBOT_VA_ENV' exists"
        check_py_in_conda "import msgpack_numpy, zmq, torch, diffusers"  "$LINGBOT_VA_ENV" \
            "import msgpack_numpy, zmq, torch, diffusers"
        check_py_in_conda "import wan_va.gr00t_compat_server.LingbotVAPolicy" "$LINGBOT_VA_ENV" \
            "import sys; sys.path.insert(0, '$LINGBOT_VA_REPO'); from wan_va.gr00t_compat_server import LingbotVAPolicy"
        check_py_in_conda "torch CUDA available"                         "$LINGBOT_VA_ENV" \
            "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
    else
        fail "conda env '$LINGBOT_VA_ENV' not found"
    fi
else
    fail "conda not detected — set CONDA_PROFILE or install miniconda"
fi

# ──────────────────────────────────────────────────────────────────────────────
section "5. Launcher --help round-trips"

# start_inference_va.sh --help should print and exit 0 without running anything.
if bash "$REPO_ROOT/start_inference_va.sh" --help &>/dev/null; then
    pass "start_inference_va.sh --help"
else
    fail "start_inference_va.sh --help failed"
fi

# setup_workspace_va.sh --help
if [[ -x "$REPO_ROOT/setup_workspace_va.sh" ]]; then
    if bash "$REPO_ROOT/setup_workspace_va.sh" --help &>/dev/null; then
        pass "setup_workspace_va.sh --help"
    else
        fail "setup_workspace_va.sh --help failed"
    fi
fi

# adapter --help (inside conda env)
if conda run -n "$LINGBOT_VA_ENV" --no-capture-output python -c "
import sys; sys.path.insert(0, '$LINGBOT_VA_REPO')
import argparse, wan_va.gr00t_compat_server as m
m._parse_args(['--help'])
" &>/dev/null; then
    pass "wan_va.gr00t_compat_server --help (CLI parser)"
else
    # argparse --help exits 0, but conda run may swallow it; try without --help
    if conda run -n "$LINGBOT_VA_ENV" --no-capture-output python -c "
import sys; sys.path.insert(0, '$LINGBOT_VA_REPO')
from wan_va.gr00t_compat_server import _parse_args
_parse_args(['--config-name','custom','--port','5550','--evict-t5'])
print('parser OK')" 2>&1 | grep -q "parser OK"; then
        pass "wan_va.gr00t_compat_server CLI parser"
    else
        fail "wan_va.gr00t_compat_server CLI parser failed"
    fi
fi

# ──────────────────────────────────────────────────────────────────────────────
if [[ $WITH_MODEL == 1 ]]; then
    section "6. Heavy: start adapter server + ZMQ ping"

    LOG="/tmp/check_workspace_va_lva_server.log"
    PIDFILE="/tmp/check_workspace_va_lva_server.pid"
    rm -f "$LOG" "$PIDFILE"

    if ! [[ -d "$LINGBOT_VA_BASE" && -d "$LINGBOT_VA_TRANSFORMER" ]]; then
        fail "model paths missing — skipping heavy test" \
             "set LINGBOT_VA_BASE and LINGBOT_VA_TRANSFORMER"
    else
        echo "  starting lingbot-va policy server in background (model load ~30s)..."
        (
            cd "$LINGBOT_VA_REPO"
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
            MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 \
            conda run -n "$LINGBOT_VA_ENV" --no-capture-output \
                python -u -m wan_va.gr00t_compat_server \
                    --config-name custom \
                    --base-model-path "$LINGBOT_VA_BASE" \
                    --transformer-path "$LINGBOT_VA_TRANSFORMER" \
                    --port 5550 --evict-t5 \
                    --initial-prompt "smoke test" > "$LOG" 2>&1 &
            echo $! > "$PIDFILE"
        )
        sleep 1
        SERVER_PID=$(cat "$PIDFILE" 2>/dev/null || echo "")
        if [[ -z "$SERVER_PID" ]]; then
            fail "server PID not captured" "see $LOG"
        else
            # Wait up to 120s for port 5550 to be bound
            BOUND=0
            for i in $(seq 1 120); do
                if ss -tln 2>/dev/null | grep -q ':5550 '; then
                    BOUND=1
                    pass "server bound to tcp://*:5550 after ${i}s"
                    break
                fi
                if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                    fail "server died during startup" "tail of $LOG: $(tail -5 "$LOG" | tr '\n' ' ')"
                    break
                fi
                sleep 1
            done
            if [[ $BOUND == 1 ]]; then
                # Send a ping
                PING_OUT="$(conda run -n "$LINGBOT_VA_ENV" --no-capture-output python -c "
import zmq, msgpack_numpy as mnp
ctx = zmq.Context.instance()
s = ctx.socket(zmq.REQ); s.setsockopt(zmq.RCVTIMEO, 5000); s.setsockopt(zmq.SNDTIMEO, 5000)
s.connect('tcp://127.0.0.1:5550')
s.send(mnp.packb({'endpoint':'ping'}))
print(mnp.unpackb(s.recv(), raw=False))
" 2>&1)"
                if echo "$PING_OUT" | grep -q "lingbot-va server is running"; then
                    pass "ZMQ ping round-trip"
                    # Report VRAM
                    VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
                    [[ -n "$VRAM" ]] && echo "         VRAM used: ${VRAM} MiB"
                else
                    fail "ping reply did not match" "got: $PING_OUT"
                fi
            fi
            # Cleanup — kill both the inner python and the `conda run` wrapper.
            pkill -TERM -f "wan_va.gr00t_compat_server" 2>/dev/null
            pkill -TERM -f "conda run -n ${LINGBOT_VA_ENV}" 2>/dev/null
            sleep 2
            pkill -KILL -f "wan_va.gr00t_compat_server" 2>/dev/null
            pkill -KILL -f "conda run -n ${LINGBOT_VA_ENV}" 2>/dev/null
            kill -9 "$SERVER_PID" 2>/dev/null
        fi
        rm -f "$PIDFILE"
    fi
fi

# ──────────────────────────────────────────────────────────────────────────────
echo ""
printf "${B}══════════════════════════════════════════════════════════════════${D}\n"
printf "  Summary: ${G}%d pass${D}, ${R}%d fail${D}, ${Y}%d warn${D}\n" "$PASS" "$FAIL" "$WARN"
if [[ $FAIL -gt 0 ]]; then
    printf "  Failed items:\n"
    for it in "${FAILED_ITEMS[@]}"; do
        printf "    - %s\n" "$it"
    done
    printf "${R}══════════════════════════════════════════════════════════════════${D}\n"
    exit 1
else
    printf "${G}══════════════════════════════════════════════════════════════════${D}\n"
    if [[ $WITH_MODEL == 0 ]]; then
        printf "  Tip: run with ${C}--with-model${D} to also smoke-test the policy server.\n"
    fi
    exit 0
fi
