#!/usr/bin/env bash
# setup_workspace_va.sh — one-click env setup for the lingbot-va inference stack.
#
# What it does (in order):
#   0. preflight — verify required tools + repos
#   1. path anchor — rewrite the repo-relative ${PROJECT_ROOT} path in
#      gear_sonic/pyproject.toml to the absolute local path, and create the
#      .isaac_groot symlink to ../Isaac-GR00T (no sudo).
#   2. install .venv_inference        (gear_sonic[inference] client)
#   3. install .venv_data_collection  (camera-viewer + data exporter)
#   4. build gear_sonic_deploy/build/ (C++ controller; `just build` only,
#      no deploy.sh — that requires a robot NIC)
#   5. sanity-check the lingbot-va conda env + adapter import
#
# What it skips (not needed for inference-only setup on a dev box):
#   - install_camera_server.sh, install_pico.sh, install_mujoco_sim.sh,
#     install_leap_sdk.sh, install_ros.sh
#   - Isaac-GR00T/.venv (uv sync) — server runs from lingbot-va env now
#
# Usage:
#   bash setup_workspace_va.sh                       # default: interactive
#   bash setup_workspace_va.sh --yes                 # no prompt
#   bash setup_workspace_va.sh --skip-build          # skip C++ build
#   bash setup_workspace_va.sh --help
set -euo pipefail

# ── Paths & defaults ──────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
ISAAC_GROOT_SRC="$WORKSPACE_ROOT/Isaac-GR00T"
ISAAC_GROOT_LINK="$REPO_ROOT/.isaac_groot"
PYPROJECT="$REPO_ROOT/gear_sonic/pyproject.toml"

LINGBOT_VA_REPO="${LINGBOT_VA_REPO:-$HOME/lingbot-va}"
LINGBOT_VA_ENV="${LINGBOT_VA_ENV:-lingbot-va}"

OLD_FILE_URI='file:///${PROJECT_ROOT}/.isaac_groot'
NEW_FILE_URI="file://${ISAAC_GROOT_LINK}"

SKIP_INFERENCE=0
SKIP_DATA_COLLECTION=0
SKIP_BUILD=0
SKIP_LINGBOT_CHECK=0
ASSUME_YES=0

# ── Colors (tty-only) ─────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    R='\033[31m'; G='\033[32m'; Y='\033[33m'; B='\033[34m'; C='\033[36m'; D='\033[0m'
else
    R=''; G=''; Y=''; B=''; C=''; D=''
fi
say()  { printf "${B}==>${D} %s\n" "$*"; }
ok()   { printf "${G}[OK]${D}  %s\n" "$*"; }
warn() { printf "${Y}[WARN]${D} %s\n" "$*"; }
die()  { printf "${R}[FAIL]${D} %s\n" "$*" >&2; exit 1; }

# ── CLI parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-inference)        SKIP_INFERENCE=1;        shift ;;
        --skip-data-collection)  SKIP_DATA_COLLECTION=1;  shift ;;
        --skip-build)            SKIP_BUILD=1;            shift ;;
        --skip-lingbot-check)    SKIP_LINGBOT_CHECK=1;    shift ;;
        --yes|-y)                ASSUME_YES=1;            shift ;;
        --help|-h)
            sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) die "Unknown option: $1  (use --help)" ;;
    esac
done

# ── Step 0: preflight ─────────────────────────────────────────────────────────
say "Step 0/5 — preflight"

require_tool() {
    command -v "$1" &>/dev/null || die "missing required tool: $1"
}
for t in cmake clang git just conda uv tmux make; do require_tool "$t"; done
ok "system tools present"

[[ -d "$ISAAC_GROOT_SRC" ]] || die "Isaac-GR00T not found at $ISAAC_GROOT_SRC"
# Verify the submodule is materialised (not just a bare .git pointer file)
if [[ ! -d "$ISAAC_GROOT_SRC/gr00t" ]]; then
    die "Isaac-GR00T appears empty (no gr00t/). Run 'git submodule update --init --force' first."
fi
ok "Isaac-GR00T present at $ISAAC_GROOT_SRC"

[[ -f "$PYPROJECT" ]] || die "pyproject.toml not found: $PYPROJECT"

# conda env check
CONDA_BASE="$(conda info --base 2>/dev/null)" || die "cannot resolve conda base"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda env list | awk '{print $1}' | grep -qx "$LINGBOT_VA_ENV" \
    || die "conda env '$LINGBOT_VA_ENV' not found"
ok "conda env '$LINGBOT_VA_ENV' exists"

[[ -f "$LINGBOT_VA_REPO/wan_va/gr00t_compat_server.py" ]] \
    || die "adapter not found at $LINGBOT_VA_REPO/wan_va/gr00t_compat_server.py"
ok "lingbot-va repo + adapter present"

# ── Plan summary ──────────────────────────────────────────────────────────────
echo ""
printf "${C}Planned actions:${D}\n"
printf "  1. Rewrite pyproject path → %s\n" "$NEW_FILE_URI"
printf "     and create symlink %s -> ../Isaac-GR00T\n" "$(basename "$ISAAC_GROOT_LINK")"
[[ $SKIP_INFERENCE       == 0 ]] && printf "  2. bash install_scripts/install_inference.sh        (.venv_inference)\n"        || printf "  2. ${Y}skipped${D}: install_inference.sh\n"
[[ $SKIP_DATA_COLLECTION == 0 ]] && printf "  3. bash install_scripts/install_data_collection.sh  (.venv_data_collection)\n" || printf "  3. ${Y}skipped${D}: install_data_collection.sh\n"
[[ $SKIP_BUILD           == 0 ]] && printf "  4. just build  (in gear_sonic_deploy/)\n"                                       || printf "  4. ${Y}skipped${D}: C++ build\n"
[[ $SKIP_LINGBOT_CHECK   == 0 ]] && printf "  5. lingbot-va env sanity + adapter import smoke test\n"                          || printf "  5. ${Y}skipped${D}: lingbot-va sanity\n"
echo ""

if [[ $ASSUME_YES == 0 ]]; then
    read -r -p "Proceed? [y/N] " ans
    [[ "${ans:-}" == "y" || "${ans:-}" == "Y" ]] || { echo "aborted."; exit 0; }
fi

# ── Step 1: path anchor ───────────────────────────────────────────────────────
say "Step 1/5 — path anchor"

# 1a. rewrite pyproject.toml only if the old path is still present
if grep -qF "$OLD_FILE_URI" "$PYPROJECT"; then
    sed -i.bak "s#${OLD_FILE_URI//\//\\/}#${NEW_FILE_URI//\//\\/}#g" "$PYPROJECT"
    ok "rewrote $PYPROJECT (backup: ${PYPROJECT}.bak)"
elif grep -qF "$NEW_FILE_URI" "$PYPROJECT"; then
    ok "$PYPROJECT already points at local Isaac-GR00T"
else
    warn "neither old nor new file:// path found in pyproject.toml — leaving it as-is"
fi

# 1b. .isaac_groot symlink
ln -sfn "$ISAAC_GROOT_SRC" "$ISAAC_GROOT_LINK"
ok "symlink $ISAAC_GROOT_LINK -> $ISAAC_GROOT_SRC"

# 1c. ask git to ignore working-tree changes to pyproject.toml
if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree &>/dev/null; then
    if git -C "$REPO_ROOT" ls-files --error-unmatch gear_sonic/pyproject.toml &>/dev/null; then
        git -C "$REPO_ROOT" update-index --skip-worktree gear_sonic/pyproject.toml \
            && ok "git update-index --skip-worktree gear_sonic/pyproject.toml" \
            || warn "git update-index failed (continuing)"
    else
        warn "gear_sonic/pyproject.toml is not tracked here — skipping update-index"
    fi
else
    warn "not inside a git repo — skipping update-index"
fi

# ── Step 2: .venv_inference ───────────────────────────────────────────────────
if [[ $SKIP_INFERENCE == 0 ]]; then
    say "Step 2/5 — install .venv_inference"
    (cd "$REPO_ROOT" && bash install_scripts/install_inference.sh)
    ok ".venv_inference ready"
else
    warn "step 2 skipped (--skip-inference)"
fi

# ── Step 3: .venv_data_collection ─────────────────────────────────────────────
if [[ $SKIP_DATA_COLLECTION == 0 ]]; then
    say "Step 3/5 — install .venv_data_collection"
    (cd "$REPO_ROOT" && bash install_scripts/install_data_collection.sh)
    ok ".venv_data_collection ready"
else
    warn "step 3 skipped (--skip-data-collection)"
fi

# ── Step 4: C++ build ─────────────────────────────────────────────────────────
if [[ $SKIP_BUILD == 0 ]]; then
    say "Step 4/5 — just build (gear_sonic_deploy)"
    (
        cd "$REPO_ROOT/gear_sonic_deploy"
        # setup_env.sh is harmless on this machine; tolerate failures (jetson, etc.)
        set +e
        # shellcheck disable=SC1091
        source scripts/setup_env.sh 2>/dev/null
        set -e
        just build
    )
    ok "C++ binaries built into $REPO_ROOT/gear_sonic_deploy/build"
else
    warn "step 4 skipped (--skip-build)"
fi

# ── Step 5: lingbot-va sanity ─────────────────────────────────────────────────
if [[ $SKIP_LINGBOT_CHECK == 0 ]]; then
    say "Step 5/5 — lingbot-va env sanity"
    conda activate "$LINGBOT_VA_ENV"
    python -m pip install --quiet msgpack-numpy pyzmq
    PYTHONPATH="$LINGBOT_VA_REPO:${PYTHONPATH:-}" python - <<'PY'
import sys
from wan_va.gr00t_compat_server import LingbotVAPolicy  # noqa: F401
print("[OK] adapter import succeeded")
PY
    conda deactivate
    ok "lingbot-va env ready"
else
    warn "step 5 skipped (--skip-lingbot-check)"
fi

# ── Final banner ──────────────────────────────────────────────────────────────
echo ""
printf "${G}══════════════════════════════════════════════════════════════════${D}\n"
printf "${G}  Setup complete!${D}\n\n"
printf "  ✓ .venv_inference         @ %s/.venv_inference\n"          "$REPO_ROOT"
printf "  ✓ .venv_data_collection   @ %s/.venv_data_collection\n"    "$REPO_ROOT"
printf "  ✓ C++ binaries            @ %s/gear_sonic_deploy/build\n"  "$REPO_ROOT"
printf "  ✓ .isaac_groot            @ %s -> ../Isaac-GR00T\n"        "$ISAAC_GROOT_LINK"
printf "  ✓ lingbot-va env          ready (%s)\n"                    "$LINGBOT_VA_ENV"
echo ""
printf "  Next, launch the VLA stack (needs camera + robot online):\n\n"
printf "    bash %s/start_inference_va.sh \\\\\n" "$REPO_ROOT"
printf "        -p \"your prompt here\" \\\\\n"
printf "        -b /path/to/lingbot-va-base \\\\\n"
printf "        -t /path/to/lingbot-va/checkpoint/transformer \\\\\n"
printf "        -r %s\n" "$LINGBOT_VA_REPO"
printf "${G}══════════════════════════════════════════════════════════════════${D}\n"
