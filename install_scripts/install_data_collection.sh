#!/usr/bin/env bash
# Modified by LocoManiLab (Gamma Robotics), 2026
# install_data_collection.sh
# Sets up the .venv_data_collection venv for recording teleop demonstrations
# in LeRobot dataset format (for post-training with Isaac-GR00T).
#
# Installs gear_sonic[data_collection] which pulls in LeRobot, PyAV, OpenCV,
# and the other dependencies needed by run_data_exporter.py.
#
# Usage:  bash install_scripts/install_data_collection.sh   (run from repo root)
#
# 国内环境默认走阿里云镜像（关闭：DATA_COLLECTION_NO_ALIYUN=1）：
#   - PyPI：UV_DEFAULT_INDEX / PIP_INDEX_URL（可用 DATA_COLLECTION_PYPI_INDEX 覆盖）
#   - apt：切换 Ubuntu 源到 mirrors.aliyun.com（跳过：DATA_COLLECTION_APT_ALIYUN=0）
#   - uv 托管 Python：GitHub 下载慢时可设 DATA_COLLECTION_PYTHON_INSTALL_MIRROR（见脚本内说明）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Aliyun mirrors (PyPI + optional apt) ─────────────────────────────────────
if [ "${DATA_COLLECTION_NO_ALIYUN:-0}" != "1" ]; then
    _PYPI_INDEX="${DATA_COLLECTION_PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
    export UV_DEFAULT_INDEX="$_PYPI_INDEX"
    export UV_INDEX_URL="$_PYPI_INDEX"
    export PIP_INDEX_URL="$_PYPI_INDEX"
    echo "[INFO] Using PyPI index: $_PYPI_INDEX"
    if [ -n "${DATA_COLLECTION_PYTHON_INSTALL_MIRROR:-}" ]; then
        export UV_PYTHON_INSTALL_MIRROR="$DATA_COLLECTION_PYTHON_INSTALL_MIRROR"
        echo "[INFO] UV_PYTHON_INSTALL_MIRROR=$UV_PYTHON_INSTALL_MIRROR"
    fi
fi

configure_apt_aliyun() {
    if [ "${DATA_COLLECTION_NO_ALIYUN:-0}" = "1" ] || [ "${DATA_COLLECTION_APT_ALIYUN:-1}" = "0" ]; then
        return 0
    fi
    command -v apt-get &>/dev/null || return 0
    if [ ! -f /etc/os-release ]; then
        return 0
    fi
    # shellcheck disable=SC1091
    . /etc/os-release
    if [ "${ID:-}" != "ubuntu" ]; then
        return 0
    fi

    echo "[INFO] 将 apt 源替换为阿里云镜像（跳过：DATA_COLLECTION_APT_ALIYUN=0）…"
    _ts="$(date +%Y%m%d_%H%M%S)"
    if [ -f /etc/apt/sources.list ]; then
        sudo cp -a /etc/apt/sources.list "/etc/apt/sources.list.bak.gr00t_data_collection.${_ts}" 2>/dev/null || true
        sudo sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
            -e 's|https://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
            -e 's|http://cn.archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
            -e 's|https://cn.archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
            -e 's|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
            -e 's|https://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
            -e 's|http://ports.ubuntu.com/ubuntu-ports|http://mirrors.aliyun.com/ubuntu-ports|g' \
            -e 's|https://ports.ubuntu.com/ubuntu-ports|https://mirrors.aliyun.com/ubuntu-ports|g' \
            /etc/apt/sources.list 2>/dev/null || true
    fi
    shopt -s nullglob
    for _f in /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list; do
        sudo cp -a "$_f" "${_f}.bak.gr00t_data_collection.${_ts}" 2>/dev/null || true
        sudo sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
            -e 's|https://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
            -e 's|http://cn.archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
            -e 's|https://cn.archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
            -e 's|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
            -e 's|https://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
            -e 's|http://ports.ubuntu.com/ubuntu-ports|http://mirrors.aliyun.com/ubuntu-ports|g' \
            -e 's|https://ports.ubuntu.com/ubuntu-ports|https://mirrors.aliyun.com/ubuntu-ports|g' \
            "$_f" 2>/dev/null || true
    done
    shopt -u nullglob
    sudo apt-get update -qq || echo "[WARN] apt-get update failed — continuing"
}

# ── 0. System dependencies ────────────────────────────────────────────────────
ARCH="$(uname -m)"
echo "[OK] Architecture: $ARCH"

echo "[INFO] Installing system dependencies (espeak for voice feedback) …"
if command -v apt-get &>/dev/null; then
    configure_apt_aliyun
    sudo apt-get install -y espeak >/dev/null 2>&1 || echo "[WARN] Could not install espeak — voice feedback will be disabled"
fi

# ── 1. Ensure uv is installed and available ──────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[INFO] uv not found – installing via official installer …"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source the uv env so it's available in this session
    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    else
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if ! command -v uv &>/dev/null; then
        echo "[ERROR] uv installation succeeded but binary not found on PATH."
        echo "        Please add ~/.local/bin (or ~/.cargo/bin) to your PATH and re-run."
        exit 1
    fi
fi
echo "[OK] uv $(uv --version)"

# ── 2. Install a uv-managed Python 3.10 (includes dev headers / Python.h) ────
echo "[INFO] Installing uv-managed Python 3.10 (includes development headers) …"
uv python install 3.10
MANAGED_PY="$(uv python find --no-project 3.10)"
echo "[OK] Using Python: $MANAGED_PY"

# ── 3. Clean previous venv (if any) ──────────────────────────────────────────
cd "$REPO_ROOT"
echo "[INFO] Removing old .venv_data_collection (if present) …"
rm -rf .venv_data_collection

# ── 4. Create venv & install data_collection extra ───────────────────────────
echo "[INFO] Creating .venv_data_collection with uv-managed Python 3.10 …"
uv venv .venv_data_collection --python "$MANAGED_PY" --prompt gear_sonic_data_collection
# shellcheck disable=SC1091
source .venv_data_collection/bin/activate
echo "[INFO] Installing gear_sonic[data_collection] (this may take a few minutes) …"
# LeRobot's git repo contains LFS test artifacts that aren't needed at runtime.
# Skip them to avoid download failures and save bandwidth.
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e "gear_sonic[data_collection]"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Setup complete!  Activate the venv with:"
echo ""
echo "    source .venv_data_collection/bin/activate"
echo ""
echo "  You should see (gear_sonic_data_collection) in your prompt."
echo ""
echo "  Then run the data exporter with:"
echo "    python gear_sonic/scripts/run_data_exporter.py"
echo "══════════════════════════════════════════════════════════════"
