#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 在这里填写要合并的所有数据集路径(每行一个,可任意增减)
# Fill in all dataset paths to merge below (one per line).
# ============================================================
DATASET_PATHS=(
    "outputs/2026-06-15-16-13-52_convey_belt_lxz1"
    "outputs/2026-06-15-16-46-32_convey_belt_zcy1"
    "outputs/2026-06-15-20-28-43_convey_belt_lxz2"
    "outputs/2026-06-16-15-20-38_convey_belt_zcy"
    "outputs/2026-06-17-11-02-26_convey_belt_zcy1"
    "outputs/2026-06-17-16-49-24_convey_belt_zcy2"
    "outputs/2026-06-17-18-13-07_convey_belt_lxz1"
    "outputs/2026-06-26-17-20-20_convey_belt_zcy"
)

# 合并后输出路径 / Output path for the merged dataset
OUTPUT_PATH="outputs_pre/convey_belt_merged_0626"

# ============================================================
# 以下内容一般不用改动 / Usually no need to edit below
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活虚拟环境 / Activate venv
source .venv_data_collection/bin/activate

python gear_sonic/scripts/prepare_dataset.py \
    --dataset-paths "${DATASET_PATHS[@]}" \
    --output-path "$OUTPUT_PATH"
