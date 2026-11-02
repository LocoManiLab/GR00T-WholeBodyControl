"""Prepare G1 LeRobot hand columns: derive open/close labels, smooth, and prune.

Reads datasets from --input-root (default: outputs/) and writes results to
--output-root (default: outputs_pre/).  The original data is never modified:
each dataset is copied to the output root before any processing begins.

For each parquet episode we (optionally) add:

    teleop.left_hand_open            (1,)   0.0 = open, 1.0 = closed
    teleop.right_hand_open           (1,)
    teleop.left_hand_joints_smooth   (7,)   centered moving-average
    teleop.right_hand_joints_smooth  (7,)
    teleop.left_hand_open_smooth     (1,)   continuous in [0, 1]
    teleop.right_hand_open_smooth    (1,)

Open/close derivation: Dex3 has seven joints per side; index_0/index_1/
middle_0/middle_1 are the four dominant flexion joints, and their absolute
angle scales linearly with finger curl. We average the per-joint normalized
closure to get a continuous closure value in [0, 1] and threshold it
(default 0.2). The recorded teleop.left_hand_joints field is all zeros
(only the right hand was teleoperated), so we derive both labels from joint
feedback (observation.state).

Smoothing: centered moving average with shrinking-edge handling
(min_periods=1) — output length matches input length, edges average only
in-bounds frames.

Discard: read meta/info.json["discarded_episode_indices"] and remove each
listed episode (parquet + videos + episodes.jsonl line); decrement totals
and clear the list.

Each meta/info.json, meta/modality.json, and meta/stats.json is updated
in the output copy.

Usage:
    # process all datasets found in outputs/ → outputs_pre/
    python gear_sonic/scripts/prepare_hand.py

    # process specific datasets
    python gear_sonic/scripts/prepare_hand.py \\
        --dataset-paths outputs/2026-04-28-take-walk-and-put-hand \\
        --window 11

    # custom roots
    python gear_sonic/scripts/prepare_hand.py \\
        --input-root /data/raw --output-root /data/processed
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Open/close labeling
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_ROOT = _REPO_ROOT / "outputs"
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "outputs_pre"

LEFT_HAND_SLICE = slice(22, 29)
RIGHT_HAND_SLICE = slice(36, 43)
FLEXION_JOINTS = [0, 1, 2, 3]
JOINT_MAX_ABS = np.array([1.6, 1.74, 1.6, 1.74], dtype=np.float32)
DEFAULT_CLOSURE_THRESHOLD = 0.2

LEFT_OPEN_KEY = "teleop.left_hand_open"
RIGHT_OPEN_KEY = "teleop.right_hand_open"

# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

LEFT_JOINTS_KEY = "teleop.left_hand_joints"
RIGHT_JOINTS_KEY = "teleop.right_hand_joints"

LEFT_JOINTS_SMOOTH_KEY = "teleop.left_hand_joints_smooth"
RIGHT_JOINTS_SMOOTH_KEY = "teleop.right_hand_joints_smooth"
LEFT_OPEN_SMOOTH_KEY = "teleop.left_hand_open_smooth"
RIGHT_OPEN_SMOOTH_KEY = "teleop.right_hand_open_smooth"

# (src_key, smooth_key, dim)
SMOOTH_SPECS = [
    (LEFT_JOINTS_KEY, LEFT_JOINTS_SMOOTH_KEY, 7),
    (RIGHT_JOINTS_KEY, RIGHT_JOINTS_SMOOTH_KEY, 7),
    (LEFT_OPEN_KEY, LEFT_OPEN_SMOOTH_KEY, 1),
    (RIGHT_OPEN_KEY, RIGHT_OPEN_SMOOTH_KEY, 1),
]

OPEN_SPECS = [(LEFT_OPEN_KEY, 1), (RIGHT_OPEN_KEY, 1)]

DEFAULT_WINDOW = 31


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hand_closed_label(hand_joints: np.ndarray, threshold: float) -> np.ndarray:
    flex = np.abs(hand_joints[:, FLEXION_JOINTS]).astype(np.float32)
    closure = np.clip(flex / JOINT_MAX_ABS, 0.0, 1.0).mean(axis=1)
    return (closure > threshold).astype(np.float32)


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.astype(np.float32)
    smoothed = (
        pd.DataFrame(arr)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    return smoothed.astype(np.float32)


def _stats(arr: np.ndarray) -> dict:
    arr = arr.astype(np.float32)
    return {
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Per-parquet processing (one read, one write)
# ---------------------------------------------------------------------------


def process_parquet(
    path: Path,
    threshold: float,
    window: int,
    do_open: bool,
    do_smooth: bool,
    force: bool,
) -> dict[str, np.ndarray]:
    df = pd.read_parquet(path)
    dirty = False
    out: dict[str, np.ndarray] = {}

    if do_open:
        need = force or LEFT_OPEN_KEY not in df.columns or RIGHT_OPEN_KEY not in df.columns
        if need:
            state = np.stack(df["observation.state"].values).astype(np.float32)
            left = hand_closed_label(state[:, LEFT_HAND_SLICE], threshold)
            right = hand_closed_label(state[:, RIGHT_HAND_SLICE], threshold)
            df[LEFT_OPEN_KEY] = list(left[:, None])
            df[RIGHT_OPEN_KEY] = list(right[:, None])
            dirty = True
        out[LEFT_OPEN_KEY] = np.stack(df[LEFT_OPEN_KEY].values).astype(np.float32)
        out[RIGHT_OPEN_KEY] = np.stack(df[RIGHT_OPEN_KEY].values).astype(np.float32)

    if do_smooth:
        for src, dst, _ in SMOOTH_SPECS:
            if src not in df.columns:
                raise KeyError(
                    f"{path}: missing source column '{src}' "
                    f"(run with open-labeling enabled, or check dataset)"
                )
            need = force or dst not in df.columns
            if need:
                raw = np.stack(df[src].values).astype(np.float32)
                smoothed = moving_average(raw, window)
                df[dst] = list(smoothed)
                dirty = True
            out[dst] = np.stack(df[dst].values).astype(np.float32)

    if dirty:
        df.to_parquet(path, index=False)
    return out


# ---------------------------------------------------------------------------
# Meta updates
# ---------------------------------------------------------------------------


def update_info(info_path: Path, do_open: bool, do_smooth: bool) -> None:
    info = json.loads(info_path.read_text())
    features = info["features"]
    changed = False
    specs: list[tuple[str, int]] = []
    if do_open:
        specs.extend(OPEN_SPECS)
    if do_smooth:
        specs.extend((dst, dim) for _, dst, dim in SMOOTH_SPECS)
    for key, dim in specs:
        if key not in features:
            features[key] = {"dtype": "float32", "shape": [dim], "names": None}
            changed = True
    if changed:
        info_path.write_text(json.dumps(info, indent=4))


def update_modality(modality_path: Path, do_open: bool, do_smooth: bool) -> None:
    modality = json.loads(modality_path.read_text())
    action = modality.setdefault("action", {})
    changed = False
    entries: list[tuple[str, str, int]] = []
    if do_open:
        entries.append(("left_hand_open", LEFT_OPEN_KEY, 1))
        entries.append(("right_hand_open", RIGHT_OPEN_KEY, 1))
    if do_smooth:
        for _, dst, dim in SMOOTH_SPECS:
            entries.append((dst.split(".", 1)[1], dst, dim))
    for short, original, dim in entries:
        if short not in action:
            action[short] = {"start": 0, "end": dim, "original_key": original}
            changed = True
    if changed:
        modality_path.write_text(json.dumps(modality, indent=4))


def update_stats(stats_path: Path, accum: dict[str, list[np.ndarray]]) -> None:
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for key, chunks in accum.items():
        if not chunks:
            continue
        arr = np.concatenate(chunks, axis=0)
        stats[key] = _stats(arr)
    stats_path.write_text(json.dumps(stats, indent=4))


# ---------------------------------------------------------------------------
# Discard
# ---------------------------------------------------------------------------


def _episode_parquet(dataset_path: Path, idx: int) -> Path | None:
    matches = list(dataset_path.glob(f"data/chunk-*/episode_{idx:06d}.parquet"))
    return matches[0] if matches else None


def _episode_videos(dataset_path: Path, idx: int) -> list[Path]:
    return list(dataset_path.glob(f"videos/chunk-*/*/episode_{idx:06d}.mp4"))


def discard_dataset(dataset_path: Path) -> None:
    info_path = dataset_path / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    discarded = sorted(set(info.get("discarded_episode_indices", []) or []))
    if not discarded:
        print(f"  {dataset_path.name}: no discarded_episode_indices, skipping")
        return

    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episode_lines = episodes_path.read_text().splitlines()
    by_idx: dict[int, dict] = {}
    for line in episode_lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        by_idx[rec["episode_index"]] = rec

    removed_frames = 0
    removed_videos = 0
    removed_parquets = 0
    for idx in discarded:
        rec = by_idx.pop(idx, None)
        if rec is not None:
            removed_frames += int(rec.get("length", 0))

        pq = _episode_parquet(dataset_path, idx)
        if pq is not None and pq.exists():
            pq.unlink()
            removed_parquets += 1

        for vp in _episode_videos(dataset_path, idx):
            if vp.exists():
                vp.unlink()
                removed_videos += 1

    kept = [json.loads(line) for line in episode_lines if line.strip()]
    kept = [r for r in kept if r["episode_index"] not in set(discarded)]
    episodes_path.write_text("".join(json.dumps(r) + "\n" for r in kept))

    info["total_episodes"] = max(0, int(info.get("total_episodes", 0)) - len(discarded))
    info["total_frames"] = max(0, int(info.get("total_frames", 0)) - removed_frames)
    info["total_videos"] = max(0, int(info.get("total_videos", 0)) - removed_videos)
    info["discarded_episode_indices"] = []
    info_path.write_text(json.dumps(info, indent=4))

    print(
        f"  {dataset_path.name}: discarded {len(discarded)} episodes "
        f"(parquets={removed_parquets}, videos={removed_videos}, "
        f"frames={removed_frames}); indices={discarded}"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def process_dataset(
    dataset_path: Path,
    threshold: float,
    window: int,
    do_open: bool,
    do_smooth: bool,
    do_discard: bool,
    force: bool,
) -> None:
    if do_open or do_smooth:
        parquets = sorted(dataset_path.glob("data/chunk-*/*.parquet"))
        if not parquets:
            raise FileNotFoundError(f"No parquet episodes under {dataset_path}/data")

        accum_keys: list[str] = []
        if do_open:
            accum_keys.extend(k for k, _ in OPEN_SPECS)
        if do_smooth:
            accum_keys.extend(dst for _, dst, _ in SMOOTH_SPECS)
        accum: dict[str, list[np.ndarray]] = {k: [] for k in accum_keys}

        desc_parts = []
        if do_open:
            desc_parts.append("open")
        if do_smooth:
            desc_parts.append(f"smooth(w={window})")
        desc = "+".join(desc_parts)

        for p in tqdm(parquets, desc=f"{desc} {dataset_path.name}"):
            out = process_parquet(
                p,
                threshold=threshold,
                window=window,
                do_open=do_open,
                do_smooth=do_smooth,
                force=force,
            )
            for k, v in out.items():
                accum[k].append(v)

        update_info(dataset_path / "meta" / "info.json", do_open, do_smooth)
        update_modality(dataset_path / "meta" / "modality.json", do_open, do_smooth)
        update_stats(dataset_path / "meta" / "stats.json", accum)

        msg = f"  {dataset_path.name}: processed {len(parquets)} episodes"
        if do_open:
            left_mean = np.concatenate(accum[LEFT_OPEN_KEY]).mean()
            right_mean = np.concatenate(accum[RIGHT_OPEN_KEY]).mean()
            msg += f" (left closed={left_mean:.3f}, right closed={right_mean:.3f})"
        print(msg)

    if do_discard:
        discard_dataset(dataset_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare hand columns; reads from --input-root, writes to --output-root."
    )
    parser.add_argument(
        "--dataset-paths",
        nargs="*",
        type=Path,
        default=None,
        help="explicit dataset dirs to process (default: all dirs under --input-root)",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"root that contains raw dataset dirs (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"root where processed copies are written (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--closure-threshold",
        type=float,
        default=DEFAULT_CLOSURE_THRESHOLD,
        help="mean fingertip closure above this is labeled closed (1.0)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="centered moving-average window in frames (odd)",
    )
    parser.add_argument("--no-open", action="store_true", help="skip open/close labeling")
    parser.add_argument("--no-smooth", action="store_true", help="skip smoothing step")
    parser.add_argument("--no-discard", action="store_true", help="skip discard step")
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute columns even if they already exist",
    )
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    # Discover source datasets
    if args.dataset_paths:
        src_datasets = [p.resolve() for p in args.dataset_paths]
    else:
        if not input_root.is_dir():
            parser.error(f"--input-root does not exist: {input_root}")
        src_datasets = sorted(p for p in input_root.iterdir() if p.is_dir())
        if not src_datasets:
            parser.error(f"No dataset directories found under {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    for src in src_datasets:
        # Destination mirrors the relative path inside input_root when possible
        try:
            rel = src.relative_to(input_root)
            dst = output_root / rel
        except ValueError:
            dst = output_root / src.name

        # Copy source → destination (skip if already exists, unless --force)
        if dst.exists() and args.force:
            shutil.rmtree(dst)
        if not dst.exists():
            print(f"Copying {src.name} → {dst} ...")
            shutil.copytree(src, dst)
        else:
            print(f"Output already exists, reusing: {dst}")

        process_dataset(
            dst,
            threshold=args.closure_threshold,
            window=args.window,
            do_open=not args.no_open,
            do_smooth=not args.no_smooth,
            do_discard=not args.no_discard,
            force=args.force,
        )


if __name__ == "__main__":
    main()
