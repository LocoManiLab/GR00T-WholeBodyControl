"""Unified LeRobot dataset preparation pipeline.

Steps (all enabled by default, each optional via --no-* flags):
  1. Discard   — remove episodes listed in discarded_episode_indices
  2. SMPL      — remove stale SMPL frames (all-zero teleop.smpl_pose and
                 frozen lead-in frames that precede them)
  3. Merge     — combine multiple input datasets into one, renumbering episodes
  4. Hand      — compute open/close labels from observation.state joints
  5. Smooth    — centered moving-average over hand joints and open/close signals
  6. Meta      — write info.json, modality.json, stats.json for the output

Output is always written to --output-path (never in-place).
Merged meta considers all input datasets (episode counts, features, tasks).
Script configs must match across all input datasets.

Usage:
    # Single dataset: full pipeline
    python gear_sonic/scripts/prepare_dataset.py \\
        --dataset-paths outputs/session1 \\
        --output-path outputs/session1_ready

    # Merge two sessions + full pipeline
    python gear_sonic/scripts/prepare_dataset.py \\
        --dataset-paths outputs/session1 outputs/session2 \\
        --output-path outputs/merged_ready

    # Read dataset paths from a text file (one per line, # = comment)
    python gear_sonic/scripts/prepare_dataset.py \\
        --dataset-list datasets.txt \\
        --output-path outputs/merged_ready

    # Skip discard and smoothing
    python gear_sonic/scripts/prepare_dataset.py \\
        --dataset-paths outputs/session1 \\
        --output-path outputs/out \\
        --no-discard --no-smooth
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SMPL_POSE_COLUMN = "teleop.smpl_pose"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "outputs_pre"

LEFT_HAND_SLICE = slice(22, 29)
RIGHT_HAND_SLICE = slice(36, 43)
FLEXION_JOINTS = [0, 1, 2, 3]
JOINT_MAX_ABS = np.array([1.6, 1.74, 1.6, 1.74], dtype=np.float32)
DEFAULT_CLOSURE_THRESHOLD = 0.2
DEFAULT_WINDOW = 31

LEFT_OPEN_KEY = "teleop.left_hand_open"
RIGHT_OPEN_KEY = "teleop.right_hand_open"
LEFT_JOINTS_KEY = "teleop.left_hand_joints"
RIGHT_JOINTS_KEY = "teleop.right_hand_joints"
LEFT_JOINTS_SMOOTH_KEY = "teleop.left_hand_joints_smooth"
RIGHT_JOINTS_SMOOTH_KEY = "teleop.right_hand_joints_smooth"
LEFT_OPEN_SMOOTH_KEY = "teleop.left_hand_open_smooth"
RIGHT_OPEN_SMOOTH_KEY = "teleop.right_hand_open_smooth"

# (source_col, output_col, dim)
SMOOTH_SPECS: list[tuple[str, str, int]] = [
    (LEFT_JOINTS_KEY, LEFT_JOINTS_SMOOTH_KEY, 7),
    (RIGHT_JOINTS_KEY, RIGHT_JOINTS_SMOOTH_KEY, 7),
    (LEFT_OPEN_KEY, LEFT_OPEN_SMOOTH_KEY, 1),
    (RIGHT_OPEN_KEY, RIGHT_OPEN_SMOOTH_KEY, 1),
]

OPEN_SPECS: list[tuple[str, int]] = [
    (LEFT_OPEN_KEY, 1),
    (RIGHT_OPEN_KEY, 1),
]


# ---------------------------------------------------------------------------
# Hand-labeling helpers
# ---------------------------------------------------------------------------

def hand_closed_label(hand_joints: np.ndarray, threshold: float) -> np.ndarray:
    """Binary (0.0 open / 1.0 closed) per frame, shape (N,)."""
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


def add_hand_columns(
    df: pd.DataFrame,
    threshold: float,
    window: int,
    do_open: bool,
    do_smooth: bool,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Add hand open/close and smoothed columns to *df* (in-place) and return
    a dict of column-name → array for stats accumulation."""
    out: dict[str, np.ndarray] = {}

    if do_open:
        if "observation.state" in df.columns:
            state = np.stack(df["observation.state"].values).astype(np.float32)
            left = hand_closed_label(state[:, LEFT_HAND_SLICE], threshold)
            right = hand_closed_label(state[:, RIGHT_HAND_SLICE], threshold)
            df[LEFT_OPEN_KEY] = list(left[:, None])
            df[RIGHT_OPEN_KEY] = list(right[:, None])
            out[LEFT_OPEN_KEY] = left[:, None]
            out[RIGHT_OPEN_KEY] = right[:, None]
        else:
            print("  WARNING: 'observation.state' missing — skipping hand open/close labeling")

    if do_smooth:
        for src, dst, _ in SMOOTH_SPECS:
            if src not in df.columns:
                print(f"  WARNING: '{src}' missing — skipping smooth column '{dst}'")
                continue
            raw = np.stack(df[src].values).astype(np.float32)
            smoothed = moving_average(raw, window)
            df[dst] = list(smoothed)
            out[dst] = smoothed

    return df, out


# ---------------------------------------------------------------------------
# SMPL stale-frame helpers
# ---------------------------------------------------------------------------

def build_stale_mask(smpl_arr: np.ndarray) -> np.ndarray:
    """True where a frame should be removed (all-zero or frozen lead-in)."""
    n = len(smpl_arr)
    is_zero = np.all(smpl_arr == 0, axis=1)
    remove = is_zero.copy()

    diffs = np.zeros(n)
    diffs[1:] = np.sum(np.abs(smpl_arr[1:] - smpl_arr[:-1]), axis=1)

    for i in range(n):
        if is_zero[i]:
            j = i - 1
            while j >= 0 and diffs[j] == 0.0 and not is_zero[j]:
                remove[j] = True
                j -= 1

    return remove


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def copy_or_filter_video(
    src: Path, dst: Path, valid_indices: np.ndarray | None, fps: int
) -> None:
    """Copy *src* to *dst*, keeping only *valid_indices* frames when given.

    When *valid_indices* is None, falls back to a plain file copy (fastest
    path, no decode/encode).  When filtering, decodes directly from *src* and
    encodes straight to *dst* in one pass — no intermediate copy.
    """
    if valid_indices is None:
        shutil.copy2(src, dst)
        return

    in_c = av.open(str(src))
    stream = in_c.streams.video[0]
    all_frames = [f.to_ndarray(format="rgb24") for f in in_c.decode(stream)]
    in_c.close()

    filtered = [all_frames[i] for i in valid_indices if i < len(all_frames)]
    if not filtered:
        return

    out_c = av.open(str(dst), mode="w")
    out_s = out_c.add_stream("h264", rate=fps)
    h, w = filtered[0].shape[:2]
    out_s.width, out_s.height, out_s.pix_fmt = w, h, "yuv420p"
    for img in filtered:
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for pkt in out_s.encode(frame):
            out_c.mux(pkt)
    for pkt in out_s.encode():
        out_c.mux(pkt)
    out_c.close()


# ---------------------------------------------------------------------------
# LeRobot on-disk helpers
# ---------------------------------------------------------------------------

def load_info(dataset_path: Path) -> dict:
    return json.loads((dataset_path / "meta" / "info.json").read_text())


def load_episodes_meta(dataset_path: Path) -> list[dict]:
    path = dataset_path / "meta" / "episodes.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_tasks_meta(dataset_path: Path) -> list[dict]:
    path = dataset_path / "meta" / "tasks.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def get_parquet_path(dataset_path: Path, info: dict, episode_index: int) -> Path:
    pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    chunk = episode_index // info.get("chunks_size", 1000)
    return dataset_path / pattern.format(episode_chunk=chunk, episode_index=episode_index)


def get_video_keys(info: dict) -> list[str]:
    keys = info.get("video_keys", [])
    if not keys:
        keys = [
            k for k, v in info.get("features", {}).items()
            if v.get("dtype") in ("video", "image")
        ]
    return keys


def get_video_paths(dataset_path: Path, info: dict, episode_index: int) -> dict[str, Path]:
    pattern = info.get(
        "video_path",
        "videos/{video_key}/episode_{episode_index:06d}.mp4",
    )
    chunk = episode_index // info.get("chunks_size", 1000)
    return {
        key: dataset_path / pattern.format(
            video_key=key, episode_index=episode_index, episode_chunk=chunk
        )
        for key in get_video_keys(info)
    }


# ---------------------------------------------------------------------------
# Script-config validation
# ---------------------------------------------------------------------------

def validate_script_configs(dataset_paths: list[Path]) -> dict | None:
    configs: dict[str, dict] = {}
    for ds in dataset_paths:
        sc = load_info(ds).get("script_config")
        if sc is not None:
            configs[ds.name] = sc
    if not configs:
        return None

    canonical = json.dumps(next(iter(configs.values())), sort_keys=True)
    mismatched = [n for n, c in configs.items() if json.dumps(c, sort_keys=True) != canonical]
    if mismatched:
        ref = next(iter(configs))
        msg = [
            "\nERROR: script_config mismatch across datasets.",
            f"  Reference: {ref}",
            *[f"  Differs:   {n}" for n in mismatched],
            "\nDatasets recorded with different robot configs cannot be merged.",
        ]
        print("\n".join(msg))
        raise SystemExit(1)

    return next(iter(configs.values()))


# ---------------------------------------------------------------------------
# Per-dataset loading (with discard + SMPL filtering)
# ---------------------------------------------------------------------------

def load_dataset_episodes(
    dataset_path: Path,
    do_discard: bool,
    do_smpl: bool,
) -> tuple[list[dict], dict, int, int]:
    """Scan episodes, apply discard + SMPL stale-frame detection.

    Returns (episode_list, info, n_discarded, n_dropped_smpl).
    Each episode dict stores the *path* to the parquet (loaded lazily at write
    time) and the precomputed *valid_indices* mask.  The SMPL check reads only
    the SMPL column to avoid loading all parquet data upfront.
    """
    info = load_info(dataset_path)
    fps = info.get("fps", 50)
    episodes_meta = load_episodes_meta(dataset_path)

    discarded_set: set[int] = set()
    if do_discard:
        discarded_set = set(info.get("discarded_episode_indices") or [])
        if discarded_set:
            print(f"  Discarding {len(discarded_set)} episode(s): {sorted(discarded_set)}")

    n_discarded = len(discarded_set)
    n_dropped_smpl = 0
    episodes: list[dict] = []

    for ep_meta in episodes_meta:
        ep_idx = ep_meta["episode_index"]

        if ep_idx in discarded_set:
            continue

        parquet_path = get_parquet_path(dataset_path, info, ep_idx)
        if not parquet_path.exists():
            print(f"  WARNING: Missing parquet for episode {ep_idx}, skipping")
            continue

        valid_indices = None

        if do_smpl:
            # Column-pruned read: only load the SMPL column for stale detection.
            try:
                smpl_df = pd.read_parquet(parquet_path, columns=[SMPL_POSE_COLUMN])
                smpl_arr = np.vstack(
                    [np.asarray(x, dtype=np.float32) for x in smpl_df[SMPL_POSE_COLUMN]]
                )
                mask = build_stale_mask(smpl_arr)
                n_remove = int(mask.sum())
                if n_remove > 0:
                    ep_len = len(smpl_df)
                    n_zero = int(np.all(smpl_arr == 0, axis=1).sum())
                    n_frozen = n_remove - n_zero
                    pct = 100.0 * n_remove / ep_len
                    print(
                        f"  Episode {ep_idx}: removing {n_remove}/{ep_len} frames "
                        f"({pct:.1f}%) — {n_zero} zero + {n_frozen} frozen lead-in"
                    )
                    if n_remove == ep_len:
                        print(f"  Episode {ep_idx}: ALL frames stale — dropping episode")
                        n_dropped_smpl += 1
                        continue
                    valid_indices = np.where(~mask)[0]
            except Exception:
                pass  # SMPL column absent — skip detection

        episodes.append({
            "parquet_path": parquet_path,
            "source_video_paths": get_video_paths(dataset_path, info, ep_idx),
            "valid_indices": valid_indices,
            "episode_meta": ep_meta,
            "fps": fps,
        })

    return episodes, info, n_discarded, n_dropped_smpl


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _compute_stats(arr: np.ndarray) -> dict:
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
# Meta construction for merged output
# ---------------------------------------------------------------------------

# Column names that changed in the collector, old -> new. Consumers that classify
# columns by the LeRobot constants (PRTS matches OBS_STATE = "observation.state") see NO
# state at all under a non-standard name, and report it as an empty-tensor error rather
# than an unknown column. amo wrote "states" until 2026-08-07; renaming here means
# re-running prepare migrates a legacy dataset instead of it having to be re-collected.
# Delete this table once no pre-2026-08-07 amo data is still in use.
LEGACY_COLUMN_RENAMES = {"states": "observation.state"}
_MODALITY_DEFAULT_FEATURE = {"state": "observation.state", "action": "action"}


def migrate_legacy_meta(info: dict, modality: dict | None) -> None:
    """Apply LEGACY_COLUMN_RENAMES to a reference dataset's meta, in place.

    Both halves matter: the feature table names the column, and each modality group's
    ``original_key`` points at it. Renaming only the dataframe would leave the meta
    pointing at a column that no longer exists -- which the phantom-key pruning below
    would then delete, silently dropping the whole state group.
    """
    features = info.get("features") or {}
    for old, new in LEGACY_COLUMN_RENAMES.items():
        if old in features:
            features[new] = features.pop(old)
    if not modality:
        return
    for section, groups in modality.items():
        default = _MODALITY_DEFAULT_FEATURE.get(section)
        for spec in (groups or {}).values():
            origin = (spec or {}).get("original_key")
            if origin in LEGACY_COLUMN_RENAMES:
                new = LEGACY_COLUMN_RENAMES[origin]
                # the collector omits original_key when it equals the section default
                if new == default:
                    spec.pop("original_key", None)
                else:
                    spec["original_key"] = new


def per_episode_stats(df: pd.DataFrame) -> dict:
    """Per-column min/max/mean/std/count for one episode's numeric columns.

    LeRobot v2.1 expects meta/episodes_stats.jsonl, and its loader treats a dataset
    without one as incomplete: it falls back to fetching the repo from the Hub, which on
    an offline cluster surfaces as a DNS error rather than "your meta is missing a file"
    (that is how PRTS failed to read an amo dataset on 2026-08-07). Video columns are
    skipped — the frames live in videos/, not the parquet.
    """
    stats: dict = {}
    for col in df.columns:
        first = df[col].iloc[0]
        if isinstance(first, str):
            continue
        try:
            arr = np.vstack([np.asarray(x, dtype=np.float32).reshape(-1) for x in df[col]])
        except Exception:  # noqa: BLE001  — ragged or non-numeric column
            continue
        stats[col] = {
            "min": arr.min(0).tolist(),
            "max": arr.max(0).tolist(),
            "mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(),
            "count": [int(arr.shape[0])],
        }
    return stats


def action_signature(modality: dict | None) -> frozenset[str]:
    """The dataset's action-space fingerprint: its meta/modality.json action keys.

    This is what makes two datasets mergeable or not. Each whole-body controller records
    a different action (sonic: motion_token+hands, heft: ref_qpos+hands, amo: the teleop
    command stream), and merging across them produces a dataset whose rows come from all
    inputs while its meta describes only the first one -- silently unusable.
    """
    if not modality:
        return frozenset()
    return frozenset((modality.get("action") or {}).keys())


def assert_same_controller(dataset_paths: list[Path]) -> None:
    """Refuse to merge datasets recorded through different controllers."""
    seen: dict[frozenset[str], list[Path]] = {}
    for ds in dataset_paths:
        p = ds / "meta" / "modality.json"
        if not p.exists():
            continue
        seen.setdefault(action_signature(json.loads(p.read_text())), []).append(ds)
    if len(seen) <= 1:
        return
    groups = "\n".join(
        f"  [{', '.join(sorted(sig)) or '(no action keys)'}]\n"
        + "".join(f"      {d}\n" for d in paths)
        for sig, paths in seen.items()
    )
    raise SystemExit(
        "ERROR: these datasets were recorded through different controllers and cannot be "
        "merged -- their action spaces are not the same quantity, so a merged dataset would "
        "describe one and contain all:\n" + groups
        + "  Merge each controller's datasets separately."
    )


def build_merged_info(
    reference_info: dict,
    total_episodes: int,
    total_frames: int,
    video_keys: list[str],
    produced_columns: set[str],
    written_columns: set[str],
    script_config: dict | None,
) -> dict:
    info = reference_info.copy()
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_videos"] = total_episodes * len(video_keys)
    info["discarded_episode_indices"] = []

    if script_config is not None:
        info["script_config"] = script_config

    # Declare ONLY the columns this run actually wrote. Keying this off the --open/--smooth
    # flags instead was how amo datasets ended up advertising six teleop.* features their
    # parquet never had: the stages skip (with a warning) when the source columns are
    # absent, so a flag says what was asked for, not what exists. Anything that walks the
    # declared keys -- gr00t's dataset statistics, for one -- then dies on a KeyError.
    features = info.setdefault("features", {})
    for key, dim in [*OPEN_SPECS, *((dst, dim) for _, dst, dim in SMOOTH_SPECS)]:
        if key in produced_columns and key not in features:
            features[key] = {"dtype": "float32", "shape": [dim], "names": None}

    # Drop anything the reference info advertised that this output does not contain.
    # Video features live in videos/, not the parquet, so they are never dropped here.
    if written_columns:
        phantom = [k for k, v in features.items()
                   if (v or {}).get("dtype") != "video" and k not in written_columns]
        for k in phantom:
            features.pop(k)
        if phantom:
            print(f"  meta: dropped {len(phantom)} feature(s) with no column in the output: "
                  f"{', '.join(sorted(phantom))}")

    return info


def build_merged_modality(
    reference_modality: dict | None,
    produced_columns: set[str],
    written_columns: set[str],
) -> dict:
    """Reference modality plus the hand keys this run actually produced, minus phantoms.

    Same rule as build_merged_info: a key is declared only if its column exists. The
    reference modality otherwise carries over as-is -- it describes the controller's own
    action space, and assert_same_controller() has already established that every input
    dataset shares it -- except for entries whose `original_key` names a column the output
    does not have. Those are dropped, which is also what makes re-running prepare on an
    already-broken dataset repair it.
    """
    modality = copy.deepcopy(reference_modality) if reference_modality else {}
    if written_columns:
        for group in ("state", "action"):
            entries = modality.get(group) or {}
            phantom = [k for k, spec in entries.items()
                       if (spec or {}).get("original_key") not in (None, *written_columns)]
            for k in phantom:
                entries.pop(k)
            if phantom:
                print(f"  meta: dropped {group} key(s) with no column in the output: "
                      f"{', '.join(sorted(phantom))}")
    action = modality.setdefault("action", {})

    for short, original, dim in [
        ("left_hand_open", LEFT_OPEN_KEY, 1),
        ("right_hand_open", RIGHT_OPEN_KEY, 1),
        *((dst.split(".", 1)[1], dst, dim) for _, dst, dim in SMOOTH_SPECS),
    ]:
        if original in produced_columns and short not in action:
            action[short] = {"start": 0, "end": dim, "original_key": original}

    return modality


def load_reference_modality(dataset_paths: list[Path]) -> dict | None:
    for ds in dataset_paths:
        p = ds / "meta" / "modality.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_merged_dataset(
    output_path: Path,
    all_episodes: list[dict],
    reference_info: dict,
    reference_modality: dict | None,
    tasks_meta: list[dict],
    script_config: dict | None,
    do_open: bool,
    do_smooth: bool,
    closure_threshold: float,
    window: int,
    video_workers: int = 4,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    meta_dir = output_path / "meta"
    meta_dir.mkdir(exist_ok=True)
    task_lookup = {t["task_index"]: t["task"] for t in tasks_meta}

    fps = reference_info.get("fps", 50)
    chunks_size = reference_info.get("chunks_size", 1000)
    data_pattern = reference_info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    video_pattern = reference_info.get(
        "video_path",
        "videos/{video_key}/episode_{episode_index:06d}.mp4",
    )
    video_keys = get_video_keys(reference_info)
    migrate_legacy_meta(reference_info, reference_modality)

    total_frames = 0
    episodes_jsonl: list[dict] = []
    episodes_stats: list[dict] = []

    stat_accum: dict[str, list[np.ndarray]] = {}
    if do_open:
        stat_accum[LEFT_OPEN_KEY] = []
        stat_accum[RIGHT_OPEN_KEY] = []
    if do_smooth:
        for _, dst, _ in SMOOTH_SPECS:
            stat_accum[dst] = []

    hand_stats_left_closed: list[float] = []
    hand_stats_right_closed: list[float] = []
    # Columns the hand stages actually wrote; the meta is built from this, not from the
    # --open/--smooth flags (a stage skips when its source column is absent).
    produced_columns: set[str] = set()
    # Every column present in the written parquet, so the meta can drop entries that
    # describe columns this output does not have.
    written_columns: set[str] = set()
    video_futures = []

    with ThreadPoolExecutor(max_workers=video_workers) as executor:
        for new_idx, ep in enumerate(tqdm(all_episodes, desc="Writing episodes")):
            episode_chunk = new_idx // chunks_size

            # Lazy load: read the full parquet only now, one at a time.
            df = pd.read_parquet(ep["parquet_path"])
            df = df.rename(columns={k: v for k, v in LEGACY_COLUMN_RENAMES.items()
                                    if k in df.columns})

            # Apply SMPL row filter if needed.
            valid = ep["valid_indices"]
            if valid is not None:
                df = df.iloc[valid].reset_index(drop=True)
                if "timestamp" in df.columns:
                    df["timestamp"] -= df["timestamp"].iloc[0]

            ep_len = len(df)

            # Remap task_index into the merged task table (identity map = no-op).
            ep_task_strings = ep["episode_meta"].get("tasks", [])
            task_map = ep.get("task_map")
            if task_map is not None and "task_index" in df.columns:
                remapped = df["task_index"].map(task_map)
                if remapped.isna().any():
                    bad = sorted(set(df["task_index"][remapped.isna()].tolist()))
                    raise SystemExit(
                        f"ERROR: {ep['parquet_path']} has task_index values {bad} "
                        "not present in its meta/tasks.jsonl"
                    )
                df["task_index"] = remapped.astype("int64")
                ep_task_strings = [
                    task_lookup[i] for i in sorted(df["task_index"].unique().tolist())
                ]

            # Fix index columns with numpy (avoids Python-level loops).
            df["episode_index"] = new_idx
            df["index"] = np.arange(total_frames, total_frames + ep_len)
            df["frame_index"] = np.arange(ep_len)
            if "timestamp" in df.columns:
                df["timestamp"] = np.arange(ep_len, dtype=np.float32) / fps

            # Compute hand fields.
            df, hand_out = add_hand_columns(df, closure_threshold, window, do_open, do_smooth)
            produced_columns.update(hand_out.keys())
            for key, arr in hand_out.items():
                stat_accum[key].append(arr)
            if LEFT_OPEN_KEY in hand_out:
                hand_stats_left_closed.append(float(hand_out[LEFT_OPEN_KEY].mean()))
            if RIGHT_OPEN_KEY in hand_out:
                hand_stats_right_closed.append(float(hand_out[RIGHT_OPEN_KEY].mean()))

            # Write parquet (main thread).
            parquet_rel = data_pattern.format(episode_chunk=episode_chunk, episode_index=new_idx)
            parquet_path = output_path / parquet_rel
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path, index=False)
            written_columns.update(df.columns)
            episodes_stats.append(
                {"episode_index": new_idx, "stats": per_episode_stats(df)})

            # Submit video ops to the thread pool (non-blocking).
            # copy_or_filter_video decodes from source directly → no double-copy.
            src_fps = ep["fps"]
            for vkey in video_keys:
                src_video = ep["source_video_paths"].get(vkey)
                if not (src_video and src_video.exists()):
                    continue
                dst_rel = video_pattern.format(
                    video_key=vkey, episode_index=new_idx, episode_chunk=episode_chunk
                )
                dst_video = output_path / dst_rel
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                video_futures.append(
                    executor.submit(copy_or_filter_video, src_video, dst_video, valid, src_fps)
                )

            episodes_jsonl.append({
                "episode_index": new_idx,
                "tasks": ep_task_strings,
                "length": ep_len,
            })
            total_frames += ep_len

    # Propagate any exception from a video worker.
    for fut in video_futures:
        fut.result()

    # Write meta/info.json
    merged_info = build_merged_info(
        reference_info,
        total_episodes=len(all_episodes),
        total_frames=total_frames,
        video_keys=video_keys,
        produced_columns=produced_columns,
        written_columns=written_columns,
        script_config=script_config,
    )
    if tasks_meta:
        merged_info["total_tasks"] = len(tasks_meta)
    (meta_dir / "info.json").write_text(json.dumps(merged_info, indent=4))

    # Write meta/modality.json
    merged_modality = build_merged_modality(
        reference_modality, produced_columns, written_columns)
    (meta_dir / "modality.json").write_text(json.dumps(merged_modality, indent=4))

    # Write meta/stats.json (hand fields only)
    stats: dict = {}
    for key, chunks in stat_accum.items():
        if chunks:
            stats[key] = _compute_stats(np.concatenate(chunks, axis=0))
    if stats:
        (meta_dir / "stats.json").write_text(json.dumps(stats, indent=4))

    # Write meta/episodes.jsonl
    (meta_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in episodes_jsonl)
    )

    # Write meta/episodes_stats.jsonl (LeRobot v2.1 requires it — see per_episode_stats)
    (meta_dir / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in episodes_stats)
    )

    # Write meta/tasks.jsonl
    if tasks_meta:
        (meta_dir / "tasks.jsonl").write_text(
            "".join(json.dumps(t) + "\n" for t in tasks_meta)
        )

    # Summary line for hand stats
    if hand_stats_left_closed:
        lm = np.mean(hand_stats_left_closed)
        rm = np.mean(hand_stats_right_closed)
        print(f"  Hand: left closed={lm:.3f}, right closed={rm:.3f} (mean over episodes)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified LeRobot dataset preparation: discard → SMPL clean → "
            "merge → hand fields → meta update."
        )
    )
    parser.add_argument(
        "--dataset-paths",
        nargs="*",
        type=Path,
        default=None,
        metavar="DIR",
        help="One or more source dataset directories.",
    )
    parser.add_argument(
        "--dataset-list",
        type=Path,
        default=None,
        metavar="FILE",
        help="Text file with one dataset path per line (# lines are comments).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Destination directory for the processed dataset (never in-place). "
            f"Defaults to {DEFAULT_OUTPUT_ROOT}/<dataset-name> for a single input. "
            "Required when merging multiple datasets."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        metavar="DIR",
        help=(
            "Root directory under which the default output is placed when "
            f"--output-path is not given (default: {DEFAULT_OUTPUT_ROOT})."
        ),
    )
    parser.add_argument(
        "--closure-threshold",
        type=float,
        default=DEFAULT_CLOSURE_THRESHOLD,
        help=f"Finger closure > this → hand closed (default: {DEFAULT_CLOSURE_THRESHOLD})",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Centered moving-average window in frames (default: {DEFAULT_WINDOW})",
    )
    parser.add_argument(
        "--task-overrides",
        nargs="+",
        default=None,
        metavar="TASK",
        help=(
            "Replace each input dataset's task string at merge time (sources are "
            "never modified). One per dataset, aligned with --dataset-paths order; "
            "each overridden dataset must be single-task. Merged task_index values "
            "are reassigned by deduplicating task strings and the parquet "
            "task_index column is rewritten accordingly in the output."
        ),
    )
    parser.add_argument("--no-discard", action="store_true", help="Skip discarded episode removal.")
    parser.add_argument("--no-smpl", action="store_true", help="Skip stale SMPL frame removal.")
    parser.add_argument("--no-open", action="store_true", help="Skip hand open/close labeling.")
    parser.add_argument("--no-smooth", action="store_true", help="Skip hand signal smoothing.")
    parser.add_argument(
        "--video-workers",
        type=int,
        default=min(4, os.cpu_count() or 4),
        metavar="N",
        help="Thread-pool size for concurrent video copy/encode (default: min(4, cpu_count)).",
    )
    args = parser.parse_args()

    # Collect dataset paths
    dataset_paths: list[Path] = list(args.dataset_paths or [])
    if args.dataset_list:
        for line in args.dataset_list.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                dataset_paths.append(Path(line))

    if not dataset_paths:
        parser.error("No dataset paths provided. Use --dataset-paths or --dataset-list.")

    for ds in dataset_paths:
        if not ds.exists():
            parser.error(f"Dataset path does not exist: {ds}")
        if not (ds / "meta" / "info.json").exists():
            parser.error(f"Not a valid LeRobot dataset (missing meta/info.json): {ds}")

    if args.output_path is not None:
        output_path = args.output_path.resolve()
    elif len(dataset_paths) == 1:
        output_path = args.output_root.resolve() / dataset_paths[0].resolve().name
    else:
        parser.error(
            "--output-path is required when merging multiple datasets."
        )

    if output_path.exists():
        parser.error(
            f"Output path already exists: {output_path}\n"
            "Remove or rename it before running."
        )

    do_discard = not args.no_discard
    do_smpl = not args.no_smpl
    do_open = not args.no_open
    do_smooth = not args.no_smooth

    print("=" * 70)
    print("  LeRobot Dataset Preparation Pipeline")
    print("=" * 70)
    print(f"  Input datasets  : {len(dataset_paths)}")
    for ds in dataset_paths:
        print(f"    - {ds}")
    print(f"  Output          : {output_path}")
    print(f"  Discard         : {do_discard}")
    print(f"  Remove stale SMPL: {do_smpl}")
    print(f"  Hand open/close : {do_open}  (threshold={args.closure_threshold})")
    print(f"  Hand smoothing  : {do_smooth}  (window={args.window})")
    print(f"  Video workers   : {args.video_workers}")
    print("=" * 70)

    # Validate script configs when merging
    if len(dataset_paths) > 1:
        print("\nValidating script_config across datasets...")
        script_config = validate_script_configs(dataset_paths)
        print("  All datasets share the same robot configuration.")
    else:
        script_config = load_info(dataset_paths[0]).get("script_config")

    # Collect tasks: dedupe by task *string* and reassign contiguous merged
    # task_index values.  Per-dataset {old task_index -> merged task_index} maps
    # are applied to the parquet task_index column at write time, so sources with
    # colliding indices (every single-task dataset uses 0) or --task-overrides
    # renames merge into a proper multi-task dataset without touching the inputs.
    task_overrides: list[str] | None = args.task_overrides
    if task_overrides and len(task_overrides) != len(dataset_paths):
        parser.error(
            f"--task-overrides needs one task per dataset "
            f"({len(task_overrides)} given, {len(dataset_paths)} datasets)"
        )
    merged_task_strings: list[str] = []
    task_maps: list[dict[int, int]] = []          # aligned with dataset_paths
    for i, ds in enumerate(dataset_paths):
        src_tasks = load_tasks_meta(ds)
        if task_overrides:
            if len(src_tasks) != 1:
                raise SystemExit(
                    f"ERROR: --task-overrides expects single-task inputs, "
                    f"but {ds} has {len(src_tasks)} task(s)"
                )
            src_strings = {src_tasks[0]["task_index"]: task_overrides[i]}
        else:
            src_strings = {t["task_index"]: t["task"] for t in src_tasks}
        mapping: dict[int, int] = {}
        for old_idx, task_str in src_strings.items():
            if task_str not in merged_task_strings:
                merged_task_strings.append(task_str)
            mapping[old_idx] = merged_task_strings.index(task_str)
        task_maps.append(mapping)
    all_tasks_meta: list[dict] = [
        {"task_index": i, "task": s} for i, s in enumerate(merged_task_strings)
    ]

    # The merged meta is built from the FIRST dataset's modality, so every input has to
    # describe the same action space -- check before writing anything.
    assert_same_controller(dataset_paths)

    # Reference modality (first dataset that has one)
    reference_modality = load_reference_modality(dataset_paths)

    # Process each dataset
    all_episodes: list[dict] = []
    reference_info: dict | None = None
    total_discarded = 0
    total_dropped_smpl = 0
    total_input_episodes = 0

    for ds, task_map in zip(dataset_paths, task_maps):
        print(f"\nLoading: {ds}")
        episodes, info, n_disc, n_drop = load_dataset_episodes(ds, do_discard, do_smpl)
        for ep in episodes:
            ep["task_map"] = task_map
        if reference_info is None:
            reference_info = info
        total_input_episodes += len(load_episodes_meta(ds))
        total_discarded += n_disc
        total_dropped_smpl += n_drop
        all_episodes.extend(episodes)
        print(
            f"  {ds.name}: {len(episodes)} episodes kept "
            f"({n_disc} discarded, {n_drop} dropped by SMPL)"
        )

    if not all_episodes:
        print("\nERROR: No valid episodes after filtering.")
        raise SystemExit(1)

    # Write output
    print(f"\nWriting output to {output_path} ...")
    write_merged_dataset(
        output_path=output_path,
        all_episodes=all_episodes,
        reference_info=reference_info,  # type: ignore[arg-type]
        reference_modality=reference_modality,
        tasks_meta=all_tasks_meta,
        script_config=script_config,
        do_open=do_open,
        do_smooth=do_smooth,
        closure_threshold=args.closure_threshold,
        window=args.window,
        video_workers=args.video_workers,
    )

    # Final summary
    kept = len(all_episodes)
    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)
    print(f"  Input episodes  : {total_input_episodes}")
    print(f"  Discarded       : {total_discarded}")
    print(f"  Dropped (SMPL)  : {total_dropped_smpl}")
    print(f"  Output episodes : {kept}")
    print(f"  Output path     : {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
