"""Retag a single-task LeRobot dataset IN PLACE: set its task string and task_index.

NOTE: for building a multi-task dataset you normally do NOT need this — use
prepare_dataset.py's --task-overrides, which renames/remaps tasks at merge time
and leaves the source datasets untouched.  This tool is for permanently fixing
a dataset whose recorded prompt is wrong (e.g. a "test" placeholder):

    python gear_sonic/scripts/retag_task.py \
        --dataset outputs/box_sessions --task-index 0 --task "pick up the paper box"

Touches: meta/tasks.jsonl, meta/episodes.jsonl (tasks field), and — only when
task_index changes — the `task_index` column of every data/*.parquet (rewritten
atomically via a temp file).  First run backs up tasks.jsonl/episodes.jsonl as
*.bak (existing backups are never overwritten).  Idempotent: re-running with
the same arguments is a no-op.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def backup_once(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root (contains meta/ and data/).")
    parser.add_argument("--task", required=True, help="New task string (deploy prompts must match it exactly).")
    parser.add_argument("--task-index", type=int, required=True,
                        help="New task_index; must be unique across the datasets you intend to merge.")
    args = parser.parse_args()

    meta_dir = args.dataset / "meta"
    tasks_path = meta_dir / "tasks.jsonl"
    episodes_path = meta_dir / "episodes.jsonl"

    tasks = load_jsonl(tasks_path)
    if len(tasks) != 1:
        raise SystemExit(f"ERROR: expected a single-task dataset, found {len(tasks)} tasks in {tasks_path}")
    old_index = tasks[0]["task_index"]
    old_task = tasks[0]["task"]

    print(f"Dataset : {args.dataset}")
    print(f"Task    : {old_index} {old_task!r} -> {args.task_index} {args.task!r}")

    # meta/tasks.jsonl
    if tasks[0] != {"task_index": args.task_index, "task": args.task}:
        backup_once(tasks_path)
        write_jsonl(tasks_path, [{"task_index": args.task_index, "task": args.task}])

    # meta/episodes.jsonl — the per-episode task string list
    episodes = load_jsonl(episodes_path)
    if any(ep.get("tasks") != [args.task] for ep in episodes):
        backup_once(episodes_path)
        for ep in episodes:
            ep["tasks"] = [args.task]
        write_jsonl(episodes_path, episodes)

    # data/*.parquet task_index column (only when the index actually changes)
    parquets = sorted(args.dataset.glob("data/chunk-*/*.parquet"))
    if not parquets:
        raise SystemExit(f"ERROR: no parquet files under {args.dataset}/data")
    rewritten = 0
    for pq in parquets:
        col = pd.read_parquet(pq, columns=["task_index"])["task_index"]
        uniq = set(col.unique().tolist())
        if uniq == {args.task_index}:
            continue
        if uniq != {old_index}:
            raise SystemExit(f"ERROR: {pq} has unexpected task_index values {sorted(uniq)}")
        df = pd.read_parquet(pq)
        df["task_index"] = args.task_index
        df["task_index"] = df["task_index"].astype("int64")
        tmp = pq.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(pq)
        rewritten += 1

    print(f"Episodes: {len(episodes)} | parquets rewritten: {rewritten}/{len(parquets)}")
    print("Done.")


if __name__ == "__main__":
    main()
