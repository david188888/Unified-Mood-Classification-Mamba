#!/usr/bin/env python
"""Generate a subset MTG-Jamendo label CSV that matches folder splits.

Creates data/MTG-Jamendo/mtg_labels.csv with exact sizes:
  - train: 2100
  - val:   750
  - test:  550

Rules:
  1) Prefer the user's curated file lists under data/MTG-Jamendo/{train,val,test}.
  2) Drop files whose track_id is not present in mtg_split_labels.csv (no labels).
  3) If a split is short, deterministically top-up from mtg_split_labels.csv (same split)
     until the target size is reached.
  4) Write audio_path as the REAL existing numeric path: {track_id%100:02d}/{track_id}.low.mp3
"""

from __future__ import annotations

import csv
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def _parse_track_id_from_filename(filename: str) -> Optional[int]:
    # Expected format: track_0000948.low.mp3
    if filename.startswith("track_") and filename.endswith(".low.mp3"):
        try:
            tid_str = filename.split("_")[1].split(".")[0]
            return int(tid_str)
        except Exception:
            return None
    m = re.search(r"\d+", filename)
    return int(m.group()) if m else None


def _numeric_audio_relpath(track_id: int) -> str:
    return f"{track_id % 100:02d}/{track_id}.low.mp3"


def generate_mtg_labels_csv(
    base_dir: str,
    seed: int = 42,
    target_counts: Optional[Dict[str, int]] = None,
    source_csv_name: str = "mtg_split_labels.csv",
    out_csv_name: str = "mtg_labels.csv",
) -> str:
    if target_counts is None:
        target_counts = {"train": 2100, "val": 750, "test": 550}

    source_csv = os.path.join(base_dir, source_csv_name)
    out_csv = os.path.join(base_dir, out_csv_name)

    if not os.path.exists(source_csv):
        raise FileNotFoundError(source_csv)

    # Load metadata + per-split pools
    metadata: Dict[int, Dict[str, str]] = {}
    pool_by_split: Dict[str, List[int]] = defaultdict(list)
    with open(source_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tid = int(row["track_id"])
            except Exception:
                continue
            metadata[tid] = row
            split = (row.get("split") or "").strip()
            if split in ("train", "val", "test"):
                pool_by_split[split].append(tid)

    for split in list(pool_by_split.keys()):
        pool_by_split[split] = sorted(set(pool_by_split[split]))

    print(f"Loaded metadata for {len(metadata)} tracks from {source_csv}")

    # Read folder selections
    selected_by_split: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    dropped_no_metadata: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for split in ("train", "val", "test"):
        split_dir = os.path.join(base_dir, split)
        files = glob.glob(os.path.join(split_dir, "*.mp3")) if os.path.isdir(split_dir) else []
        for fp in files:
            fn = os.path.basename(fp)
            tid = _parse_track_id_from_filename(fn)
            if tid is None:
                continue
            if tid not in metadata:
                dropped_no_metadata[split] += 1
                continue
            selected_by_split[split].append(tid)
        selected_by_split[split] = sorted(set(selected_by_split[split]))

    print("Folder-selected (with labels) counts:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {len(selected_by_split[split])} (dropped_no_metadata={dropped_no_metadata[split]})")

    rng = np.random.default_rng(seed)
    final_by_split: Dict[str, List[int]] = {"train": [], "val": [], "test": []}

    for split in ("train", "val", "test"):
        target = int(target_counts[split])
        chosen = list(selected_by_split[split])

        # Downsample if too many
        if len(chosen) > target:
            idx = rng.choice(len(chosen), size=target, replace=False)
            chosen = sorted([chosen[i] for i in idx])

        # Top-up if too few
        if len(chosen) < target:
            need = target - len(chosen)
            already: Set[int] = set(chosen)
            pool = list(pool_by_split.get(split, []))
            rng.shuffle(pool)
            added = 0
            for tid in pool:
                if tid in already:
                    continue
                # Ensure audio exists (numeric dirs)
                rel = _numeric_audio_relpath(tid)
                if not os.path.exists(os.path.join(base_dir, rel)):
                    continue
                chosen.append(tid)
                already.add(tid)
                added += 1
                if added >= need:
                    break

            if len(chosen) != target:
                raise RuntimeError(f"Unable to reach target for {split}: got {len(chosen)} vs {target}")

        final_by_split[split] = sorted(chosen)

    # Write output CSV
    fieldnames = ["split", "track_id", "artist_id", "album_id", "duration", "mood_tags", "audio_path"]
    out_rows: List[Dict[str, str]] = []
    for split in ("train", "val", "test"):
        for tid in final_by_split[split]:
            row = metadata[tid].copy()
            row["split"] = split
            row["track_id"] = str(tid)
            row["audio_path"] = _numeric_audio_relpath(tid)
            out_rows.append({k: row.get(k, "") for k in fieldnames})

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {out_csv}")
    print("Final counts:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {len(final_by_split[split])}")

    return out_csv


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    mtg_dir = os.path.join(repo_root, "data", "MTG-Jamendo")
    generate_mtg_labels_csv(base_dir=mtg_dir)
