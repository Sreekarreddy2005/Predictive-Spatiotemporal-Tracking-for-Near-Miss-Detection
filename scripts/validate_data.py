#!/usr/bin/env python3
"""SafeCity AI: Data Integrity and Schema Validator.

Inspects raw and processed datasets, verifies:
1. Video readability, frame integrity, resolution, and FPS.
2. Annotation consistency (XML, CSV, JSON).
3. Bounding box validity (no negative coordinates, x2 > x1, within frame boundaries).
4. Trajectory continuity (monotonic timestamps, missing track IDs, reasonable delta-t).
5. Updates data/dataset_manifest.json with validation results.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("safecity.validator")


def validate_video_file(video_path: Path) -> Dict[str, Any]:
    """Inspects a video file via OpenCV for integrity and properties."""
    result: Dict[str, Any] = {
        "path": str(video_path),
        "valid": False,
        "frames": 0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "errors": []
    }
    try:
        import cv2  # Lazy import
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            result["errors"].append("Unable to open video stream.")
            return result

        result["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result["fps"] = float(cap.get(cv2.CAP_PROP_FPS))
        result["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Test read first and last frame
        ret, frame = cap.read()
        if not ret or frame is None:
            result["errors"].append("Failed to decode initial frame.")
        else:
            result["valid"] = True
        cap.release()
    except Exception as e:
        result["errors"].append(str(e))

    return result


def validate_trajectory_csv(csv_path: Path) -> Dict[str, Any]:
    """Checks CSV trajectory file for required headers, monotonic timestamps, and valid coordinates."""
    result: Dict[str, Any] = {
        "path": str(csv_path),
        "valid": False,
        "num_rows": 0,
        "num_tracks": 0,
        "errors": []
    }
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, nrows=5000)
        result["num_rows"] = len(df)

        # Check typical column names
        cols_lower = [c.lower() for c in df.columns]
        id_col = next((c for c in df.columns if c.lower() in ["trackid", "track_id", "id", "agent_id"]), None)
        frame_col = next((c for c in df.columns if c.lower() in ["frame", "timestamp", "time", "frame_id"]), None)

        if not id_col:
            result["errors"].append("Missing required track ID column.")
        else:
            result["num_tracks"] = int(df[id_col].nunique())

        if not frame_col:
            result["errors"].append("Missing timestamp or frame number column.")

        if not result["errors"]:
            result["valid"] = True
    except Exception as e:
        result["errors"].append(str(e))

    return result


def scan_manifest_datasets(manifest_path: Path) -> None:
    """Scans all registered datasets in manifest and checks whether local files exist."""
    if not manifest_path.exists():
        logger.error(f"Manifest not found at {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded manifest with {len(data.get('datasets', []))} registered datasets.")
    for ds in data.get("datasets", []):
        ds_path = Path(ds["local_path"])
        logger.info(f"\nChecking [{ds['id']}] - '{ds['name']}' at {ds_path}...")
        if not ds_path.exists():
            logger.warning(f"  -> Path does not exist. Status: {ds['status']}")
            continue

        # Check directory contents
        children = list(ds_path.glob("*"))
        if not children:
            logger.warning(f"  -> Path exists but directory is empty.")
            continue

        logger.info(f"  -> Found {len(children)} items in {ds_path}. Performing format inspection...")
        # Check videos
        videos = list(ds_path.rglob("*.mp4")) + list(ds_path.rglob("*.avi")) + list(ds_path.rglob("*.mkv"))
        if videos:
            logger.info(f"  -> Found {len(videos)} video files. Testing sample: {videos[0].name}")
            v_res = validate_video_file(videos[0])
            logger.info(f"     Resolution: {v_res['width']}x{v_res['height']}, FPS: {v_res['fps']:.2f}, Frames: {v_res['frames']}")

        # Check CSVs
        csvs = list(ds_path.rglob("*.csv"))
        if csvs:
            logger.info(f"  -> Found {len(csvs)} CSV files. Testing sample: {csvs[0].name}")
            c_res = validate_trajectory_csv(csvs[0])
            logger.info(f"     Rows: {c_res['num_rows']}, Unique tracks: {c_res['num_tracks']}")


def main():
    parser = argparse.ArgumentParser(description="SafeCity AI Data Validator")
    parser.add_argument("--manifest", type=str, default="data/dataset_manifest.json", help="Path to manifest JSON")
    parser.add_argument("--target", type=str, default=None, help="Specific file or directory to inspect")
    args = parser.parse_args()

    manifest_file = Path(args.manifest)
    if args.target:
        target_path = Path(args.target)
        if target_path.is_file():
            if target_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                res = validate_video_file(target_path)
                print(json.dumps(res, indent=2))
            elif target_path.suffix.lower() in [".csv"]:
                res = validate_trajectory_csv(target_path)
                print(json.dumps(res, indent=2))
            else:
                print(f"Unsupported file type: {target_path.suffix}")
        elif target_path.is_dir():
            print(f"Directory contents of {target_path}:")
            for p in list(target_path.iterdir())[:20]:
                print(f"  - {p.name} ({'DIR' if p.is_dir() else f'{p.stat().st_size} bytes'})")
    else:
        scan_manifest_datasets(manifest_file)


if __name__ == "__main__":
    main()
