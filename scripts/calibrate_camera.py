#!/usr/bin/env python3
"""SafeCity AI: Camera Calibration Tool.

Computes 3x3 planar homography matrix from road reference points with known real-world
metric distances. Validates reprojection error and saves calibration to YAML.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import yaml

from safecity.calibration.homography import HomographyCalibrator

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("safecity.calibrate_cli")


def calibrate_from_config(config_path: str, output_path: str) -> HomographyCalibrator:
    """Reads point correspondences from template config, computes H, and writes result."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    pts = data.get("points", [])
    if len(pts) < 4:
        raise ValueError(f"Need at least 4 point correspondences in {cfg_file}, found {len(pts)}")

    img_pts = []
    wld_pts = []
    for p in pts:
        img_pts.append(tuple(p["image_coords"]))
        wld_pts.append(tuple(p["world_coords"]))
        logger.info(f"Point {p.get('id', '?')}: Pixel {p['image_coords']} -> World {p['world_coords']} m ({p.get('description', '')})")

    calibrator = HomographyCalibrator.from_points(
        image_points=img_pts,
        world_points=wld_pts,
        camera_id=data.get("camera_id", "camera_01")
    )

    logger.info(f"✅ Computed Homography Matrix successfully!")
    logger.info(f"Mean Reprojection Error: {calibrator.reprojection_error:.4f} meters")
    calibrator.to_yaml(output_path)
    return calibrator


def main():
    parser = argparse.ArgumentParser(description="SafeCity AI Camera Calibration Tool")
    parser.add_argument("--config", type=str, default="configs/calibration_template.yaml", help="Path to calibration YAML")
    parser.add_argument("--output", type=str, default="configs/camera_calibration.yaml", help="Output path for calibrated YAML")
    args = parser.parse_args()

    calibrator = calibrate_from_config(args.config, args.output)
    print("\nCalibration Summary:")
    print(f"Status: {'CALIBRATED' if calibrator.is_calibrated else 'UNAVAILABLE'}")
    print(f"Reprojection Error: {calibrator.reprojection_error:.4f} meters")
    print(f"Matrix saved to: {args.output}")


if __name__ == "__main__":
    main()
