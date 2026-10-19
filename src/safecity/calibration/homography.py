"""SafeCity AI: Camera Calibration & Ground-Plane Homography.

Transforms image-space pixel coordinates (u, v) into real-world ground-plane
metric coordinates (X, Y) in meters, and validates reprojection errors.
Maintains strict separation between pixel space and world space.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import yaml

logger = logging.getLogger("safecity.calibration")


class HomographyCalibrator:
    """Estimates and applies 3x3 planar homography mapping road plane <-> image plane."""

    # Default vehicle physical footprint dimensions [length_m, width_m] by class
    DEFAULT_FOOTPRINTS_METERS = {
        "car": (4.5, 1.8),
        "bus": (12.0, 2.5),
        "truck": (10.0, 2.5),
        "motorcycle": (2.0, 0.8),
        "bicycle": (1.8, 0.6),
        "pedestrian": (0.6, 0.6)
    }

    def __init__(
        self,
        homography_matrix: Optional[np.ndarray] = None,
        camera_id: str = "default_cam",
        reprojection_error: Optional[float] = None,
        units: str = "meters"
    ) -> None:
        self.camera_id = camera_id
        self.units = units
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.reprojection_error: Optional[float] = reprojection_error

        if homography_matrix is not None:
            self.set_homography(homography_matrix, reprojection_error)

    @property
    def is_calibrated(self) -> bool:
        """Returns True only when a valid, invertible homography matrix is active."""
        return self.H is not None and self.H_inv is not None

    def set_homography(self, H: np.ndarray, reprojection_error: Optional[float] = None) -> None:
        """Sets homography matrix and computes its inverse."""
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"Homography matrix must have shape (3, 3), got {H.shape}")

        det = np.linalg.det(H)
        if np.isclose(det, 0.0):
            raise ValueError("Homography matrix is singular (determinant is 0) and cannot be inverted.")

        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.reprojection_error = reprojection_error
        logger.info(f"Calibrator '{self.camera_id}' initialized. Reprojection error: {reprojection_error}")

    @classmethod
    def from_points(
        cls,
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]],
        camera_id: str = "cam_01"
    ) -> HomographyCalibrator:
        """Computes homography from 4 or more corresponding (image, world) point pairs.

        image_points: List of [u, v] in pixels.
        world_points: List of [X, Y] in meters on the road plane.
        """
        if len(image_points) < 4 or len(world_points) < 4:
            raise ValueError("At least 4 coplanar point correspondences are required for planar homography.")
        if len(image_points) != len(world_points):
            raise ValueError(f"Mismatched points: {len(image_points)} image pts vs {len(world_points)} world pts.")

        src = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        dst = np.array(world_points, dtype=np.float32).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            raise RuntimeError("Failed to compute homography matrix from provided point sets.")

        # Compute mean reprojection error in world units (meters)
        projected = cv2.perspectiveTransform(src, H)
        errors = np.linalg.norm(projected - dst, axis=2)
        mean_error = float(np.mean(errors))

        return cls(homography_matrix=H, camera_id=camera_id, reprojection_error=mean_error)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> HomographyCalibrator:
        """Loads calibration from YAML configuration file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        H_data = data.get("homography_matrix")
        cam_id = data.get("camera_id", "cam_01")
        err = data.get("reprojection_error")

        if H_data is not None:
            H = np.array(H_data, dtype=np.float64)
            return cls(homography_matrix=H, camera_id=cam_id, reprojection_error=err)

        # Alternatively, compute from points if present in yaml
        pts = data.get("points", [])
        if len(pts) >= 4:
            img_pts = [tuple(p["image_coords"]) for p in pts if "image_coords" in p]
            wld_pts = [tuple(p["world_coords"]) for p in pts if "world_coords" in p]
            if len(img_pts) >= 4 and len(img_pts) == len(wld_pts):
                return cls.from_points(img_pts, wld_pts, camera_id=cam_id)

        logger.warning(f"File {path} does not contain valid homography or points. Returning uncalibrated.")
        return cls(camera_id=cam_id)

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Saves active calibration matrix to YAML."""
        out_path = Path(yaml_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "camera_id": self.camera_id,
            "units": self.units,
            "validation_status": "calibrated" if self.is_calibrated else "uncalibrated",
            "reprojection_error": float(self.reprojection_error) if self.reprojection_error is not None else None,
            "homography_matrix": self.H.tolist() if self.H is not None else None
        }

        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(payload, f, default_flow_style=False)
        logger.info(f"Saved calibration to {out_path}")

    def image_to_world(self, points: Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray]) -> Optional[np.ndarray]:
        """Maps image pixel coordinates (u, v) -> world road plane (X, Y) in meters.

        Returns None if calibrator is uncalibrated.
        """
        if not self.is_calibrated:
            return None

        pts = np.asarray(points, dtype=np.float32)
        single_point = (pts.ndim == 1)
        if single_point:
            pts = pts.reshape(1, 1, 2)
        elif pts.ndim == 2:
            pts = pts.reshape(-1, 1, 2)

        world_pts = cv2.perspectiveTransform(pts, self.H)
        world_pts = world_pts.reshape(-1, 2)

        return world_pts[0] if single_point else world_pts

    def world_to_image(self, points: Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray]) -> Optional[np.ndarray]:
        """Maps world road coordinates (X, Y) in meters -> image pixels (u, v).

        Returns None if calibrator is uncalibrated.
        """
        if not self.is_calibrated:
            return None

        pts = np.asarray(points, dtype=np.float32)
        single_point = (pts.ndim == 1)
        if single_point:
            pts = pts.reshape(1, 1, 2)
        elif pts.ndim == 2:
            pts = pts.reshape(-1, 1, 2)

        img_pts = cv2.perspectiveTransform(pts, self.H_inv)
        img_pts = img_pts.reshape(-1, 2)

        return img_pts[0] if single_point else img_pts

    def get_ground_contact_point(self, bbox: List[float]) -> Tuple[float, float]:
        """Calculates bottom-center image coordinate of bounding box as road contact point."""
        x1, _, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        return (float(cx), float(y2))

    def estimate_speed_kmh(
        self,
        trail_pixels: List[Tuple[float, float]],
        fps: float = 12.5,
        window: int = 5
    ) -> Optional[float]:
        """Calculates speed in km/h if calibrated. Returns None if uncalibrated."""
        if not self.is_calibrated or len(trail_pixels) < 2 or fps <= 0:
            return None

        # Take the most recent window of observations
        sub_trail = trail_pixels[-window:]
        world_pts = self.image_to_world(sub_trail)
        if world_pts is None or len(world_pts) < 2:
            return None

        # Compute displacement between first and last in window
        dist_m = float(np.linalg.norm(world_pts[-1] - world_pts[0]))
        elapsed_sec = (len(sub_trail) - 1) / fps
        if elapsed_sec <= 0:
            return None

        speed_m_s = dist_m / elapsed_sec
        speed_km_h = speed_m_s * 3.6

        # Plausibility clamp (0 to 180 km/h)
        return float(np.clip(speed_km_h, 0.0, 180.0))
