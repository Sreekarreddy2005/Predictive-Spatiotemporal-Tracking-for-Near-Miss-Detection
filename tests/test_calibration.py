"""Unit tests for camera calibration, homography estimation, and reprojection."""

import numpy as np
import pytest
from safecity.calibration.homography import HomographyCalibrator


def test_homography_forward_and_inverse_roundtrip():
    # Synthetic ground-plane square in meters
    world_pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    # Perspective projection onto image sensor pixels
    image_pts = [(100.0, 500.0), (800.0, 500.0), (650.0, 200.0), (250.0, 200.0)]

    calibrator = HomographyCalibrator.from_points(
        image_points=image_pts,
        world_points=world_pts,
        camera_id="test_cam"
    )

    assert calibrator.is_calibrated
    assert calibrator.reprojection_error is not None
    assert calibrator.reprojection_error < 1e-4

    # Test round-trip for each reference point
    for img_p, wld_p in zip(image_pts, world_pts):
        proj_world = calibrator.image_to_world(img_p)
        assert proj_world is not None
        np.testing.assert_allclose(proj_world, wld_p, atol=1e-3)

        reproj_img = calibrator.world_to_image(proj_world)
        assert reproj_img is not None
        np.testing.assert_allclose(reproj_img, img_p, atol=1e-3)


def test_uncalibrated_graceful_fallback():
    calibrator = HomographyCalibrator(homography_matrix=None)
    assert not calibrator.is_calibrated

    # Should safely return None for world coordinates and km/h
    assert calibrator.image_to_world((100.0, 200.0)) is None
    assert calibrator.world_to_image((5.0, 10.0)) is None
    assert calibrator.estimate_speed_kmh([(100.0, 100.0), (110.0, 100.0)]) is None


def test_speed_estimation_meters_to_kmh():
    # 1 pixel = 0.1 meters
    H = np.diag([0.1, 0.1, 1.0])
    calibrator = HomographyCalibrator(homography_matrix=H, camera_id="linear_cam")

    # Object moves 100 pixels in 1 second (10 meters/second = 36 km/h)
    # At 10 FPS, 10 frames = 1 second
    trail = [(0.0, 0.0)] + [(10.0 * i, 0.0) for i in range(1, 11)]
    speed_kmh = calibrator.estimate_speed_kmh(trail, fps=10.0, window=10)

    assert speed_kmh is not None
    np.testing.assert_allclose(speed_kmh, 36.0, atol=0.5)
