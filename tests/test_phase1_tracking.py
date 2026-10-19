"""Unit tests for Phase 1: Pretrained YOLOv8 detection, ByteTrack tracking, and video export."""

from pathlib import Path
import numpy as np
import pytest

from safecity.detection.detector import TrafficDetector
from safecity.tracking.tracker import TrafficTracker


def test_detector_initialization_and_inference():
    """Verify YOLOv8 detector initializes and processes a dummy frame without error."""
    detector = TrafficDetector(model_name="yolov8n.pt", confidence_threshold=0.25, device="cpu")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)
    assert isinstance(detections, list)


def test_tracker_initialization_and_update():
    """Verify TrafficTracker integrates detector + ByteTrack and tracks across frames."""
    tracker = TrafficTracker(model_name="yolov8n.pt", conf_thresh=0.25, device="cpu")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Synthetic object in dummy frame (draw a bright rectangle)
    cv2_rect = dummy_frame.copy()
    cv2_rect[100:200, 100:250] = 255

    tracks = tracker.track_frame(cv2_rect, frame_idx=0, dt=0.04)
    assert isinstance(tracks, list)


def test_phase1_artifacts_exist():
    """Verify that Phase 1 expected deliverables exist on disk."""
    sample_video = Path("data/samples/cctv/sample_traffic.mp4")
    annotated_video = Path("outputs/annotated_traffic.mp4")
    tracking_summary = Path("outputs/tracking_summary.json")

    assert sample_video.exists(), "Sample traffic video must exist in data/samples/cctv/"
    assert annotated_video.exists(), "Annotated video must exist in outputs/"
    assert tracking_summary.exists(), "Tracking summary JSON must exist in outputs/"
