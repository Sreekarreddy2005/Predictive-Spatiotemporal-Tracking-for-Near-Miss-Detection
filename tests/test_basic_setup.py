"""Unit tests for Phase 0 setup, configuration validity, and dataset manifest."""

import json
from pathlib import Path
import yaml


def test_manifest_schema():
    manifest_path = Path("data/dataset_manifest.json")
    assert manifest_path.exists(), "Manifest file data/dataset_manifest.json must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "manifest_version" in data
    assert "datasets" in data
    assert isinstance(data["datasets"], list)
    assert len(data["datasets"]) >= 3

    for ds in data["datasets"]:
        assert "id" in ds
        assert "name" in ds
        assert "local_path" in ds
        assert "status" in ds
        assert "splits" in ds


def test_default_pipeline_config():
    config_path = Path("configs/default_pipeline.yaml")
    assert config_path.exists(), "configs/default_pipeline.yaml must exist"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "pipeline" in cfg
    assert "detection" in cfg
    assert "tracking" in cfg
    assert "calibration" in cfg
    assert "prediction" in cfg
    assert "metrics" in cfg
    assert "novelty_modules" in cfg

    # Verify COCO classes include cars, pedestrians, bikes
    target_classes = cfg["detection"]["target_classes"]
    assert 0 in target_classes  # pedestrian
    assert 2 in target_classes  # car


def test_calibration_template_config():
    calib_path = Path("configs/calibration_template.yaml")
    assert calib_path.exists(), "configs/calibration_template.yaml must exist"

    with open(calib_path, "r", encoding="utf-8") as f:
        calib = yaml.safe_load(f)

    assert "camera_id" in calib
    assert "resolution" in calib
    assert "points" in calib
    assert len(calib["points"]) >= 4, "Must provide at least 4 coplanar points for homography"
