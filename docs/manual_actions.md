# Immediate Manual Actions & Roadmap Checklist

This document details the exact actions required from you to support SafeCity AI development.
Only **Part 1 (Immediate Phase 0 / Phase 1 Requirements)** is active now.
**Part 2 (Future Milestone Checklist)** is provided for planning and will be activated in later phases.

---

## Part 1: Immediate Manual Actions (Active Now)

### Action 1.1: Supply One Sample Traffic Video (or Download 1 UA-DETRAC Sequence)

* **What you need to obtain or do**:
  Provide a single sample traffic video recorded from an elevated / surveillance perspective (such as a fixed intersection, road crosswalk, or overpass CCTV clip) OR download 1 sequence from UA-DETRAC.
* **Why it is needed**:
  To verify the end-to-end video decoding pipeline, YOLOv8 object detection, ByteTrack tracking ID persistence, coordinate extraction, and Streamlit dashboard video player without downloading tens of gigabytes.
* **Official Link / Source**:
  * Option A (Recommended for custom demo): Any 15–60 second urban traffic video (`.mp4`, `.avi`, or `.mov`) you have access to.
  * Option B (UA-DETRAC): [UA-DETRAC Google Drive](https://drive.google.com/drive/folders/1am_x3ywLJnABq4REtzbi0XTu5csFrp8v?usp=sharing) and [DETRAC-Images](https://drive.google.com/file/d/1fQC6CEWOeL9pnJ4ZdD7U2gromcnWDRgm/view?usp=sharing).
* **Which download, split, or files to choose**:
  A single sequence (e.g. `MVI_20011` or any single `.mp4` video clip, 720p or 1080p).
* **Whether registration or approval is required**:
  None for UA-DETRAC Google Drive or personal/open CCTV video.
* **Download size & extracted size**:
  ~20 MB to ~150 MB (small and fast).
* **Where to place the files**:
  Place the video file into:
  `data/samples/cctv/sample_traffic.mp4`
  (If using a UA-DETRAC sequence folder with JPEGs, place inside `data/raw/UA-DETRAC/sample_sequence/`).
* **Expected Folder Structure**:
  ```text
  data/
  └── samples/
      └── cctv/
          └── sample_traffic.mp4
  ```
* **What you should send me afterward**:
  Simply tell me the file name and path you placed it in (e.g. `data/samples/cctv/sample_traffic.mp4`).
* **How I will verify that it is correct**:
  I will run:
  ```bash
  python scripts/validate_data.py --target data/samples/cctv/sample_traffic.mp4
  ```
  This will automatically report frame count, resolution, FPS, decode validity, and confirm it is ready for the detection/tracking pipeline.

---

### Action 1.2: Apply for Academic Access to the inD Dataset (Preparation for Phase 3)

* **What you need to obtain or do**:
  Submit an academic access request form on the levelXdata portal for the inD dataset.
* **Why it is needed**:
  inD contains high-precision naturalistic vehicle, pedestrian, and cyclist trajectories at German urban intersections (recorded at 25 Hz). This will serve as our core benchmark for training the Per-Agent LSTM and the interaction-aware model (Social-LSTM or ST-GNN).
* **Official Link**:
  [https://levelxdata.com/ind-dataset/](https://levelxdata.com/ind-dataset/)
* **Which download / split**:
  Academic non-commercial license. You will only need **Location 1** (`00_tracks.csv` to `06_tracks.csv`) initially.
* **Whether registration or approval is required**:
  **Yes**. Academic verification is required (use your university/institutional email ID). Approval typically takes 1–3 business days.
* **Download size**:
  A single recording CSV is ~20 MB; the complete bundle is ~5 GB.
* **Where to place the files (when received)**:
  `data/raw/inD/`
* **What you should send me afterward**:
  Confirmation when your access link arrives. (Our pipeline development will proceed using constant-velocity baselines and synthetic/sample trajectories while approval is pending).

---

## Part 2: Future Milestones Checklist (Pending Later Phases)

| Phase | Milestone Action | Description | Dependencies | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 2** | Camera Calibration Reference Points | Supply 4 coplanar ground reference points or known distance markers on the sample video frame | Sample video from Phase 1 | *Pending Phase 1* |
| **Phase 3** | inD Location 1 Trajectory Download | Download `00_tracks.csv` to `06_tracks.csv` and place in `data/raw/inD/` | Academic approval from levelXdata | *Pending Approval* |
| **Phase 4** | Weather / Lighting Categorized Clips | Provide or identify day, night, rain surveillance segments for Visual Context testing | Video sequences | *Pending Phase 3* |
| **Phase 5** | Near-Miss Ground-Truth Annotations | Review flagged conflict intervals and confirm evasive action labels | Event logs from Phase 4 | *Pending Phase 4* |
