# Dataset Acquisition Plan & Setup: SafeCity AI

## 1. Separation of Data Requirements Across Modules

A critical methodological principle in SafeCity AI is recognizing that **no single dataset satisfies all six operational requirements**. Attempting to force one dataset (e.g. an aerial trajectory dataset or a highway tracking benchmark) into all tasks produces invalid evaluation. We explicitly decouple data requirements:

| Module Need | Functional Requirement | Critical Data Characteristics | Candidate Sources |
| :--- | :--- | :--- | :--- |
| **A. Object Detection** | Bounding box localization of road users | High-resolution surveillance camera perspectives with realistic scale, occlusion, and road user variety | COCO (pretrained baseline), UA-DETRAC, VisDrone-DET |
| **B. Multi-Object Tracking (MOT)** | Continuous tracking across frames with persistent IDs | Uncut frame sequences (15–30 FPS), persistent object IDs across occlusions, camera fixed at surveillance angle | UA-DETRAC MOT subset, VisDrone-MOT |
| **C. Trajectory Prediction** | Forecasting future spatial coordinates $(X, Y)$ over $T_{pred}$ given history $T_{obs}$ | Continuous metric $(X,Y)$ positions, interactions between converging paths, intersection maneuvers | inD (intersection drone), NGSIM (highway/arterial), custom CCTV tracks |
| **D. Independently Labelled Near-Miss Evaluation** | Evaluating precision, recall, and lead-time of safety alerts | Explicit ground-truth collision or near-miss event annotations (timestamps, agent IDs, severity, evasive actions) | NVIDIA AI City Challenge Track 2 / SynWTS / WTS, or custom human-annotated CCTV clips |
| **E. Lighting & Weather Classification** | Visual context encoder testing (day, night, rain, dusk) | Surveillance footage under varied illuminations and meteorological conditions | UA-DETRAC (annotated with sunny, rainy, night), BDD100k, custom video samples |
| **F. End-to-End CCTV Demonstrations** | Realistic urban intersection replay in Streamlit dashboard | Continuous high-quality urban intersection video with known ground plane or identifiable road landmarks | User-supplied local CCTV, open municipal traffic streams, UA-DETRAC full sequences |

> [!WARNING]
> **Important Scientific Caveat**: Trajectory datasets (such as inD or NGSIM) and MOT datasets (such as UA-DETRAC) record **routine, non-collision traffic**. Their trajectory overlaps or proximity points represent normal navigation, not ground-truth accidents. They must **never** be reported as ground-truth near-miss events without verified event-level annotations.

---

## 2. Comprehensive Candidate Dataset Comparison Matrix

| Dataset | Priority | Intended Module | Official Landing Page & Verified Source Link | Required Files & Recommended Splits | Available Classes & Annotation Types | Content Type | Coordinate System & Units | Access Requirements | Known Size | Proposed Local Folder | Main Limitations | Fallback if Access Fails |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **UA-DETRAC** | **Needed Now** (Minimal sample first) | Detection, Tracking (MOT), Lighting/Weather Context | [Author Homepage](https://sites.google.com/view/daweidu/projects/ua-detrac) <br> [Google Drive Files](https://drive.google.com/file/d/1fQC6CEWOeL9pnJ4ZdD7U2gromcnWDRgm/view?usp=sharing) | Train / Test image sequences; XML annotations; start with 1 sequence (e.g. `MVI_20011`) | Car, Bus, Van, Others; 2D Bounding Boxes + Illumination tags | Sequential JPEG frames (140k frames total in full set) | Image space (pixels $[u, v]$) | Open via Google Drive links (no approval required) | Full set ~15 GB compressed; single test sequence ~150 MB | `data/raw/UA-DETRAC/` | Fixed angled view from overpass; no ground-truth metric calibration included; no pedestrian annotations | Use open urban CCTV video clip with pretrained YOLOv8 |
| **inD (Intersection Drone)** | **Needed Later** (Phase 3 Trajectory Model) | Trajectory Prediction (Social-LSTM / ST-GNN) | [levelXdata inD Portal](https://levelxdata.com/ind-dataset/) | CSV trajectory tables; start with 1 recording from Location 1 (`00_tracks.csv`, `00_tracksMeta.csv`) | Car, Truck, Bus, Pedestrian, Bicycle; 25 Hz trajectories, bounding boxes, orientations | Georeferenced trajectory CSVs, background aerial orthophoto | Metric UTM / local Cartesian (meters, m/s) | Free academic registration form; requires institutional email & agreement | Full dataset ~5–10 GB; single recording CSV ~20 MB | `data/raw/inD/` | Bird's-eye view (drone) rather than CCTV surveillance angle; natural traffic without verified conflict labels | NGSIM Lankershim Blvd or open Argoverse / INTERACTION dataset sample |
| **VisDrone-DET / MOT** | **Optional** (Ablation study) | Object Detection & Tracking under extreme scales | [VisDrone GitHub](https://github.com/VisDrone/VisDrone-Dataset) | `VisDrone2019-DET-val` or `VisDrone2019-MOT-val` | Pedestrian, person, car, van, bus, truck, motor, bicycle, awning-tricycle | Image sequences / video frames | Image space (pixels) | Public via Google Drive / BaiduYun links | Val set: ~70 MB; Train set: ~1.44 GB | `data/raw/VisDrone/` | Drone perspective; very small object pixel areas; distinct camera motion compared to static CCTV | Pretrained YOLOv8 on COCO |
| **NGSIM (Next Gen Simulation)** | **Optional** (Trajectory benchmark) | Trajectory Prediction baseline | [data.gov NGSIM Portal](https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data) | Lankershim Boulevard or I-80 CSV trajectory tables | Vehicle trajectories (cars, trucks, motorcycles) | Trajectory CSV records | Local Cartesian feet / meters (converted) | Completely public open data (direct HTTP download) | Full files ~1–2 GB | `data/raw/NGSIM/` | High noise in raw velocities; mostly highway corridors; pedestrian interactions limited to specific crosswalks | inD or custom synthetic trajectories |
| **AI City Challenge Track 2 (SynWTS / WTS)** | **Needed Later** (Phase 4–5 Evaluation) | Independently Labelled Near-Miss Evaluation | [AI City Challenge](https://www.aicitychallenge.org/ai-city-challenge-dataset-access/) & [SynWTS HF](https://huggingface.co/datasets/mlcglab/synwts) | 2024 Track 2 / SynWTS validation split | Vehicles, pedestrians; safety conflict descriptions, incident video clips | Short surveillance & dashcam video clips + incident annotations | Video frame timestamps + bounding boxes | Open on Hugging Face / evaluation portal with academic email | SynWTS subsets ~2–8 GB | `data/raw/AI_City_Track2/` | Synthetic/simulated scenarios mixed with real clips; focus on video captioning and anomaly detection | Human-annotated custom CCTV conflict clips |
| **highD** | **Optional** | Highway trajectory prediction (not intersections) | [levelXdata highD Portal](https://levelxdata.com/highd-dataset/) | Highway trajectory CSVs | Cars, Trucks | Drone trajectory CSVs | Metric Cartesian (meters) | Academic registration | ~2 GB | `data/raw/highD/` | Highway lane-following only; lacks urban intersection conflict geometry | inD is far superior for urban near-miss conflicts |

---

## 3. Recommended Minimal Initial Dataset Combination

To avoid downloading dozens of gigabytes before validating our pipeline, we recommend a **two-step acquisition strategy**:

### Step 1: Immediate Verification Download (Milestone Phase 0 / Phase 1)
1. **One UA-DETRAC Test Sequence** (or 1 short urban CCTV video clip):
   * Size: ~100–200 MB.
   * Purpose: Validate video loading, YOLOv8 detection, ByteTrack ID persistence, image-space trajectory history, and dashboard replay.
2. **One inD Sample CSV** (or NGSIM sample table):
   * Size: ~15–30 MB.
   * Purpose: Validate trajectory loader, data normalization, train/val split logic, and constant-velocity predictor.

### Step 2: Full Scaled Acquisition (Milestone Phase 3 / Phase 5)
* Complete inD multi-intersection package for interaction-aware model training.
* AI City Challenge Track 2 / SynWTS for ground-truth near-miss benchmark evaluation.

---

## 4. Independent Near-Miss Annotation Protocol (For Custom CCTV Videos)

When working with real surveillance videos without pre-existing conflict labels, we follow a rigorous human annotation protocol:

1. **Conflict Definition**: A near-miss is defined as an encounter between two or more road users where collision was avoided only through sudden evasive action (braking, swerving, evasive pedestrian jump) or where physical trajectories intersected with critical temporal proximity ($TTC < 2.0\text{ s}$ or $PET < 1.5\text{ s}$).
2. **Annotation Schema** (`data/annotations/near_miss_labels.json`):
   ```json
   {
     "video_name": "intersection_demo_01.mp4",
     "events": [
       {
         "event_id": "EVT_001",
         "agent_ids": [12, 18],
         "agent_types": ["car", "pedestrian"],
         "start_frame": 145,
         "alert_frame": 160,
         "closest_approach_frame": 172,
         "end_frame": 195,
         "conflict_type": "vehicle_pedestrian_crossing",
         "evasive_action": "sharp_braking",
         "reviewer_confidence": "high",
         "severity_level": "critical"
       }
     ]
   }
   ```
3. **Double Review**: Borderline interactions are flagged with `reviewer_confidence: "ambiguous"` to prevent noisy test set pollution.
