# SafeCity AI: Predictive Spatiotemporal Tracking for Near-Miss Detection

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch MPS](https://img.shields.io/badge/PyTorch-Apple%20Silicon%20MPS-red.svg)](https://pytorch.org/)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Research-green.svg)](LICENSE)
[![Status: Complete](https://img.shields.io/badge/Status-Project%20Complete%20(Phases%200--4)-brightgreen.svg)](docs/project_status.md)
[![Test Suite: Passing](https://img.shields.io/badge/Tests-32%2F32%20Passed-brightgreen.svg)](tests/)

> **Shifting urban traffic safety from reactive accident reporting to proactive, predictive risk prevention through Computer Vision, Multi-Agent Graph Attention, and Probabilistic Collision Forecasting.**

---

## Table of Contents
1. [Executive Summary & Motivation](#1-executive-summary--motivation)
2. [Domain Alignment: Computer Vision Based Autonomous Systems](#2-domain-alignment-computer-vision-based-autonomous-systems)
3. [End-to-End System Architecture](#3-end-to-end-system-architecture)
4. [The 5 Academic Research Novelties & Mathematical Formulations](#4-the-5-academic-research-novelties--mathematical-formulations)
   - [Novelty 1: ST-GAT Multi-Agent Interaction-Aware Forecasting](#novelty-1-st-gat-multi-agent-interaction-aware-trajectory-forecasting)
   - [Novelty 2: Probabilistic Monte Carlo TTC Engine](#novelty-2-probabilistic-monte-carlo-ttc-engine)
   - [Novelty 3: Visual Environmental Context Encoder (AASHTO Physics)](#novelty-3-visual-environmental-context-encoder)
   - [Novelty 4: Multi-Metric Adaptive Risk Fusion Network](#novelty-4-multi-metric-adaptive-risk-fusion-network)
   - [Novelty 5: Spatiotemporal Danger-Zone Density Analytics](#novelty-5-spatiotemporal-danger-zone-density-analytics)
5. [Empirical Benchmarks & Experimental Results](#5-empirical-benchmarks--experimental-results)
   - [Trajectory Forecasting Benchmark (ADE / FDE)](#a-trajectory-forecasting-benchmark-novelty-1-vs-baselines)
   - [Near-Miss Conflict Benchmark (ROC-AUC / FAR)](#b-near-miss-conflict-prediction-benchmark-novelty-2-vs-literature)
   - [System Latency & Inference Performance](#c-real-time-inference-performance)
6. [Dataset Architecture & Preprocessing](#6-dataset-architecture--preprocessing)
7. [Interactive Streamlit Dashboard](#7-interactive-streamlit-dashboard)
8. [Installation & Execution Guide](#8-installation--execution-guide)
9. [Google Colab 1-Click Training Notebook](#9-google-colab-1-click-training-notebook)
10. [Repository Directory Structure](#10-repository-directory-structure)
11. [Academic References](#11-academic-references)

---

## 1. Executive Summary & Motivation

Traditional municipal traffic safety relies entirely on **reactive accident statistics**: speed cameras, traffic lights, and road redesigns are only deployed *after* serious crashes, property damage, or fatalities have already occurred. According to **Heinrich's Safety Triangle**, for every major fatal collision, there are approximately **30 minor collisions** and over **300 near-miss conflict events**.

**SafeCity AI** transforms traffic management from a reactive post-mortem analysis into a **proactive, predictive safety paradigm**. Operating on fixed intersection CCTV surveillance and roadside unit (RSU) cameras, the system:
1. Detects and tracks heterogeneous road users (cars, trucks, buses, pedestrians, cyclists) in real time at **72.1 FPS**.
2. Calibrates perspective video into metric Bird's-Eye-View coordinates using **Planar Homography**.
3. Forecasts future multi-agent trajectories 3.0 seconds into the future using **Spatiotemporal Graph Attention (ST-GAT)**.
4. Predicts near-miss conflicts under perception uncertainty using a **Probabilistic Monte Carlo TTC Engine**, cutting false alarms by **79.6%** compared to classical literature (*Hayward 1972*).
5. Adapts stopping safety margins dynamically based on real-time **visual illumination and road friction degradation** (AASHTO guidelines).
6. Fuses all signals into a continuous **0–100 Unified Safety Index** with automated SQLite logging and isolated MP4 evidence clips.

---

## 2. Domain Alignment: Computer Vision Based Autonomous Systems

SafeCity AI directly addresses the core curriculum and technical frontiers of **Computer Vision based Autonomous Systems**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               SafeCity AI: V2X Cooperative Roadside Unit (RSU) Architecture            │
├──────────────────────────┬─────────────────────────────────────────────────────────────┤
│ AV Course Pillar         │ SafeCity AI Implementation                                  │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 1. Deep Perception       │ YOLOv8n real-time detection across vehicles, vulnerable     │
│                          │ road users (VRUs), and micro-mobility at 72.1 FPS.         │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 2. Multi-Object Tracking │ ByteTrack association with persistent track IDs, occlusion  │
│                          │ handling, and ground-contact trajectory buffers.            │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 3. Camera Geometry       │ Planar Homography ($H \in \mathbb{R}^{3 \times 3}$) metric  │
│                          │ transformation (0.18m mean reprojection error).             │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 4. Behavior Prediction   │ Multi-agent Graph Neural Networks (ST-GAT) with social      │
│                          │ multi-head attention (64.8% ADE reduction vs Physics).      │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 5. ADAS & Collision Risk │ Time-to-Collision (TTC), Post-Encroachment Time (PET),      │
│                          │ Deceleration Rate to Avoid Crash (DRAC), CPA distance.      │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 6. Perception Uncertainty│ Monte Carlo particle filter propagation modeling perception │
│                          │ jitter, yielding continuous $P(\text{collision} \le \tau)$. │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 7. Environmental Vision  │ Physics-informed vision computing friction loss ($\mu$)     │
│                          │ and lighting perception delay to expand braking horizons.   │
└──────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 3. End-to-End System Architecture

```text
                     Surveillance Video Frame Stream (CCTV / MP4 Upload)
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │      Frame Preprocessing & Dynamic Masking    │
                    └───────────────────────┬───────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
       ┌─────────────────────────┐                     ┌─────────────────────────┐
       │ YOLOv8 Object Detection │                     │   Novelty 3: Visual     │
       │ (Cars, Pedestrians, etc)│                     │  Environmental Encoder  │
       └────────────┬────────────┘                     │ (Luminance, Visibility, │
                    │                                  │  Friction Factor α_env) │
                    ▼                                  └────────────┬────────────┘
       ┌─────────────────────────┐                                  │
       │ ByteTrack Multi-Object  │                                  │
       │   Tracking (Unique IDs) │                                  │
       └────────────┬────────────┘                                  │
                    │                                               │
                    ▼                                               │
       ┌─────────────────────────┐                                  │
       │ Planar Homography (H)   │                                  │
       │  (Pixels ➔ World Meters)│                                  │
       └────────────┬────────────┘                                  │
                    │                                               │
         ┌──────────┴──────────┐                                    │
         ▼                     ▼                                    │
┌──────────────────┐ ┌──────────────────┐                           │
│ Novelty 1:       │ │ Novelty 2:       │                           │
│ ST-GAT Graph     │ │ Probabilistic    │                           │
│ Attention Model  │ │ Monte Carlo TTC  │                           │
│ (Multi-Agent)    │ │ (250 Particles)  │                           │
└────────┬─────────┘ └────────┬─────────┘                           │
         │                    │                                     │
         └─────────────┐      │      ┌──────────────────────────────┘
                       ▼      ▼      ▼
        ┌────────────────────────────────────────────────────────┐
        │      Novelty 4: Multi-Metric Adaptive Risk Fusion      │
        │       (Unified SafeCity Risk Score: 0 – 100%)          │
        │          CRITICAL  |  WARNING  |  WATCH  |  SAFE       │
        └────────────────────────┬───────────────────────────────┘
                                 │
        ┌────────────────────────┴───────────────────────────────┐
        ▼                                                        ▼
┌───────────────────────────────┐        ┌───────────────────────────────┐
│ SQLite Event Database         │        │ Interactive Streamlit App     │
│ - 3-Frame Hysteresis Filter   │        │ - Live Detection & Replay     │
│ - 30-Frame Cooldown Filter    │        │ - Novelty 1 & 2 Benchmarks    │
│ - Isolated MP4 Evidence Clips │        │ - Novelty 5: Spatial Heatmaps │
└───────────────────────────────┘        └───────────────────────────────┘
```

---

## 4. The 5 Academic Research Novelties & Mathematical Formulations

### Novelty 1: ST-GAT Multi-Agent Interaction-Aware Trajectory Forecasting

Traditional tracking systems extrapolate vehicles independently using single-agent constant velocity or per-agent LSTMs, ignoring how converging vehicles interact, yield, or swerve at intersections.

**Mathematical Formulation**:
For an ego vehicle $i$ and neighbor vehicles $j \in \mathcal{N}_i$ within radius $R = 30\text{m}$:
1. **Temporal Encoding**:
   $$\mathbf{h}_i^{(t)} = \text{LSTM}_{\text{enc}}(\Delta \mathbf{p}_i^{(t)}, \mathbf{h}_i^{(t-1)})$$
2. **Spatial Graph Multi-Head Attention**:
   $$e_{ij} = \text{LeakyReLU}\left( \mathbf{a}^T [\mathbf{W}_q \mathbf{h}_i \,\|\, \mathbf{W}_k \mathbf{s}_j] \right)$$
   $$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}, \quad \mathbf{c}_i = \sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W}_v \mathbf{s}_j$$
   where $\mathbf{s}_j = [\Delta x_{ij}, \Delta y_{ij}, v_{x,j}, v_{y,j}]^T$.
3. **Adaptive Social Gating**:
   $$g_i = \sigma\left(\mathbf{W}_g [\mathbf{h}_i \,\|\, \mathbf{c}_i]\right), \quad \tilde{\mathbf{h}}_i = (1 - g_i) \mathbf{h}_i + g_i \mathbf{c}_i$$
4. **Autoregressive Decoding**:
   $$\hat{\Delta \mathbf{p}}_i^{(t+1)} = \mathbf{W}_{\text{out}} \text{LSTM}_{\text{dec}}(\hat{\Delta \mathbf{p}}_i^{(t)}, \tilde{\mathbf{h}}_i)$$

*Results*: Reduces 3.0s Average Displacement Error by **64.8% vs Physics** and **37.6% vs LSTM**.

---

### Novelty 2: Probabilistic Monte Carlo TTC Engine

Standard deterministic Time-to-Collision (*Hayward 1972*) assumes zero acceleration and exact straight-line velocity vectors:
$$\text{TTC}_{\text{det}} = \frac{\|\mathbf{p}_1 - \mathbf{p}_2\| - R_{\text{crit}}}{\|\mathbf{v}_1 - \mathbf{v}_2\|_{\text{closing}}}$$
Under real-world camera tracking jitter ($\sigma_{\text{pos}} \approx 0.20\text{m}$, $\sigma_{\text{vel}} \approx 0.45\text{m/s}$), slight micro-vector fluctuations cause parallel vehicles to intersect artificially, producing a catastrophic **66.8% false alarm rate**.

**Mathematical Formulation**:
Novelty 2 models tracking uncertainty as Gaussian stochastic processes propagated over 250 Monte Carlo particles:
$$\mathbf{p}_i^{(k)}(t) \sim \mathcal{N}\left(\mathbf{p}_i + \mathbf{v}_i t + \frac{1}{2} \mathbf{a}_i t^2, \; \boldsymbol{\Sigma}_{\text{pos}} + t^2 \boldsymbol{\Sigma}_{\text{vel}}\right)$$

The continuous collision probability over horizon $\tau \in [1.0\text{s}, 3.0\text{s}]$ is computed as:
$$P_{\text{coll}}(\tau) = \frac{1}{N} \sum_{k=1}^{N} \mathbb{I}\left( \min_{t \in [0, \tau]} \|\mathbf{p}_1^{(k)}(t) - \mathbf{p}_2^{(k)}(t)\|_2 \le R_{\text{crit}} \right)$$

*Results*: Increases ROC-AUC from **0.9543 to 0.9823**, provides **+0.14s earlier warning lead time**, and achieves a **79.6% reduction in false alarms**.

---

### Novelty 3: Visual Environmental Context Encoder

Static ADAS thresholds ignore environmental lighting and wet road conditions. Under **AASHTO (American Association of State Highway and Transportation Officials)** and **ISO 15623** vehicle standards, required stopping sight distance is:
$$d_{\text{stopping}} = v \cdot t_{\text{react}} + \frac{v^2}{2 g (\mu \pm G)}$$

When driving in low illumination or rain:
1. Perception-reaction time increases: $t_{\text{react}}' = t_{\text{react}} \cdot (1 + \delta_{\text{light}} + \delta_{\text{vis}})$
2. Road tire-pavement friction drops: $\mu' = \mu_{\text{dry}} \cdot (1 - \delta_{\text{friction}})$

**Dynamic Environmental Expansion Factor**:
$$\alpha_{\text{env}} = \left(\frac{t_{\text{react}}'}{t_{\text{react}}}\right) \cdot \left(\frac{\mu_{\text{dry}}}{\mu'}\right) \in [1.0, 2.2]$$
$$\text{TTC}_{\text{crit}}(\text{env}) = \text{TTC}_{\text{crit}}^{\text{nominal}} \times \alpha_{\text{env}}$$

The encoder analyzes scene illumination ($L \in [0, 255]$ in LAB space), Laplacian contrast variance ($\text{Var}(\nabla^2 I)$), and surface specularity directly from surveillance video, automatically expanding critical stopping margins from **2.0s to up to 2.8s** during hazardous dusk/night/wet conditions.

---

### Novelty 4: Multi-Metric Adaptive Risk Fusion Network

Existing systems rely on a single brittle metric (e.g. only TTC or only PET). Novelty 4 maps a 7-dimensional normalized urgency feature vector:
$$\mathbf{x} = \left[ f_{\text{TTC}}(\alpha_{\text{env}}), \, f_{\text{PET}}, \, f_{\text{DRAC}}, \, P_{\text{coll}}(1\text{s}), \, P_{\text{coll}}(2\text{s}), \, P_{\text{coll}}(3\text{s}), \, f_{\text{prox}} \right]^T$$

through a calibrated neural fusion mapping to a continuous **0–100 Unified Safety Index**:
$$\text{Risk} = \frac{100}{1 + \exp\left(-k (\mathbf{w}^T \mathbf{x} + \beta_{\text{env}} - x_0)\right)}$$

*Calibrated Severity Tiers*:
* **`CRITICAL` Risk ($\ge 72\%$)**: Imminent conflict; immediate AEB / evasive intervention required.
* **`WARNING` Risk ($48\% - 71\%$)**: High near-miss risk; evasive steering/braking recommended.
* **`WATCH` Risk ($28\% - 47\%$)**: Elevated spatial convergence; dynamic monitoring active.
* **`SAFE` Risk ($< 28\%$)**: Nominal, unconstrained traffic flow.

---

### Novelty 5: Spatiotemporal Danger-Zone Density Analytics

Aggregates all near-miss events and continuous risk fields across time into 2D Kernel Density Estimations (KDE):
$$\hat{f}(x, y) = \frac{1}{n h^2 2\pi} \sum_{i=1}^{n} \exp\left(-\frac{(x - x_i)^2 + (y - y_i)^2}{2 h^2}\right) \cdot \text{Risk}_i$$
Identifies persistent geometric design flaws in intersection crosswalks, turn pockets, and merging lanes directly within the Streamlit dashboard.

---

## 5. Empirical Benchmarks & Experimental Results

### A. Trajectory Forecasting Benchmark (Novelty 1 vs. Baselines)
Evaluated on **298 held-out test sequences** ($T_{\text{obs}} = 2.0\text{s}$, $T_{\text{pred}} = 3.0\text{s}$, 5 Hz downsampled):

| Prediction Horizon | Constant Velocity (Physics) | Seq2Seq LSTM (Deep Learning) | Proposed ST-GAT (Novelty 1) | Error Drop vs. Physics | Error Drop vs. LSTM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ADE @ 1.0s (Short-term)** | `1.87 m` | `1.26 m` | **`1.08 m`** | **42.2%** | **14.3%** |
| **ADE @ 2.0s (Mid-term)** | `3.37 m` | `1.96 m` | **`1.41 m`** | **58.2%** | **28.1%** |
| **ADE @ 3.0s (Long-term)** | `4.94 m` | `2.79 m` | **`1.74 m`** | **64.8%** | **37.6%** |
| **FDE @ 3.0s (Endpoint Error)**| `9.18 m` | `4.74 m` | **`3.07 m`** | **66.5%** | **35.2%** |

*Artifact: [`outputs/prediction_benchmark.json`](outputs/prediction_benchmark.json)*

---

### B. Near-Miss Conflict Prediction Benchmark (Novelty 2 vs. Literature)
Evaluated on **500 diverse encounter scenarios** under realistic CCTV sensor jitter:

| Performance Metric | Deterministic TTC (*Hayward 1972*) | Probabilistic TTC (**Novelty 2**) | Improvement / Significance |
| :--- | :---: | :---: | :--- |
| **ROC-AUC Score** | `0.9543` | **`0.9823`** | **+0.0280** (Superior Discriminability) |
| **False Alarm Rate @ 95% Recall** | `66.8%` | **`13.6%`** | **79.6% Reduction in False Alarms** |
| **True Positive Conflict Recall** | `93.2%` | **`99.6%`** | **+6.4%** (Virtually Zero Missed Conflicts) |
| **Warning Lead Time before CPA** | `1.41 s` | **`1.54 s`** | **+0.14 s Earlier Alert** before impact |

*Artifact: [`outputs/probabilistic_ttc_benchmark.json`](outputs/probabilistic_ttc_benchmark.json)*

---

### C. Real-Time Inference Performance

| Component | Hardware Target | Execution Time / Rate | Academic Standard |
| :--- | :--- | :--- | :--- |
| **YOLOv8n Object Detector** | Apple Silicon M5 (MPS) | `13.9 ms / frame` (**72.1 FPS**) | Real-time (> 25 FPS) |
| **ByteTrack MOT Tracking** | CPU Vectorized | `1.8 ms / frame` | Sub-millisecond tracking |
| **ST-GAT Trajectory Inference** | PyTorch MPS | `3.2 ms / pair` | Real-time multi-agent forecasting |
| **Monte Carlo 250-Particle TTC** | NumPy Vectorized | `0.5 ms / pair` | Sub-millisecond risk sampling |
| **Full Video Pipeline** | End-to-End System | **~48.5 FPS** (including video render) | 2x Faster than real-time CCTV |

---

## 6. Dataset Architecture & Preprocessing

SafeCity AI uses the **inD / LevelXdata** high-precision trajectory dataset recorded at urban intersections:
* **Raw Track Corpus**: `data/raw/trajectories/01_tracks.csv` (348,751 rows, 38.6 MB).
* **Sampling Rate**: Downsampled from 25 Hz to **5 Hz** ($\Delta t = 0.2\text{s}$) to match real-time CCTV prediction intervals.
* **Temporal Slicing**:
  * **Observation Window ($T_{\text{obs}}$)**: 10 frames ($2.0\text{s}$) of historical ground contact points.
  * **Prediction Horizon ($T_{\text{pred}}$)**: 15 frames ($3.0\text{s}$) of future trajectory forecast.
* **Spatial Normalization**: Displacements computed relative to ego vehicle's final observation coordinate.
* **Multi-Agent Interaction Graph**: Adjacency constructed using $R = 30.0\text{m}$ Euclidean threshold, preserving relative displacement and velocity vectors $[\Delta x, \Delta y, v_x, v_y]$.

---

## 7. Interactive Streamlit Dashboard

The web dashboard (`app/main.py`) provides an interactive interface for traffic engineers and evaluators:

1. **Tab 1: Live Pipeline & Replay**
   - Side-by-side surveillance video player: original CCTV footage vs. AI-annotated stream with bounding boxes, velocity badges, historical trails, and dotted 3.0s future trajectories.
2. **Tab 2: Near-Miss Database**
   - Searchable table of all conflict encounters saved in `outputs/events.db`.
   - Single-click evidence replay: extracts and plays the exact isolated MP4 clip of any selected incident.
3. **Tab 3: Research Benchmarks**
   - **Subtab 1**: Side-by-side ADE/FDE metrics for Constant Velocity, LSTM, and ST-GAT.
   - **Subtab 2**: Quantitative comparison table for Novelty 2 vs *Hayward (1972)*.
   - **Interactive Monte Carlo Simulator**: Sliders to test closing speeds, particle counts, and encounter geometries with real-time probability curves.
4. **Tab 4: Spatiotemporal Heatmaps**
   - Altair 2D interactive spatial concentration maps showing conflict hotspots and risk clustering across the intersection ground plane.
5. **Tab 5: Camera Calibration**
   - Planar homography matrix ($H$) viewer and reprojection error validation.

---

## 8. Installation & Execution Guide

### Prerequisites
* macOS (Apple Silicon M1/M2/M3/M4/M5 recommended) or Linux with Python 3.11.
* Recommended: Virtual environment (`.venv`).

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/Predictive-SpatioTemporal-Tracking.git
cd "Predictive-SpatioTemporal-Tracking"

# Create and activate Python 3.11 virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run the Automated Test Suite (32 Tests)
```bash
pytest -v
```
*Expected: `32 passed in ~3.2 seconds`.*

### Step 3: Launch the Streamlit Dashboard
```bash
streamlit run app/main.py
```
Open your browser at `http://localhost:8501`.

### Step 4: Reproduce the Trajectory Benchmark (Novelty 1)
```bash
python scripts/train_eval_trajectory.py
```
*Evaluates Constant Velocity, Seq2Seq LSTM, and ST-GAT, saving results to `outputs/prediction_benchmark.json`.*

### Step 5: Reproduce the Probabilistic TTC Benchmark (Novelty 2)
```bash
python scripts/benchmark_probabilistic_ttc.py --scenarios 500
```
*Evaluates 500 conflict encounters under realistic tracking noise, saving results to `outputs/probabilistic_ttc_benchmark.json`.*

---

## 9. Google Colab 1-Click Training Notebook

For university viva presentations and cloud GPU verification, a standalone Google Colab notebook is included:
* **Notebook Path**: [`notebooks/SafeCity_AI_Neural_Training.ipynb`](notebooks/SafeCity_AI_Neural_Training.ipynb)

**How to Run**:
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select `notebooks/SafeCity_AI_Neural_Training.ipynb`.
3. Set Hardware Accelerator to **T4 GPU** (`Runtime` -> `Change runtime type`).
4. Click `Runtime` -> `Run all`.
5. The notebook automatically downloads sample trajectories, trains both LSTM and ST-GAT, plots convergence curves, computes ADE/FDE metrics, runs the Monte Carlo TTC comparison, and exports trained `.pt` checkpoints.

---

## 10. Repository Directory Structure

```text
├── configs/
│   ├── default_pipeline.yaml      # Master configuration for detector, tracker, thresholds
│   └── camera_calibration.yaml    # Planar homography matrix and ground markers
├── data/
│   ├── raw/trajectories/          # inD benchmark CSV (01_tracks.csv, 348k rows)
│   ├── samples/cctv/              # CCTV sample video (sample_traffic.mp4)
│   └── dataset_manifest.json      # Dataset registry and validation metadata
├── models/
│   ├── lstm_baseline.pt           # Trained PyTorch Seq2Seq LSTM weights
│   └── interaction_stgat.pt       # Trained PyTorch ST-GAT Novelty 1 weights
├── notebooks/
│   └── SafeCity_AI_Neural_Training.ipynb # Standalone 1-click Google Colab notebook
├── outputs/
│   ├── annotated_traffic.mp4      # Full video with tracks, trails & predicted trajectories
│   ├── events.db                  # SQLite database of all logged near-miss encounters
│   ├── evidence/                  # Isolated MP4 evidence snippets per incident
│   ├── prediction_benchmark.json  # 3-way trajectory ADE/FDE evaluation results
│   └── probabilistic_ttc_benchmark.json # 500-scenario Novelty 2 evaluation results
├── src/
│   └── safecity/
│       ├── calibration/           # Planar homography estimation & speed calculation
│       ├── detection/             # YOLOv8 inference wrapper & COCO class filtering
│       ├── metrics/
│       │   ├── conflict_engine.py       # Deterministic TTC, PET, DRAC, CPA
│       │   ├── probabilistic_ttc.py     # Novelty 2: Monte Carlo Gaussian particle engine
│       │   ├── environmental_context.py # Novelty 3: AASHTO friction & illumination encoder
│       │   ├── risk_fusion.py           # Novelty 4: Adaptive 0–100 Unified Risk Fusion
│       │   └── trajectory_metrics.py    # Vectorized ADE and FDE computation
│       ├── pipeline/              # Master video orchestration & HUD rendering
│       ├── prediction/
│       │   ├── constant_velocity.py     # Analytical physics baseline
│       │   ├── dataset.py               # 5 Hz trajectory windowing & neighbor extraction
│       │   ├── lstm_predictor.py        # Seq2Seq LSTM baseline model
│       │   └── interaction_predictor.py # Novelty 1: ST-GAT multi-agent model
│       ├── storage/               # SQLite database, 3-frame hysteresis & evidence extraction
│       └── tracking/              # ByteTrack integration with persistent IDs
├── tests/                         # 32 unit and integration tests (100% passing)
├── app/
│   └── main.py                    # Streamlit interactive web dashboard
├── scripts/
│   ├── calibrate_camera.py        # Interactive 4-point homography calibration tool
│   ├── train_eval_trajectory.py   # Training & evaluation script for Novelty 1
│   ├── benchmark_probabilistic_ttc.py # 500-scenario benchmark script for Novelty 2
│   └── create_colab_notebook.py   # Colab notebook generator script
├── pyproject.toml                 # Package specification and pytest configuration
└── README.md                      # This documentation
```

---

## 11. Academic References

1. **Hayward, J. C. (1972)**. *Near-miss determination through use of a scale of danger*. Highway Research Record, 384, 24–34. *(Foundational baseline for deterministic Time-to-Collision)*.
2. **Minderhoud, M. M., & Bovy, P. H. (2001)**. *Extended time-to-collision measures for road traffic safety assessment*. Accident Analysis & Prevention, 33(1), 89–97.
3. **Alahi, A., et al. (2016)**. *Social LSTM: Human trajectory prediction in crowded spaces*. CVPR 2016.
4. **Huang, Y., et al. (2019)**. *STGAT: Modeling spatial-temporal interactions for human trajectory prediction*. ICCV 2019.
5. **AASHTO (2018)**. *A Policy on Geometric Design of Highways and Streets (The Green Book)*. American Association of State Highway and Transportation Officials. *(Physics-based stopping sight distance and friction coefficient standards)*.
6. **Bock, J., et al. (2020)**. *The inD Dataset: A Drone Dataset of Naturalistic Road User Trajectories at German Intersections*. IEEE IV 2020.
7. **Zhang, Y., et al. (2022)**. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV 2022.
