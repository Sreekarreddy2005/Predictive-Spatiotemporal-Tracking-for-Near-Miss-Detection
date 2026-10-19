# System & Hardware Requirements: SafeCity AI

## 1. Local Environment Findings (Inspected System)

* **Operating System**: macOS (Darwin Kernel Version 27.0.0 arm64, macOS 27.0.1)
* **Processor Architecture**: Apple Silicon M5 (`arm64`)
* **Unified Memory (RAM)**: 16.0 GB Unified Memory (`hw.memsize: 17179869184 bytes`)
* **Available Disk Storage**: 621 GiB available on `/System/Volumes/Data` (31% capacity used)
* **Acceleration Support**:
  * Apple Metal Performance Shaders (`mps` device in PyTorch) for native hardware-accelerated tensor operations and neural inference.
  * *Note*: Native NVIDIA CUDA is not available on macOS hardware.
* **Base Installed Utilities**:
  * `Python 3.11.16` at `/Users/sreekarreddypindi/.local/bin/python3.11`
  * `uv` fast Python package installer at `/Users/sreekarreddypindi/.local/bin/uv`
  * `ffmpeg` available at `/opt/homebrew/bin/ffmpeg`
  * `git` available at `/opt/homebrew/bin/git`

---

## 2. Compute Strategy: Local Prototyping vs. Remote GPU Training

Because SafeCity AI combines real-time video inference (YOLOv8 + ByteTrack) with deep trajectory predictors (LSTM, Social-LSTM, or ST-GNN), we divide computational tasks practically:

| Task | Suitable on Local Apple Silicon M5 (16 GB Unified Memory)? | Recommended Execution Workflow |
| :--- | :--- | :--- |
| **Pipeline Development & Integration** | **Yes** (Fast, highly responsive) | Local `.venv` using `mps` or CPU |
| **YOLOv8 Pretrained Baseline Inference** | **Yes** (30–60 FPS with YOLOv8n / YOLOv8s on `mps`) | Local `.venv` |
| **ByteTrack & Motion Estimation** | **Yes** (Lightweight CPU / NumPy / SciPy operations) | Local `.venv` |
| **Constant-Velocity Prediction & Conflict Analysis** | **Yes** (Deterministic analytical math) | Local `.venv` |
| **Streamlit Interactive Dashboard & Replay** | **Yes** (Runs locally on `localhost:8501`) | Local `.venv` |
| **Lightweight Trajectory Model Training (Per-Agent LSTM)** | **Yes** (Small sequences, reasonable epochs) | Local `.venv` (`device='mps'`) |
| **Heavy Interaction-Aware Model Training (ST-GNN / Social-LSTM on full datasets)** | **Borderline** (Viable for small splits; large multi-agent graph batches benefit from high VRAM CUDA) | Remote GPU (Google Colab T4/A100 or AWS EC2) |
| **Full YOLOv8 Domain Fine-Tuning (100k+ frames)** | **Remote Recommended** | Google Colab / Kaggle / Cloud GPU |

### Remote Training Protocol (When Needed)
If heavy training is initiated:
1. All dataset preparation scripts produce standardized, preprocessed `.npz` or `.pt` tensors locally.
2. A single standalone notebook (`notebooks/train_colab.ipynb`) will be provided containing exact training commands.
3. Only lightweight model checkpoints (`*.pt`, typically 5 MB – 50 MB) need to be downloaded back to `models/` locally.

---

## 3. Pinned Dependency Specification

Dependencies are isolated in a local Python 3.11 virtual environment (`.venv`):

```text
torch>=2.2.0
torchvision>=0.17.0
ultralytics>=8.1.0
opencv-python>=4.8.0
numpy>=1.26.0
scipy>=1.11.0
pandas>=2.1.0
pyyaml>=6.0.1
streamlit>=1.30.0
plotly>=5.18.0
pytest>=7.4.0
tqdm>=4.66.0
```

---

## 4. Local Environment Setup Commands

```bash
# 1. Activate the Python 3.11 virtual environment
source .venv/bin/activate

# 2. Verify PyTorch and Apple Silicon MPS acceleration
python -c "import torch; print('PyTorch version:', torch.__version__); print('MPS Available:', torch.backends.mps.is_available())"

# 3. Verify Ultralytics & OpenCV
python -c "import cv2, ultralytics; print('OpenCV:', cv2.__version__); print('Ultralytics:', ultralytics.__version__)"
```
