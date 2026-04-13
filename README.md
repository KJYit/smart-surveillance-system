# 🛡️ Smart Surveillance System

A real-time, multi-module smart surveillance pipeline built with deep learning. The system detects persons in challenging conditions (low-light, crowded scenes), tracks them across frames, identifies loitering behaviour, classifies violent/threatening actions via skeleton-based analysis, and suppresses false positives — all through an interactive **Streamlit** web dashboard.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Module Details](#module-details)
  - [M2 — Person Detection](#m2--person-detection)
  - [M3 — Person Tracking](#m3--person-tracking)
  - [M4-A — Loitering Detection](#m4-a--loitering-detection)
  - [M4-B — Behaviour Classification](#m4-b--behaviour-classification)
  - [M5 — False Positive Reduction](#m5--false-positive-reduction)
- [Demo Application](#demo-application)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Training Notebooks](#training-notebooks)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

This project is a **Final Year Project (FYP)** that implements an end-to-end smart surveillance system. It processes video feeds (live webcam, uploaded files, or local video files) and runs them through a five-stage pipeline to detect, track, and analyse human behaviour in real time.

### Key Features

- **Low-light person detection** using a YOLOv8 model fine-tuned on the LLVIP infrared-visible dataset.
- **Multi-object tracking** with DeepSORT for persistent person identity across frames.
- **Rule-based loitering detection** using virtual zone dwell-time analysis.
- **Skeleton-based behaviour classification** via a Transformer ensemble trained on COCO-17 keypoint features extracted from UCF-Crime clips.
- **False positive suppression** through dual-counter persistence gates.
- **Interactive Streamlit dashboard** with real-time video display, alert logging, and module-level configuration.

---

## Pipeline Architecture

```
┌──────────┐      ┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│   M2     │      │   M3     │     │    M4-A      │     │    M4-B      │     │     M5      │
│ YOLOv8   │────> │ DeepSORT │────>│  Loitering   │     │  Skeleton    │────>│  FP         │
│ LLVIP    │      │ Tracker  │     │  Detection   │     │  Transformer │     │  Reduction  │
│ Detection│      │          │     │  (dwell-time)│     │  (v8 Top-2)  │     │             │
└──────────┘      └──────────┘     └──────────────┘     └──────────────┘     └─────────────┘
     │                                    │                    │                    │
     └──── Video Frame ──────────────────>└────────────────────┘──── Alerts ────────┘
```

Each video frame flows left to right:

1. **M2** detects persons in the frame.
2. **M3** assigns persistent track IDs to each detected person.
3. **M4-A** checks whether tracked persons are loitering in a defined virtual zone.
4. **M4-B** classifies the overall scene behaviour (Violence / Threat / Normal) using skeleton pose analysis.
5. **M5** applies persistence thresholds to both M4-A and M4-B outputs to suppress transient false positives.

---

## Module Details

### M2 — Person Detection

| Property | Value |
|---|---|
| **Model** | YOLOv8n (Ultralytics) |
| **Training Data** | LLVIP (Low-Light Visible-Infrared Paired) dataset |
| **Purpose** | Detect persons in low-light and dark conditions |
| **Output** | Bounding boxes `(x1, y1, x2, y2, confidence)` |
| **Weights** | `notebook/runs/darkCondition_yolo/runs/yolov8n_llvip_person/weights/best.pt` |

The detector is fine-tuned specifically for person detection under challenging illumination, making it suitable for surveillance scenarios where standard COCO-trained models struggle.

### M3 — Person Tracking

| Property | Value |
|---|---|
| **Algorithm** | DeepSORT (deep-sort-realtime) |
| **Purpose** | Maintain persistent identity for each detected person across frames |
| **Parameters** | `max_age=30`, `n_init=3`, `max_cosine_distance=0.3` |
| **Output** | Track IDs with centroid positions and bounding boxes |

### M4-A — Loitering Detection

| Property | Value |
|---|---|
| **Method** | Rule-based dwell-time analysis |
| **Purpose** | Identify persons lingering in a user-defined virtual zone |
| **Threshold** | Configurable (default: 10 s × FPS = 250 frames) |
| **Zone Types** | Full frame, Center region, or Custom polygon coordinates |

Tracks how long each person remains inside a virtual zone. When the dwell time exceeds the threshold, the person is flagged as a potential loiterer.

### M4-B — Behaviour Classification

| Property | Value |
|---|---|
| **Architecture** | Skeleton-Transformer (custom Transformer encoder) |
| **Ensemble** | Top-2 ensemble (min4L + min2L), softmax averaging |
| **Pose Backbone** | YOLOv8n-pose (COCO-17 keypoints) |
| **Classes** | Violence, Threat, Normal |
| **Input** | 16-frame clips of skeleton features (283-dim per frame) |
| **Best F1** | 0.617 (Top-2 ensemble, v8) |

The behaviour classifier operates at the scene level:

1. **Pose Extraction** — YOLOv8-pose extracts COCO-17 keypoints for the top-2 detected persons per frame.
2. **Feature Engineering** — Per-frame features include normalised joint coordinates, joint confidences, bone lengths, joint angles, inter-person distances, and temporal velocity/acceleration.
3. **Ensemble Inference** — Two Transformer models (4-layer and 2-layer, both with minimal classification heads) process the feature tensor independently. Their softmax outputs are averaged to produce the final prediction.
4. **Empty-Scene Gate** — If fewer than 50% of frames contain a detected person, the system forces a "Normal" prediction to avoid hallucinated alerts.

### M5 — False Positive Reduction

| Property | Value |
|---|---|
| **Loitering Gate** | Per-track frame counter with configurable persistence threshold (default: 15 frames) |
| **Behaviour Gate** | Scene-level decay counter (threshold ≈ 3 predictions ≈ 1 s of sustained alert) |

Both M4-A and M4-B alerts must persist across multiple consecutive predictions before being elevated to confirmed alerts. This filters out transient, single-frame false alarms.

---

## Demo Application

The system ships with an interactive **Streamlit** dashboard (`app.py`) that provides:

- 🎥 **Real-time video display** with bounding boxes, track IDs, dwell timers, and behaviour overlays.
- ⚙️ **Sidebar configuration** for all module parameters (detection confidence, dwell thresholds, model weights, zone coordinates).
- 📊 **Live metric panels** showing M4-A loitering status, M4-B behaviour probabilities, and pipeline info (FPS, frame count, device).
- 🚨 **Alert log table** with timestamped entries for every confirmed alert.
- 📈 **Session summary** with aggregate statistics at the end of each run.

---

## Installation

### Prerequisites

- **Python** 3.10 or higher
- **CUDA** (optional, for GPU acceleration) — requires a compatible NVIDIA GPU and CUDA toolkit

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/KJYit/smart-surveillance-system
   cd smart-surveillance-system
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   py -m venv .venv

   # Windows
   .venv\Scripts\activate

   # Linux / macOS
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model weights exist:**

   Ensure the following files are present (these are generated from the training notebooks):

   | File | Description |
   |---|---|
   | `notebook/runs/darkCondition_yolo/runs/yolov8n_llvip_person/weights/best.pt` | M2 — LLVIP fine-tuned YOLOv8 weights |
   | `notebook/runs/M4B_classifier/TransformerV8/min4L_s2024.pt` | M4-B — Ensemble member 1 (4-layer Transformer) |
   | `notebook/runs/M4B_classifier/TransformerV8/min2L_s2024.pt` | M4-B — Ensemble member 2 (2-layer Transformer) |
   | `notebook/runs/M4B_classifier/TransformerV8/feat_stats.npz` | M4-B — Feature normalisation statistics |
   | `yolov8n-pose.pt` | YOLOv8-pose for keypoint extraction (auto-downloads if missing) |

---

## Usage

### Running the Application

```bash
streamlit run app.py
```

The dashboard will open in your default browser (typically at `http://localhost:8501`).

---

## Optional: GPU acceleration

The default installation uses CPU-only PyTorch for maximum compatibility.
If you have an NVIDIA GPU with CUDA support and want real-time performance,
install the CUDA build of PyTorch *after* running `pip install -r requirements.txt`:

```bash
# For CUDA 12.1 (check your CUDA version with `nvidia-smi`)
pip install torch --index-url https://download.pytorch.org/whl/cu121 --upgrade
```

---

### Video Input Options

| Mode | Description |
|---|---|
| **Upload file** | Upload a `.mp4`, `.avi`, `.mov`, or `.mkv` video through the browser |
| **Local path** | Select from videos in the `video/` directory, or enter a custom path |
| **Webcam** | Use your system's default webcam (device 0) for live processing |

### Sidebar Configuration

All pipeline parameters can be adjusted through the sidebar before running:

- **Display Settings** — Adjust video display width (400–2000 px).
- **M2 — Person Detection** — Set YOLO weights path and detection confidence threshold.
- **M4-A — Loitering** — Configure dwell time threshold (seconds) and video FPS.
- **M4-B — Behaviour Classifier** — Point to ensemble checkpoint paths, feature stats, and pose weights. Enable/disable the module.
- **M5 — False Positive Reduction** — Set the persistence threshold (frames).
- **Virtual Zone** — Choose zone type (full frame / center region / custom coordinates).

---

## Project Structure

```
FYP-2/
├── app.py                          # Main Streamlit application (full pipeline)
├── requirements.txt                # Python dependencies
├── yolov8n-pose.pt                 # YOLOv8-pose weights (COCO keypoint extraction)
│
├── notebook/                       # Training & evaluation notebooks
│   ├── lowLightFineTuneLLVIP.ipynb          # M2: LLVIP fine-tuning
│   ├── DarkCondition_Comparison.ipynb       # M2: Dark condition evaluation
│   ├── crowdedCondition.ipynb               # M3: Crowded scene evaluation
│   ├── CrowdedCondition_Comparison.ipynb    # M3: Tracker comparison
│   ├── TrackerComparison.ipynb              # M3: Detailed tracker benchmarks
│   ├── M4A_RuleBasedLoitering.ipynb         # M4-A: Loitering rule-based analysis
│   ├── M4B_SkeletonTransformerV8.ipynb      # M4-B: Final Skeleton-Transformer (v8)
│   ├── M4B_BehaviourClassifier_Training*.ipynb  # M4-B: CNN-LSTM iterations (v1–v4)
│   ├── M4B_SkeletonLSTM.ipynb               # M4-B: Skeleton-LSTM baseline
│   ├── M4B_SkeletonTransformerV*.ipynb       # M4-B: Transformer iterations (V2–V7)
│   ├── M4B_TransformerTuning.ipynb           # M4-B: Hyperparameter search
│   │
│   └── runs/                       # Saved model weights & training outputs
│       ├── darkCondition_yolo/             # M2 YOLOv8 LLVIP runs
│       ├── crowdedCondition_yolo/          # Crowded condition YOLOv8 runs
│       └── M4B_classifier/                # All M4-B model checkpoints
│           ├── CNNLSTM*/                       # CNN-LSTM iterations
│           ├── SkeletonLSTM*/                  # Skeleton-LSTM iterations
│           ├── TransformerV2–V7*/               # Transformer iterations
│           └── TransformerV8/                  # ★ Final production checkpoints
│               ├── min4L_s2024.pt              #   Ensemble member 1
│               ├── min2L_s2024.pt              #   Ensemble member 2
│               └── feat_stats.npz              #   Feature normalisation stats
│
├── video/                          # Sample test videos
│   ├── crosswalk.avi
│   ├── night.avi
│   └── TownCentreXVID.mp4
│   └── normal.mp4
│   └── fight1.mp4
│   └── fight2.mp4
│
├── evaluationDataset/              # Evaluation datasets & ground truth
│   ├── Avenue Dataset/
│   └── ground_truth_demo/
│
├── Anomaly_Videos/                 # Anomaly video samples
├── LLVIP/                          # LLVIP dataset (low-light fine-tuning)
├── MOT17/                          # MOT17 dataset (multi-object tracking)
├── darkCondition/                  # Dark condition test data
├── darkCondition_yolo/             # Dark condition YOLO format data
├── M4B_clips/                      # M4-B raw video clips
├── M4B_skeleton_clips/             # M4-B skeleton feature clips
└── M4B_skeleton_clips_v2/         # M4-B skeleton feature clips (v2)
```

---

## Training Notebooks

The `notebook/` directory contains the full experimental history. The key notebooks for reproducing the final system are:

| Notebook | Module | Description |
|---|---|---|
| `lowLightFineTuneLLVIP.ipynb` | M2 | Fine-tunes YOLOv8n on LLVIP for low-light person detection |
| `TrackerComparison.ipynb` | M3 | Benchmarks DeepSORT vs. ByteTrack vs. BoT-SORT |
| `M4A_RuleBasedLoitering.ipynb` | M4-A | Rule-based loitering detection development & evaluation |
| `M4B_SkeletonTransformerV8.ipynb` | M4-B | **Final** Skeleton-Transformer training with Top-2 ensemble selection |
| `DarkCondition_Comparison.ipynb` | M2 | Evaluates detection performance across lighting conditions |
| `CrowdedCondition_Comparison.ipynb` | M3 | Evaluates tracking in crowded scenes |

---

## Configuration

### Default Model Paths

All model paths are configurable via the Streamlit sidebar. The defaults are:

```python
DEFAULT_M2_WEIGHTS   = "notebook/runs/darkCondition_yolo/runs/yolov8n_llvip_person/weights/best.pt"
DEFAULT_M4B_W1       = "notebook/runs/M4B_classifier/TransformerV8/min4L_s2024.pt"
DEFAULT_M4B_W2       = "notebook/runs/M4B_classifier/TransformerV8/min2L_s2024.pt"
DEFAULT_M4B_STATS    = "notebook/runs/M4B_classifier/TransformerV8/feat_stats.npz"
DEFAULT_POSE_WEIGHTS = "yolov8n-pose.pt"
```

### Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| Detection confidence | 0.35 | M2 — Minimum confidence for person detections |
| Dwell time threshold | 10 s | M4-A — Seconds in zone before flagging as loitering |
| Clip length | 16 frames | M4-B — Number of frames per skeleton clip |
| Stride | 8 frames | M4-B — Prediction interval (50% overlap) |
| Persistence threshold | 15 frames | M5 — Frames of sustained loitering before confirmation |
| Scene threshold | 3 predictions | M5 — Consecutive threat predictions before confirmation |

---

## Tech Stack

| Component | Technology |
|---|---|
| **Web Framework** | Streamlit |
| **Object Detection** | YOLOv8 (Ultralytics) |
| **Pose Estimation** | YOLOv8-Pose (Ultralytics) |
| **Object Tracking** | DeepSORT (deep-sort-realtime) |
| **Deep Learning** | PyTorch |
| **Computer Vision** | OpenCV |
| **Language** | Python 3.10+ |

---
 
## Acknowledgements
 
This project would not have been possible without the following publicly available datasets:
 
- **LLVIP** — A visible-infrared paired dataset for low-light vision, used for fine-tuning the M2 person detector to improve robustness on night surveillance footage.
  > Jia, X., Zhu, C., Li, M., Tang, W., & Zhou, W. (2021). LLVIP: A Visible-infrared Paired Dataset for Low-light Vision. *Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)*, 3496–3504.
 
- **Oxford Town Centre** — A high-resolution CCTV pedestrian dataset, used for cross-domain evaluation of M2 and the CentroidTracker vs DeepSORT comparison in M3.
  > Benfold, B., & Reid, I. (2011). Stable Multi-Target Tracking in Real-Time Surveillance Video. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 3457–3464.
 
- **CUHK Avenue** — A surveillance anomaly detection benchmark, used to evaluate the M4-A loitering module and the M5 false-positive reduction filter.
  > Lu, C., Shi, J., & Jia, J. (2013). Abnormal Event Detection at 150 FPS in MATLAB. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2720–2727.
 
- **UCF-Crime** — A large-scale real-world surveillance video dataset spanning multiple anomaly classes, used for training and validation of the M4-B Skeleton-Transformer behaviour classifier.
  > Sultani, W., Chen, C., & Shah, M. (2018). Real-world Anomaly Detection in Surveillance Videos. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 6479–6488.
 
The project also builds on the open-source work of the **Ultralytics YOLOv8** team and the **deep-sort-realtime** community, whose model implementations form the backbone of the M2, M3, and M4-B pose extraction modules.
 
---

## Datasets

| Dataset | Module | Purpose |
|---|---|---|
| **LLVIP** | M2 | Fine-tuning YOLOv8 for low-light person detection |
| **MOT17** | M3 | Evaluating multi-object tracking accuracy |
| **UCF-Crime** | M4-B | Training skeleton-based behaviour classifiers (Violence / Threat / Normal) |
| **COCO Keypoints** | M4-B | Pre-trained YOLOv8-pose for 17-joint skeleton extraction |
| **Avenue Dataset** | Evaluation | Anomaly detection benchmark for pipeline evaluation |

---

## License
This project was developed as a Final Year Project for academic purposes.