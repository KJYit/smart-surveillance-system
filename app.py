"""
Smart Surveillance System — Full Pipeline Streamlit App
M2 (YOLOv8-LLVIP) → M3 (DeepSORT) → M4-A (Loitering) + M4-B (Skeleton-Transformer v8 Top-2) → M5 (FP Reduction)

M4-B: Skeleton-based behaviour classification using a 2-model ensemble from v8:
      - min4L_s2024 (4-layer Transformer, minimal head, frozen body)
      - min2L_s2024 (2-layer Transformer, minimal head, frozen body)
      Predictions are fused by softmax averaging (Top-2 ensemble, F1=0.617).
      Pose is extracted live via YOLOv8-pose (COCO-17 keypoints).

Usage:
    pip install streamlit ultralytics deep-sort-realtime opencv-python torch numpy
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import tempfile
import time
import os
import math
from collections import defaultdict, OrderedDict, deque
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — UPDATE THESE PATHS
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_M2_WEIGHTS  = r"notebook/runs/darkCondition_yolo/runs/yolov8n_llvip_person/weights/best.pt"
DEFAULT_M4B_W1      = r"notebook/runs/M4B_classifier/TransformerV8/min4L_s2024.pt"
DEFAULT_M4B_W2      = r"notebook/runs/M4B_classifier/TransformerV8/min2L_s2024.pt"
DEFAULT_M4B_STATS   = r"notebook/runs/M4B_classifier/TransformerV8/feat_stats.npz"
DEFAULT_POSE_WEIGHTS = r"yolov8n-pose.pt"   # auto-downloads if missing
DEFAULT_VIDEO_DIR   = r"video"
TARGET_CLASSES      = ["Violence", "Threat", "Normal"]

# ── M4-B skeleton feature spec (must match v8 training) ─────────────────────
SKEL_CLIP_FRAMES = 16
MAX_PERSONS      = 2
N_KEYPOINTS      = 17
FEATURE_DIM      = 283   # from v8 notebook

BONES = [(5,7),(7,9),(6,8),(8,10),(5,6),(11,13),(13,15),(12,14),(14,16),
         (11,12),(5,11),(6,12),(0,5),(0,6),(0,1),(0,2)]
ANGLE_TRIPLETS = [(5,7,9),(6,8,10),(11,13,15),(12,14,16),(5,0,6)]


# ══════════════════════════════════════════════════════════════════════════════
# M2: Person Detection (YOLOv8 LLVIP fine-tuned)
# ══════════════════════════════════════════════════════════════════════════════
class PersonDetectionModule:
    def __init__(self, weights, conf=0.35, iou=0.45):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou

    def detect(self, frame):
        res = self.model(frame, conf=self.conf, iou=self.iou,
                         classes=[0], verbose=False)[0]
        dets = []
        if res.boxes is not None:
            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                dets.append((x1, y1, x2, y2, float(b.conf[0])))
        return dets


# ══════════════════════════════════════════════════════════════════════════════
# M3: DeepSORT Tracker
# ══════════════════════════════════════════════════════════════════════════════
class DeepSORTTracker:
    def __init__(self, max_age=30, n_init=3, max_cosine_distance=0.3, nn_budget=100):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self.tracker = DeepSort(
            max_age=max_age, n_init=n_init,
            max_cosine_distance=max_cosine_distance, nn_budget=nn_budget,
        )

    def update(self, frame, dets):
        ds_input = [([x1, y1, x2-x1, y2-y1], cf, 'person') for (x1,y1,x2,y2,cf) in dets]
        tracks = self.tracker.update_tracks(ds_input, frame=frame)
        objects, boxes = OrderedDict(), OrderedDict()
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            l, top, r, bot = t.to_ltrb()
            x1, y1, x2, y2 = int(l), int(top), int(r), int(bot)
            objects[tid] = np.array([(x1+x2)//2, (y1+y2)//2])
            boxes[tid] = (x1, y1, x2, y2)
        return objects, boxes


# ══════════════════════════════════════════════════════════════════════════════
# M4-A: Loitering Detection (dwell-time in virtual zone)
# ══════════════════════════════════════════════════════════════════════════════
class LoiteringDetector:
    def __init__(self, zone_polygon, loitering_threshold_frames=250, fps=25):
        self.zone_polygon = np.array(zone_polygon, np.int32)
        self.loitering_threshold = loitering_threshold_frames
        self.fps = fps
        self.dwell_timers = defaultdict(int)

    def is_in_zone(self, centroid):
        return cv2.pointPolygonTest(
            self.zone_polygon, (int(centroid[0]), int(centroid[1])), False) >= 0

    def update(self, tracked_objects):
        suspicious_ids = set()
        for tid, centroid in tracked_objects.items():
            if self.is_in_zone(centroid):
                self.dwell_timers[tid] += 1
            else:
                self.dwell_timers[tid] = 0
            if self.dwell_timers[tid] >= self.loitering_threshold:
                suspicious_ids.add(tid)
        return suspicious_ids, dict(self.dwell_timers)

    def get_dwell_seconds(self, tid):
        return self.dwell_timers.get(tid, 0) / self.fps


# ══════════════════════════════════════════════════════════════════════════════
# M4-B: Skeleton-Transformer v8 Top-2 Ensemble (scene-level behaviour)
# ══════════════════════════════════════════════════════════════════════════════
class SkelTransformer(nn.Module):
    """Must match v8 architecture exactly (see M4B_SkeletonTransformerV8.ipynb)."""
    def __init__(self, input_dim, nc=3, d=128, nh=4, nl=4, ff=256, drop=0.3, head='minimal'):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d), nn.LayerNorm(d), nn.Dropout(drop * 0.3))
        self.pos = nn.Parameter(torch.randn(1, SKEL_CLIP_FRAMES, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=nh, dim_feedforward=ff, dropout=drop,
            batch_first=True, activation='gelu', norm_first=True)
        self.tf = nn.TransformerEncoder(enc, num_layers=nl)
        self.norm = nn.LayerNorm(d)
        if head == 'minimal':
            self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d, nc))
        else:
            self.head = nn.Sequential(
                nn.Dropout(drop), nn.Linear(d, d // 2), nn.GELU(),
                nn.Dropout(drop * 0.5), nn.Linear(d // 2, nc))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        pos = torch.cat(
            [torch.zeros(1, 1, x.shape[2], device=x.device), self.pos], dim=1)
        x = self.tf(x + pos)
        return self.head(self.norm(x[:, 0]))


# ── Feature extraction helpers (must match v8 CleanDataset._build) ────────────
def _compute_bones(kps):
    return np.array(
        [math.sqrt((kps[a, 0] - kps[b, 0]) ** 2
                   + (kps[a, 1] - kps[b, 1]) ** 2 + 1e-8) for a, b in BONES],
        dtype=np.float32)


def _compute_angles(kps):
    out = []
    for a, v, b in ANGLE_TRIPLETS:
        va, vb = kps[a] - kps[v], kps[b] - kps[v]
        c = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
        out.append(math.acos(np.clip(c, -1, 1)) / math.pi)
    return np.array(out, dtype=np.float32)


def _compute_inter(k1, k2):
    cd = np.linalg.norm(k1[:, :2].mean(0) - k2[:, :2].mean(0))
    md = np.linalg.norm(k1[:, :2][:, None] - k2[:, :2][None, :], axis=2).min()
    hh = min(np.linalg.norm(k1[9, :2] - k2[0, :2]),
             np.linalg.norm(k1[10, :2] - k2[0, :2]),
             np.linalg.norm(k2[9, :2] - k1[0, :2]),
             np.linalg.norm(k2[10, :2] - k1[0, :2]))
    return np.array([cd, md, hh], dtype=np.float32)


def build_feature_tensor(clip):
    """Convert (T, MAX_PERSONS, 17, 3) keypoint clip → (T, FEATURE_DIM) features.

    Replicates CleanDataset._build from the v8 notebook exactly: impute low-conf
    joints with per-joint mean, then per frame concatenate
    [xy, conf, bones, angles] × persons + inter-person distances, then append
    velocity and acceleration deltas for each person's xy.
    """
    T = clip.shape[0]
    # impute low-confidence joints with per-joint mean over visible frames
    for p in range(MAX_PERSONS):
        v = clip[:, p, :, 2] > 0.3
        for j in range(N_KEYPOINTS):
            if v[:, j].sum() > 0:
                mp = clip[v[:, j], p, j, :2].mean(0)
            else:
                mp = np.array([0.5, 0.5], dtype=np.float32)
            for t in range(T):
                if not v[t, j]:
                    clip[t, p, j, :2] = mp
                    clip[t, p, j, 2] = 0.0

    rows = []
    for t in range(T):
        f = []
        for p in range(MAX_PERSONS):
            k = clip[t, p]
            f.extend([k[:, :2].flatten(), k[:, 2],
                      _compute_bones(k), _compute_angles(k)])
        f.append(_compute_inter(clip[t, 0], clip[t, 1]))
        rows.append(np.concatenate(f))
    feat = np.stack(rows)

    # velocity + acceleration on xy of each person (first 34 cols of each 72-block)
    for p in range(MAX_PERSONS):
        s = p * 72
        pos = feat[:, s:s + 34]
        vel = np.zeros_like(pos); vel[1:] = pos[1:] - pos[:-1]
        acc = np.zeros_like(vel); acc[2:] = vel[2:] - vel[1:-1]
        feat = np.concatenate([feat, vel, acc], axis=1)
    return feat.astype(np.float32)


# ── Live pose extractor (YOLOv8-pose) ─────────────────────────────────────────
class PoseExtractor:
    """Extracts top-2 persons (by detection confidence) as COCO-17 keypoints,
    normalised to [0,1] by frame size. Output: (MAX_PERSONS, 17, 3) per frame.
    """
    def __init__(self, weights=DEFAULT_POSE_WEIGHTS, conf=0.3):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf

    def __call__(self, frame):
        h, w = frame.shape[:2]
        res = self.model(frame, conf=self.conf, verbose=False)[0]
        out = np.zeros((MAX_PERSONS, N_KEYPOINTS, 3), dtype=np.float32)
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return out
        # rank persons by box confidence, take top MAX_PERSONS
        box_conf = res.boxes.conf.cpu().numpy()
        order = np.argsort(-box_conf)[:MAX_PERSONS]
        kxy = res.keypoints.xy.cpu().numpy()     # (N, 17, 2) in pixel coords
        kcf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None \
              else np.ones(kxy.shape[:2], dtype=np.float32)
        for slot, idx in enumerate(order):
            out[slot, :, 0] = kxy[idx, :, 0] / max(w, 1)
            out[slot, :, 1] = kxy[idx, :, 1] / max(h, 1)
            out[slot, :, 2] = kcf[idx]
        return out


class BehaviourClassifier:
    """Skeleton-Transformer v8 Top-2 ensemble.

    Two checkpoints (min4L_s2024 nl=4 + min2L_s2024 nl=2) are averaged in
    softmax space. Requires the feature-normalisation stats (feat_stats.npz
    with keys 'fm' and 'fs') saved from the v8 training notebook.
    """
    def __init__(self, w1_path, w2_path, stats_path, pose_weights=DEFAULT_POSE_WEIGHTS,
                 device="cpu", clip_frames=SKEL_CLIP_FRAMES, stride=None):
        self.device = device
        self.clip_frames = clip_frames
        self.stride = stride or (clip_frames // 2)     # 50% overlap, matches training eval
        self.kp_buffer = deque(maxlen=clip_frames)
        self.frames_since_pred = 0
        self.model_loaded = False
        self.load_error = None

        # normalisation stats
        self.fm = None
        self.fs = None
        if Path(stats_path).exists():
            try:
                st_ = np.load(stats_path)
                self.fm = st_['fm'].astype(np.float32)
                self.fs = st_['fs'].astype(np.float32)
            except Exception as e:
                self.load_error = f"stats load failed: {e}"
                return
        else:
            self.load_error = (
                f"feature stats not found at {stats_path}. Export FM/FS from the "
                f"v8 notebook via: np.savez('feat_stats.npz', fm=FM, fs=FS)")
            return

        # two ensemble members
        self.models = []
        for path, nl in [(w1_path, 4), (w2_path, 2)]:
            if not Path(path).exists():
                self.load_error = f"checkpoint not found: {path}"
                return
            m = SkelTransformer(FEATURE_DIM, nc=len(TARGET_CLASSES),
                                 nl=nl, head='minimal').to(device)
            try:
                m.load_state_dict(torch.load(path, map_location=device))
            except Exception as e:
                self.load_error = f"load {Path(path).name}: {e}"
                return
            m.eval()
            self.models.append(m)

        # pose extractor
        try:
            self.pose = PoseExtractor(weights=pose_weights)
        except Exception as e:
            self.load_error = f"pose extractor failed: {e}"
            return

        self.model_loaded = True

    def add_frame(self, frame):
        """Feed one BGR frame. Returns a prediction dict every `stride` frames
        once the buffer is full, else None."""
        if not self.model_loaded:
            return None
        kps = self.pose(frame)   # (MAX_PERSONS, 17, 3)
        self.kp_buffer.append(kps)
        if len(self.kp_buffer) < self.clip_frames:
            return None
        self.frames_since_pred += 1
        if self.frames_since_pred < self.stride:
            return None
        self.frames_since_pred = 0
        return self._predict()

    def _predict(self):
        clip = np.stack(list(self.kp_buffer), axis=0).copy()   # (T, P, 17, 3)

        # ── Empty-scene gate ──────────────────────────────────────────────
        # The skeleton model was trained exclusively on clips containing
        # people (UCF-Crime). If YOLOv8-pose found no person in most frames
        # of the current clip, the feature vector is out-of-distribution and
        # the model's output is meaningless. Force a Normal prediction in
        # that case instead of letting the model hallucinate a threat.
        # A frame is considered "populated" if at least one joint in the top
        # person slot has confidence > 0.3 (the same threshold used during
        # training for joint imputation).
        per_frame_populated = (clip[:, 0, :, 2] > 0.3).any(axis=1)
        populated_frac = float(per_frame_populated.mean())
        if populated_frac < 0.5:
            return {
                "prediction": "Normal",
                "confidence": 1.0,
                "probabilities": {"Violence": 0.0, "Threat": 0.0, "Normal": 1.0},
                "is_alert": False,
                "gated_empty_scene": True,
            }

        feat = build_feature_tensor(clip)                       # (T, FEATURE_DIM)
        feat = (feat - self.fm) / (self.fs + 1e-8)
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)

        probs_sum = None
        with torch.no_grad():
            for m in self.models:
                p = torch.softmax(m(x), dim=1)[0]
                probs_sum = p if probs_sum is None else probs_sum + p
        probs = probs_sum / len(self.models)
        pred_idx = int(probs.argmax().item())
        # Alert only if the predicted (top) class is Violence or Threat.
        # A looser confidence gate (0.4) handles borderline cases, and the
        # combined V+T > N condition catches cases where the threat mass is
        # split across Violence and Threat but the top class is still one of
        # them (not Normal).
        is_non_normal_top = TARGET_CLASSES[pred_idx] != "Normal"
        top_conf = probs[pred_idx].item()
        v_plus_t = (probs[TARGET_CLASSES.index("Violence")].item()
                    + probs[TARGET_CLASSES.index("Threat")].item())
        n_prob = probs[TARGET_CLASSES.index("Normal")].item()
        return {
            "prediction": TARGET_CLASSES[pred_idx],
            "confidence": float(top_conf),
            "probabilities": {c: float(probs[i].item()) for i, c in enumerate(TARGET_CLASSES)},
            "is_alert": is_non_normal_top and (top_conf > 0.4 or v_plus_t > n_prob),
            "gated_empty_scene": False,
        }

    @property
    def buffer_fill(self):
        return len(self.kp_buffer)


# ══════════════════════════════════════════════════════════════════════════════
# M5: False Positive Reduction (dual-counter)
# ══════════════════════════════════════════════════════════════════════════════
class FalsePositiveReduction:
    def __init__(self, persistence_threshold=15, scene_threshold=None, scene_decay=2):
        self.persistence_threshold = persistence_threshold      # for per-track loitering (frame-level)
        # Scene counter runs on M4-B predictions, which are issued every `stride`
        # frames (default 8). A threshold of ~3 predictions ≈ 1 second of
        # sustained alert, which matches the typical duration of a real fight
        # burst while still filtering single-prediction flickers.
        self.scene_threshold = scene_threshold if scene_threshold is not None else 3
        self.scene_decay = scene_decay
        self.track_counters = defaultdict(int)
        self.scene_counter = 0

    def validate_loitering(self, track_id, is_suspicious):
        if is_suspicious:
            self.track_counters[track_id] += 1
        else:
            self.track_counters[track_id] = 0
        return self.track_counters[track_id] >= self.persistence_threshold

    def validate_behaviour(self, is_threat):
        # Decay-based counter on M4-B predictions (called once per prediction,
        # not once per frame). Tolerates brief flicker while a sustained run
        # of Normal still drains the counter to zero.
        if is_threat:
            self.scene_counter = min(self.scene_counter + 1,
                                      self.scene_threshold * 2)
        else:
            self.scene_counter = max(self.scene_counter - self.scene_decay, 0)
        return self.scene_counter >= self.scene_threshold


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Surveillance System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global theme / CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* App background */
    .stApp { background: linear-gradient(180deg, #0e1117 0%, #11151c 100%); }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f766e 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        color: #ffffff; margin: 0; font-size: 1.85rem; font-weight: 700;
        letter-spacing: -0.02em;
    }
    .hero p {
        color: rgba(255,255,255,0.78); margin: 0.35rem 0 0 0; font-size: 0.92rem;
    }
    .hero .pipeline {
        margin-top: 0.9rem; display: flex; flex-wrap: wrap; gap: 0.4rem;
    }
    .hero .stage {
        background: rgba(255,255,255,0.12);
        color: #fff; padding: 0.25rem 0.7rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 500;
        border: 1px solid rgba(255,255,255,0.18);
    }
    .hero .arrow { color: rgba(255,255,255,0.5); align-self: center; }

    /* Section card */
    .section-card {
        background: #161b24; border: 1px solid #232936;
        border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    }
    .section-title {
        font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 0.5rem;
    }

    /* Metric panels */
    .metric-panel {
        background: #161b24; border: 1px solid #232936;
        border-radius: 10px; padding: 0.9rem 1rem; height: 100%;
    }
    .metric-panel.m4a { border-left: 3px solid #f59e0b; }
    .metric-panel.m4b { border-left: 3px solid #06b6d4; }
    .metric-panel.info { border-left: 3px solid #8b5cf6; }
    .metric-label {
        font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.06em; color: #94a3b8; margin-bottom: 0.4rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px; font-weight: 600; transition: all 0.15s ease;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        border: none; box-shadow: 0 2px 8px rgba(20,184,166,0.3);
    }
    .stButton > button[kind="primary"]:hover { transform: translateY(-1px); }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0b0e14; border-right: 1px solid #1f2937; }
    [data-testid="stSidebar"] h1 { font-size: 1.1rem; }

    /* Hide default Streamlit chrome */
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding: 0.6rem 0 1rem 0; border-bottom: 1px solid #1f2937; margin-bottom: 1rem;">
    <div style="font-size: 1.15rem; font-weight: 700; color: #f1f5f9;">⚙️ Configuration</div>
    <div style="font-size: 0.78rem; color: #64748b; margin-top: 0.2rem;">Pipeline modules & parameters</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar.expander("Display Settings", expanded=False):
    output_width = st.slider("Video Display Width (Pixels)", 400, 2000, 1000, 50)
    st.caption("Adjust this to make the video window larger or smaller. Note: Changing this while running will stop the video.")

with st.sidebar.expander("M2 — Person Detection", expanded=False):
    m2_weights = st.text_input("YOLO weights", DEFAULT_M2_WEIGHTS)
    conf_threshold = st.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)

with st.sidebar.expander("M4-A — Loitering", expanded=False):
    dwell_seconds = st.slider("Dwell time threshold (s)", 3, 30, 10)
    fps_setting = st.number_input("Video FPS", 10, 60, 25)
    loitering_frames = dwell_seconds * fps_setting

with st.sidebar.expander("M4-B — Behaviour Classifier (Skeleton-Transformer v8 Top-2)", expanded=False):
    m4b_w1 = st.text_input("Member 1 — min4L_s2024", DEFAULT_M4B_W1)
    m4b_w2 = st.text_input("Member 2 — min2L_s2024", DEFAULT_M4B_W2)
    m4b_stats = st.text_input("Feature stats (feat_stats.npz)", DEFAULT_M4B_STATS)
    m4b_pose = st.text_input("Pose weights (YOLOv8-pose)", DEFAULT_POSE_WEIGHTS)
    m4b_enabled = st.checkbox("Enable M4-B", value=True)

with st.sidebar.expander("M5 — False Positive Reduction", expanded=False):
    persistence = st.slider("Persistence threshold (frames)", 5, 30, 15)

with st.sidebar.expander("Virtual Zone", expanded=False):
    zone_mode = st.selectbox("Zone type", ["Full frame", "Center region", "Custom coordinates"])
    if zone_mode == "Custom coordinates":
        st.caption("4 corner points (x,y) clockwise from top-left")
        z_x1 = st.sidebar.number_input("Top-left X", 0, 3840, 100)
        z_y1 = st.sidebar.number_input("Top-left Y", 0, 2160, 100)
        z_x2 = st.sidebar.number_input("Top-right X", 0, 3840, 800)
        z_y2 = st.sidebar.number_input("Top-right Y", 0, 2160, 100)
        z_x3 = st.sidebar.number_input("Bottom-right X", 0, 3840, 800)
        z_y3 = st.sidebar.number_input("Bottom-right Y", 0, 2160, 600)
        z_x4 = st.sidebar.number_input("Bottom-left X", 0, 3840, 100)
        z_y4 = st.sidebar.number_input("Bottom-left Y", 0, 2160, 600)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Smart Surveillance System</h1>
    <p>Real-time multi-module pipeline for person detection, tracking, loitering & behaviour analysis</p>
    <div class="pipeline">
        <span class="stage">M2 · Detection</span><span class="arrow">→</span>
        <span class="stage">M3 · Tracking</span><span class="arrow">→</span>
        <span class="stage">M4-A · Loitering</span><span class="arrow">+</span>
        <span class="stage">M4-B · Behaviour</span><span class="arrow">→</span>
        <span class="stage">M5 · FP Reduction</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Video source ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📹 Video Source</div>', unsafe_allow_html=True)
source_tab = st.radio("Video source", ["Upload file", "Local path", "Webcam"],
                      horizontal=True, label_visibility="collapsed")

video_file = None
video_path_input = None
use_webcam = False

if source_tab == "Upload file":
    video_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])
elif source_tab == "Local path":
    available_videos = []
    if os.path.isdir(DEFAULT_VIDEO_DIR):
        for f in sorted(os.listdir(DEFAULT_VIDEO_DIR)):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                available_videos.append(os.path.join(DEFAULT_VIDEO_DIR, f))
    if available_videos:
        video_path_input = st.selectbox("Select video", available_videos)
    else:
        video_path_input = st.text_input("Video file path", "../video/crosswalk.avi")
else:
    use_webcam = True

# ── Run / Stop ────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top: 0.6rem;'></div>", unsafe_allow_html=True)
col_run, col_stop = st.columns([3, 1])
with col_run:
    run_clicked = st.button("▶  Run Pipeline", type="primary", use_container_width=True)
with col_stop:
    if st.button("⏹  Stop", use_container_width=True):
        st.session_state.stop_requested = True

# ── Main processing ───────────────────────────────────────────────────────────
if run_clicked:
    st.session_state.stop_requested = False

    # Resolve video path
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        video_path = tfile.name
        video_name = video_file.name
    elif video_path_input:
        if not os.path.exists(video_path_input):
            st.error(f"File not found: {video_path_input}")
            st.stop()
        video_path = video_path_input
        video_name = os.path.basename(video_path_input)
    elif use_webcam:
        video_path = 0
        video_name = "Webcam"
    else:
        st.warning("Select a video source first.")
        st.stop()

    # ── Load modules ──────────────────────────────────────────────────────────
    load_msg = st.empty()

    load_msg.info("Loading M2 (YOLOv8 LLVIP)...")
    try:
        detector = PersonDetectionModule(m2_weights, conf=conf_threshold)
    except Exception as e:
        st.error(f"M2 failed: {e}")
        st.stop()

    load_msg.info("Loading M3 (DeepSORT)...")
    tracker = DeepSORTTracker(max_age=30, n_init=3)

    # Video info
    cap_info = cv2.VideoCapture(video_path)
    frame_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    video_fps = cap_info.get(cv2.CAP_PROP_FPS) or fps_setting
    cap_info.release()

    # Zone
    if zone_mode == "Full frame":
        zone_poly = [[30,30],[frame_w-30,30],[frame_w-30,frame_h-30],[30,frame_h-30]]
    elif zone_mode == "Center region":
        mx, my = frame_w//4, frame_h//4
        zone_poly = [[mx,my],[frame_w-mx,my],[frame_w-mx,frame_h-my],[mx,frame_h-my]]
    else:
        zone_poly = [[z_x1,z_y1],[z_x2,z_y2],[z_x3,z_y3],[z_x4,z_y4]]

    loiterer = LoiteringDetector(zone_poly, loitering_frames, video_fps)

    # M4-B
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = None
    if m4b_enabled:
        load_msg.info("Loading M4-B (Skeleton-Transformer v8 Top-2 ensemble + YOLOv8-pose)...")
        classifier = BehaviourClassifier(
            w1_path=m4b_w1, w2_path=m4b_w2, stats_path=m4b_stats,
            pose_weights=m4b_pose, device=device)
        if not classifier.model_loaded:
            st.warning(f"M4-B disabled — {classifier.load_error}")
            classifier = None
            m4b_enabled = False

    m5 = FalsePositiveReduction(persistence_threshold=persistence)

    dur = f"{total_frames/video_fps:.1f}s" if total_frames > 0 and video_fps > 0 else "live"
    load_msg.success(
        f"Ready — {video_name} ({frame_w}×{frame_h}, {video_fps:.0f}fps, {dur}) on {device.upper()}"
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    st.markdown("<div style='margin: 1.2rem 0 0.4rem 0;'></div>", unsafe_allow_html=True)
    frame_placeholder = st.empty()
    progress_bar = st.progress(0) if total_frames > 0 else None

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    col_m4a, col_m4b, col_info = st.columns(3)
    with col_m4a:
        st.markdown(
            '<div class="metric-panel m4a">'
            '<div class="metric-label">🟠 M4-A · Loitering</div>',
            unsafe_allow_html=True)
        m4a_display = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    with col_m4b:
        st.markdown(
            '<div class="metric-panel m4b">'
            '<div class="metric-label">🔵 M4-B · Behaviour</div>',
            unsafe_allow_html=True)
        m4b_display = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    with col_info:
        st.markdown(
            '<div class="metric-panel info">'
            '<div class="metric-label">🟣 Pipeline Info</div>',
            unsafe_allow_html=True)
        info_display = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    alert_header = st.empty()
    alert_display = st.empty()

    # ── Loop ──────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    alert_log = []
    m4b_result = None
    m4b_confirmed = False
    m4b_confirmed_prev = False

    while cap.isOpened():
        if st.session_state.stop_requested:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t0 = time.time()

        # M2
        detections = detector.detect(frame)

        # M3
        tracked_objects, tracked_boxes = tracker.update(frame, detections)

        # M4-A
        m4a_suspicious, _ = loiterer.update(tracked_objects)

        # M4-B — produces a new prediction every `stride` frames
        m4b_new_result = None
        if m4b_enabled and classifier:
            m4b_new_result = classifier.add_frame(frame)
            if m4b_new_result is not None:
                m4b_result = m4b_new_result   # cache latest for UI display

        # M5 — only step the scene counter on frames where M4-B produced a new
        # prediction, so each prediction counts exactly once regardless of stride
        if m4b_new_result is not None:
            m4b_is_threat = m4b_new_result["is_alert"]
            m4b_confirmed = m5.validate_behaviour(m4b_is_threat)
        else:
            m4b_is_threat = m4b_result is not None and m4b_result["is_alert"]
            # counter state unchanged; reuse previous confirmation status
            m4b_confirmed = m5.scene_counter >= m5.scene_threshold

        # ── Draw ──────────────────────────────────────────────────────────────
        display = frame.copy()

        # Zone overlay (semi-transparent fill + border)
        zone_np = np.array(zone_poly, np.int32)
        overlay = display.copy()
        cv2.fillPoly(overlay, [zone_np], (255, 255, 0))
        cv2.addWeighted(overlay, 0.08, display, 0.92, 0, display)
        cv2.polylines(display, [zone_np], True, (255, 255, 0), 2)
        cv2.putText(display, "VIRTUAL ZONE", (zone_poly[0][0]+5, zone_poly[0][1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        for tid, centroid in tracked_objects.items():
            is_raw = tid in m4a_suspicious
            is_confirmed = m5.validate_loitering(tid, is_raw)
            dwell_sec = loiterer.get_dwell_seconds(tid)
            in_zone = loiterer.is_in_zone(centroid)

            if is_confirmed:
                color = (0, 0, 255)
            elif is_raw:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            if tid in tracked_boxes:
                bx1, by1, bx2, by2 = tracked_boxes[tid]
                cv2.rectangle(display, (bx1, by1), (bx2, by2), color, 2)

                label = f"ID {tid}"
                if in_zone:
                    label += f" | {dwell_sec:.1f}s"
                lw = len(label) * 9 + 8
                cv2.rectangle(display, (bx1, by1-22), (bx1+lw, by1), color, -1)
                cv2.putText(display, label, (bx1+4, by1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(display, (cx, cy), 4, color, -1)

            if is_confirmed and tid in tracked_boxes:
                bx1, by1, _, _ = tracked_boxes[tid]
                cv2.rectangle(display, (bx1, by1-46), (bx1+200, by1-24), (0,0,200), -1)
                cv2.putText(display, "ALERT: LOITERING", (bx1+5, by1-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                aid = f"l_{tid}_{frame_count//50}"
                if aid not in [a.get("id") for a in alert_log[-30:]]:
                    alert_log.append({
                        "id": aid, "frame": frame_count,
                        "time": f"{frame_count/video_fps:.1f}s",
                        "channel": "M4-A", "type": "Loitering",
                        "detail": f"ID {tid}, {dwell_sec:.1f}s",
                    })

        # M4-B overlay
        if m4b_result:
            pred = m4b_result["prediction"]
            conf_val = m4b_result["confidence"]
            if m4b_confirmed:
                cv2.rectangle(display, (10,10), (380,60), (0,0,200), -1)
                cv2.putText(display, f"M4-B ALERT: {pred} ({conf_val:.0%})",
                            (20,42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                # Log only once per confirmed event: on the transition from
                # not-confirmed → confirmed. The alert stays "active" visually
                # as long as the counter is above threshold, but the log row
                # is written exactly once at the start of each event.
                if not m4b_confirmed_prev:
                    alert_log.append({
                        "id": f"b_{frame_count}",
                        "frame": frame_count,
                        "time": f"{frame_count/video_fps:.1f}s",
                        "channel": "M4-B", "type": pred,
                        "detail": f"conf={conf_val:.2f}",
                    })
            elif m4b_is_threat:
                txt = f"M4-B: {pred} ({conf_val:.0%}) verifying {m5.scene_counter}/{m5.scene_threshold}"
                cv2.rectangle(display, (10,10), (10+len(txt)*9, 42), (0,140,255), -1)
                cv2.putText(display, txt, (16,33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            else:
                cv2.rectangle(display, (10,10), (200,38), (0,120,0), -1)
                cv2.putText(display, f"M4-B: Normal ({conf_val:.0%})",
                            (16,30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)

        m4b_confirmed_prev = m4b_confirmed

        # Frame info
        proc_ms = (time.time() - t0) * 1000
        fps_now = 1000.0 / max(proc_ms, 1)
        cv2.putText(display, f"F{frame_count} | {fps_now:.0f}fps | {proc_ms:.0f}ms",
                    (frame_w-220, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)

        # ── Update UI ─────────────────────────────────────────────────────────
        frame_placeholder.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                                channels="RGB", width=output_width)

        if progress_bar and total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

        # M4-A panel
        n_zone = sum(1 for t in tracked_objects if loiterer.is_in_zone(tracked_objects[t]))
        lines = [f"In zone: **{n_zone}** / {len(tracked_objects)}"]
        if m4a_suspicious:
            for tid in m4a_suspicious:
                s = loiterer.get_dwell_seconds(tid)
                c = m5.track_counters.get(tid, 0)
                tag = "🔴 CONFIRMED" if c >= persistence else f"🟡 verifying ({c}/{persistence})"
                lines.append(f"ID {tid}: {s:.1f}s — {tag}")
        else:
            lines.append("🟢 No loitering")
        m4a_display.markdown("\n\n".join(lines))

        # M4-B panel
        if m4b_result:
            p = m4b_result["probabilities"]
            pred = m4b_result["prediction"]
            if m4b_confirmed:
                m4b_display.markdown(f"🔴 **{pred}** CONFIRMED\n\nV={p['Violence']:.0%} T={p['Threat']:.0%} N={p['Normal']:.0%}")
            elif m4b_is_threat:
                m4b_display.markdown(f"🟡 **{pred}** verifying ({m5.scene_counter}/{m5.scene_threshold})\n\nV={p['Violence']:.0%} T={p['Threat']:.0%} N={p['Normal']:.0%}")
            else:
                m4b_display.markdown(f"🟢 **Normal**\n\nV={p['Violence']:.0%} T={p['Threat']:.0%} N={p['Normal']:.0%}")
        elif not m4b_enabled:
            m4b_display.markdown("⚪ Disabled")
        elif classifier:
            m4b_display.markdown(f"⏳ Buffering poses ({classifier.buffer_fill}/{SKEL_CLIP_FRAMES})")

        # Info panel
        info_display.markdown(
            f"**Frame:** {frame_count}" + (f" / {total_frames}" if total_frames else "") +
            f"\n\n**Detections:** {len(detections)}"
            f"\n\n**Tracked:** {len(tracked_objects)}"
            f"\n\n**Device:** {device.upper()}"
            f"\n\n**Speed:** {proc_ms:.0f}ms/frame"
        )

        # Alert log
        if alert_log:
            alert_header.markdown(f"**Alert Log** ({len(alert_log)} total)")
            recent = alert_log[-8:]
            rows = ""
            for a in reversed(recent):
                e = "🔴" if a["channel"] == "M4-A" else "🟠"
                rows += f"| {e} {a['time']} | {a['channel']} | {a['type']} | {a['detail']} |\n"
            alert_display.markdown("| Time | Channel | Type | Detail |\n|---|---|---|---|\n" + rows)

    cap.release()
    if progress_bar:
        progress_bar.progress(1.0)
    st.session_state.stop_requested = False

    # ── Summary ───────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #0f766e 100%);
                padding: 0.8rem 1.2rem; border-radius: 10px; margin-bottom: 0.8rem;">
        <div style="color: #fff; font-size: 1.1rem; font-weight: 700;">📊 Session Summary</div>
        <div style="color: rgba(255,255,255,0.75); font-size: 0.8rem;">Pipeline run statistics</div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📹 Frames", frame_count)
    c2.metric("🚨 Total Alerts", len(alert_log))
    c3.metric("🟠 M4-A", len([a for a in alert_log if a["channel"]=="M4-A"]))
    c4.metric("🔵 M4-B", len([a for a in alert_log if a["channel"]=="M4-B"]))
    c5.metric("🎬 Video", video_name[:18])

    if alert_log:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Full Alert Log</div>', unsafe_allow_html=True)
        st.dataframe([{k:v for k,v in a.items() if k!="id"} for a in alert_log],
                     use_container_width=True)
    else:
        st.success("✅ No alerts triggered during this session.")

elif not run_clicked:
    st.markdown("""
    <div style="background: #161b24; border: 1px dashed #334155; border-radius: 12px;
                padding: 2rem; text-align: center; margin-top: 1rem;">
        <div style="font-size: 2.4rem; margin-bottom: 0.4rem;">🎯</div>
        <div style="font-size: 1.05rem; color: #f1f5f9; font-weight: 600;">Ready to begin</div>
        <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 0.4rem;">
            Select a video source above, configure modules in the sidebar,
            then click <b>Run Pipeline</b> to start.
        </div>
    </div>
    """, unsafe_allow_html=True)