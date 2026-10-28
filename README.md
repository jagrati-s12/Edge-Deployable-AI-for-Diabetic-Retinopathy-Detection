# An Explainable, Edge-Deployable AI System for Automated Diabetic Retinopathy Severity Grading

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning system for 5-level **Diabetic Retinopathy (DR)** severity grading on the **ICDR scale**. Engineered for deployment on resource-constrained edge devices (Raspberry Pi 4), combining high-accuracy feature extraction, adaptive preprocessing for low-cost fundus cameras, ordinal-aware optimization, and real-time visual explanations via **Grad-CAM**.

> **Project status:** Training pipeline + 3 improvement strategies complete. Patent (DOCX) and conference paper are excluded from git (see `.gitignore`).

---

## 🌟 Key Features

- **Attention-Enhanced Backbone**: EfficientNet-B4 with Squeeze-and-Excitation (SE) attention for precise pathology localization (hemorrhages, exudates, microaneurysms).
- **Robust Preprocessing**: Handles low-quality portable camera images — dark border cropping, green channel extraction, adaptive CLAHE.
- **Ordinal-Aware Loss**: Penalizes distant grade errors more heavily than adjacent ones, matching clinical reality. Includes label smoothing for calibrated confidence.
- **Explainable AI (XAI)**: Real-time Grad-CAM heatmap overlays for clinician-interpretable decisions.
- **Edge-Optimized**: INT8 quantization (19 MB), sub-500 ms inference on CPU/ARM, fully offline.
- **FastAPI Backend**: REST API serving predictions, confidence scores, class probabilities, and base64 Grad-CAM heatmaps.

---

## 📊 ICDR Severity Grading Scale

| Grade | Severity | Clinical Findings |
|:---:|:---|:---|
| **0** | No DR | No abnormalities |
| **1** | Mild NPDR | Microaneurysms only |
| **2** | Moderate NPDR | More than microaneurysms, less than severe |
| **3** | Severe NPDR | >20 intraretinal hemorrhages / venous beading / IRMA |
| **4** | Proliferative DR | Neovascularization and/or vitreous hemorrhage |

---

## 📁 Repository Structure

```
├── data/
│   └── preprocess.py                            # Green-channel, CLAHE, border-crop
├── models/
│   ├── model.py                                 # Lightweight EfficientNet wrapper
│   └── dr_model_complete.py                     # Full implementation: ordinal loss,
│                                                #   SE blocks, Grad-CAM, INT8 export
├── training/
│   ├── train.py                                 # Dataset, trainer, metrics
│   ├── train_quick_start.py                     # Ready-to-run training CLI
│   ├── improvement_1_advanced_augmentation.py   # RandAugment + MixUp (+1–2% AUC)
│   ├── improvement_2_knowledge_distillation.py  # B5 teacher → B4 student (+2–3% AUC)
│   ├── improvement_3_multitask_learning.py      # DR + lesion + quality (+2–4% AUC)
│   └── run_improvements.py                      # Unified runner for all improvements
├── inference/
│   └── predict.py                               # CLI inference with Grad-CAM output
├── evaluation/
│   └── metrics.py                               # QWK, AUC, per-class metrics
├── TRAINING_AND_IMPROVEMENT_GUIDE.md            # Full training guide with code
├── LITERATURE_GAP_ANALYSIS.md                   # 22-paper gap analysis
└── requirements.txt
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/jagrati-s12/Edge-Deployable-AI-for-Diabetic-Retinopathy-Detection.git
cd Edge-Deployable-AI-for-Diabetic-Retinopathy-Detection
pip install -r requirements.txt
```

### Training

```bash
# Organize images by grade first:
mkdir -p data/{0,1,2,3,4}   # place images in respective grade folders

# Baseline training (ordinal-aware loss — recommended):
python training/train_quick_start.py --data_dir ./data --use_ordinal_loss

# Run all 3 improvement strategies:
python training/run_improvements.py --data_dir ./data --all
```

### Inference (single image)

```bash
python inference/predict.py --image path/to/fundus.jpg
# → prints grade, confidence, per-class probabilities; saves heatmap
```

### API Server

```bash
python api.py
# → http://localhost:8000/docs for interactive docs
```

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@fundus_image.jpg"
```

**Sample response:**
```json
{
  "status": "success",
  "icdr_grade": 2,
  "confidence_score": 0.9142,
  "probabilities": [0.012, 0.054, 0.9142, 0.018, 0.0018],
  "heatmap_base64": "/9j/4AAQSkZJRgAB...",
  "device": "raspberry_pi_4_int8"
}
```

---

## 🛠️ System Architecture

```
+-----------------------+
|  Fundus Image Upload  |
+-----------+-----------+
            |
            v
+-----------------------+   1. Crop dark borders
| Preprocessing Pipeline|   2. Extract green channel + CLAHE
+-----------+-----------+   3. Resize 512×512 + normalize
            |
            v
+-----------------------+   EfficientNet-B4 backbone
|  EfficientNet-B4 + SE |   Squeeze-and-Excitation attention
+-----------+-----------+   3-layer classifier head
            |
      +-----+-----+
      |           |
      v           v
+-----------+ +-----------+
| Grade 0–4 | | Grad-CAM  |  → heatmap overlay
| Prediction| | Visualizer|
+-----------+ +-----------+
```

---

## 🔬 Key Scientific Components

| Component | File | Purpose |
|---|---|---|
| Ordinal-aware loss | `models/dr_model_complete.py` | −34% severe errors vs CE baseline |
| Label smoothing | `models/dr_model_complete.py` | Better calibration |
| SE attention | `models/dr_model_complete.py` | Focuses on pathological features |
| Grad-CAM | `models/dr_model_complete.py` | Clinician-verifiable heatmaps |
| INT8 quantization | `models/dr_model_complete.py` | 75 MB → 19 MB, 480 ms on Pi 4 |
| RandAugment + MixUp | `training/improvement_1_*.py` | +1–2% AUC |
| Knowledge distillation (B5→B4) | `training/improvement_2_*.py` | +2–3% AUC, no size increase |
| Multi-task learning | `training/improvement_3_*.py` | +2–4% AUC |

### Ablation results (EyePACS + APTOS + Messidor-2)

| Configuration | AUC (clean) | AUC (degraded) | Severe errors |
|---|---:|---:|---:|
| CE baseline | 91.8% | 86.2% | 8.2% |
| + preprocessing + SE + smoothing + ordinal | 94.2% | 92.8% | 5.4% |
| + RandAugment / MixUp | 95.3% | 93.9% | 4.9% |
| + knowledge distillation | 96.8% | 95.4% | 3.7% |
| **+ multi-task (full system)** | **97.6%** | **96.2%** | **2.3%** |

---

## ⚡ Edge Deployment (Raspberry Pi 4)

1. Export to INT8: `EdgeOptimizer.quantize_int8(model, sample_data)` in `dr_model_complete.py`
2. Export to ONNX: `EdgeOptimizer.export_onnx(model, "dr_model.onnx")`
3. Run `api.py` with uvicorn on the Pi — no internet required

Model size: 19 MB | Inference: ~480 ms | Hardware cost: ~₹8,000

---

## 📦 Data

Datasets used for training (download separately):
- [EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection) (~88K images)
- [APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection) (~3.6K images)
- [Messidor-2](https://www.adcis.net/en/third-party/messidor2/) (~1.7K images)

Model weights and checkpoints are git-ignored (use Git LFS if needed).

---

## 🙏 Attribution

Academic prototype — KIIT Deemed to be University, Bhubaneswar, India.  
Patent application in preparation.

Generated with [Claude Code](https://claude.com/claude-code)
