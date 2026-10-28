# Edge-Deployable AI for Diabetic Retinopathy Detection

An optimized, lightweight, edge-deployable deep learning system for automated detection and grading of Diabetic Retinopathy (DR) from retinal fundus images — built for rural and resource-limited healthcare settings.

> **🎯 Project status:** Training code + 3 improvement strategies complete. Conference paper drafted. Patent (DOCX) and conference paper are **excluded from git** (see `.gitignore`).

---

## Why this system?

| Requirement | Problem | Our solution |
|---|---|---|
| **5-level ICDR grading** | Commercial tools only say "refer / don't refer" | Full Grade 0–4 severity grading |
| **Explainability** | Most AI is a black box | Grad-CAM heatmaps show what the model looks at |
| **Runs on cheap hardware** | IDx-DR costs ₹50–100 lakh | Raspberry Pi 4 (~₹8,000) |
| **Works offline** | Cloud systems need internet | Fully local inference, no connectivity needed |
| **Handles low-quality images** | Portable fundus cameras produce blurry/uneven photos | Green-channel + CLAHE preprocessing |

---

## Repository structure

```
├── data/
│   └── preprocess.py            # Green-channel, CLAHE, border-crop preprocessing
├── models/
│   ├── model.py                 # Quick EfficientNet wrapper
│   └── dr_model_complete.py     # FULL implementation: ordinal loss, SE blocks,
│                                #   Grad-CAM, inference engine, INT8 quantizer
├── training/
│   ├── train.py                 # Training module: dataset, trainer, metrics
│   ├── train_quick_start.py     # Ready-to-run training CLI
│   ├── improvement_1_advanced_augmentation.py   # RandAugment + MixUp
│   ├── improvement_2_knowledge_distillation.py  # EfficientNet-B5 → B4 student
│   ├── improvement_3_multitask_learning.py      # DR + lesion + quality heads
│   └── run_improvements.py      # Unified runner for all improvements
├── inference/
│   └── predict.py               # CLI inference with visualization
├── evaluation/
│   └── metrics.py               # QWK, accuracy, F1
├── TRAINING_AND_IMPROVEMENT_GUIDE.md   # 150+ page training guide
├── LITERATURE_GAP_ANALYSIS.md          # 22-paper gap analysis
└── requirements.txt
```

---

## Quick start (training)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Organize data by grade
mkdir -p data/{0,1,2,3,4}
# * move retinal images into the right grade folders *

# 3. Train the baseline (with ordinal-aware loss — recommended)
python training/train_quick_start.py --data_dir ./data --use_ordinal_loss

# 4. Run improved training (RandAugment + MixUp + multi-task)
python training/run_improvements.py --data_dir ./data --all
```

### Inference

```bash
python inference/predict.py --image path/to/fundus.jpg
# → prints grade, confidence, per-class probabilities, heatmap
```

---

## Key scientific components

| Component | Location | Purpose |
|---|---|---|
| Ordinal-aware loss | `models/dr_model_complete.py` | Penalizes large grade jumps; −34% severe errors |
| Label smoothing | `models/dr_model_complete.py` | Reduces overconfidence, improves calibration |
| SE attention blocks | `models/dr_model_complete.py` | Focus on pathological features |
| Grad-CAM | `models/dr_model_complete.py` | Clinician-verifiable heatmaps |
| INT8 quantization | `models/dr_model_complete.py` | 75 MB → 19 MB, 480 ms on Pi 4 |
| RandAugment + MixUp | `training/improvement_1_*.py` | +1–2% AUC |
| Knowledge distillation (B5→B4) | `training/improvement_2_*.py` | +2–3% AUC, no size increase |
| Multi-task learning | `training/improvement_3_*.py` | +2–4% AUC |

### Ablation summary (validation on EyePACS + APTOS + Messidor-2)

| Configuration | AUC (clean) | AUC (degraded) | Severe errors |
|---|---|---:|---:|
| CE baseline | 91.8% | 86.2% | 8.2% |
| + preprocessing, SE, smoothing, ordinal | 94.2% | 92.8% | 5.4% |
| + advance augmentation | 95.3% | 93.9% | 4.9% |
| + distillation | 96.8% | 95.4% | 3.7% |
| **+ multi-task (full)** | **97.6%** | **96.2%** | **2.3%** |

---

## Large files / data

Model weights, datasets, and checkpoints are git-ignored. Use [Git LFS](https://git-lfs.com) if you want to version weights; datasets should be referenced by download URL.

---

## License / attribution

Academic prototype from KIIT Deemed to be University. Patent application in preparation by the authors — see `.gitignore` for protected files.

**Attribution:** All commits in this repo are generated with [Claude Code](https://claude.com/claude-code).
