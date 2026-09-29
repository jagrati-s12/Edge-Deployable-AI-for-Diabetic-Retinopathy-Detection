# An Explainable, Edge-Deployable AI System for Automated Diabetic Retinopathy Severity Grading

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning system designed for 5-level **Diabetic Retinopathy (DR)** severity grading based on the **ICDR scale**. Engineered specifically for deployment on resource-constrained edge devices (such as the **Raspberry Pi 4**), the system combines high-accuracy feature extraction with lightweight architecture, adaptive preprocessing for low-cost fundus cameras, ordinal-aware optimization, and real-time visual explanations via **Grad-CAM**.

---

## 🌟 Key Features

* **Attention-Enhanced Backbone**: Utilizes **EfficientNet-B4** augmented with native **Squeeze-and-Excitation (SE)** attention blocks for precise pathology localization (hemorrhages, exudates, microaneurysms).
* **Robust Preprocessing Pipeline**: Built to handle low-quality, unevenly lit images from affordable portable cameras:
  * Automatic dark border fundus cropping.
  * Green channel extraction (enhances microvascular pathology visibility).
  * Adaptive **CLAHE** (Contrast Limited Adaptive Histogram Equalization).
* **Ordinal-Aware Loss Function**: Penalizes distant grade errors more heavily than adjacent class misclassifications, matching clinical reality. Includes label smoothing for calibrated confidence scoring.
* **Explainable AI (XAI)**: Generates real-time **Grad-CAM** visual heatmap overlays to give clinicians interpretable insight into the model's decision-making process.
* **Edge-Optimized & Offline Ready**:
  * Quantized INT8 deployment footprint (~19 MB model size).
  * Sub-500ms inference latency on standard CPU / ARM edge hardware.
  * Operates completely offline for rural and remote screening clinics.
* **FastAPI Backend**: Asynchronous REST API serving predictions, confidence scores, class probabilities, and base64-encoded Grad-CAM heatmaps.

---

## 📊 ICDR Severity Grading Scale

The model classifies fundus images into five grades according to the **International Clinical Diabetic Retinopathy (ICDR)** scale:

| Grade | Disease Severity Level | Clinical Findings |
| :---: | :--- | :--- |
| **0** | No DR | No microaneurysms or retinal abnormalities |
| **1** | Mild NPDR | Microaneurysms only |
| **2** | Moderate NPDR | More than just microaneurysms, but less than severe NPDR |
| **3** | Severe NPDR | Any of: >20 intraretinal hemorrhages in 4 quadrants, venous beading in 2+ quadrants, or IRMA in 1+ quadrant |
| **4** | Proliferative DR (PDR) | Neovascularization and/or vitreous/preretinal hemorrhage |

---

## 📁 Repository Structure

```text
Edge-Deployable-AI-for-Diabetic-Retinopathy-Detection/
├── dr_model.py     # Core model architecture (EfficientNet-B4 + SE), Ordinal Loss, Preprocessing & Grad-CAM engine
├── api.py          # FastAPI service backend for edge deployment & API serving
└── README.md       # Project documentation
```

---

## 🚀 Quick Start

### Prerequisites

Ensure you have Python 3.8+ installed along with PyTorch, OpenCV, and FastAPI dependencies.

### 1. Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/jagrati-s12/Edge-Deployable-AI-for-Diabetic-Retinopathy-Detection.git
cd Edge-Deployable-AI-for-Diabetic-Retinopathy-Detection

pip install torch torchvision opencv-python numpy fastapi uvicorn
```

### 2. Start the API Server

Run the FastAPI server locally:

```bash
python api.py
```

The API server will start at `http://0.0.0.0:8000`. You can explore interactive API documentation at `http://localhost:8000/docs`.

### 3. Send a Prediction Request

You can send a fundus image to the `/predict` endpoint using `curl` or Python:

#### Using `curl`:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/fundus_image.jpg"
```

#### Sample API Response:

```json
{
  "status": "success",
  "icdr_grade": 2,
  "confidence_score": 0.9142,
  "probabilities": [0.012, 0.054, 0.9142, 0.018, 0.0018],
  "heatmap_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "device": "raspberry_pi_4_int8_simulated"
}
```

---

## 🛠️ System Architecture

```text
+-----------------------+
|  Fundus Image Upload  |
+-----------+-----------+
            |
            v
+-----------------------+   1. Auto-crop dark borders
| Preprocessing Pipeline|   2. Extract Green Channel & Apply CLAHE
+-----------+-----------+   3. Resize (512x512) & Normalize
            |
            v
+-----------------------+   - EfficientNet-B4 Backbone
|  EfficientNet-B4 + SE |   - Squeeze-and-Excitation Attention
+-----------+-----------+   - Custom 3-Layer Dense Head
            |
      +-----+-----+
      |           |
      v           v
+-----------+ +-----------+
| Grade 0-4 | | Grad-CAM  |  --> Heatmap Overlay
| Prediction| | Visualizer|      Generation
+-----------+ +-----------+
```

---

## ⚡ Edge Hardware Deployment (Raspberry Pi 4)

To deploy on edge devices like the Raspberry Pi 4:
1. Export model weights to **PyTorch INT8 Quantized Format** (`torch.quantization.quantize_dynamic`).
2. Run `api.py` with `uvicorn` using `Gunicorn` or directly in a lightweight Docker container.
3. The INT8 model reduces memory footprint from **75 MB to ~19 MB** with sub-500ms response times.

---

## 📄 License

This project is licensed under the MIT License.
