from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import torch
import cv2
import numpy as np
import base64
import os
import tempfile
from dr_model import DREfficientNetB4, preprocess_image, GradCAM, generate_heatmap_overlay

app = FastAPI(title="Diabetic Retinopathy Analyzer",
              description="Edge-Deployable AI System for DR Severity Grading",
              version="1.0.0")

# Load model (mock loading for now, but ready for quantized INT8 deployment)
device = torch.device('cpu')
model = DREfficientNetB4(num_classes=5)
# In production: model.load_state_dict(torch.load('weights_int8.pth'))
model.to(device)
model.eval()

# Initialize Grad-CAM
grad_cam = GradCAM(model, model.get_target_layer())

def encode_image(img_array):
    """Encodes an OpenCV image to base64 for API transmission"""
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict_dr(file: UploadFile = File(...)):
    """
    Analyzes a fundus image and returns ICDR severity grade (0-4) + Grad-CAM heatmap.
    """
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            temp.write(await file.read())
            temp_path = temp.name

        # 1. Preprocess
        tensor_img, raw_rgb_img = preprocess_image(temp_path)
        tensor_img = tensor_img.to(device)

        # 2. Predict & Generate Grad-CAM heatmap
        heatmap, pred_class, probs = grad_cam(tensor_img)

        # 3. Create visual overlay
        overlayed_img = generate_heatmap_overlay(raw_rgb_img, heatmap)

        confidence = float(probs[pred_class])

        os.remove(temp_path)

        return JSONResponse({
            "status": "success",
            "icdr_grade": pred_class,
            "confidence_score": round(confidence, 4),
            "probabilities": probs.tolist(),
            "heatmap_base64": encode_image(overlayed_img),
            # Hardware simulation tag
            "device": "raspberry_pi_4_int8_simulated",
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    print("Starting DR API Service (Simulating Edge deployment on RPi)...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
