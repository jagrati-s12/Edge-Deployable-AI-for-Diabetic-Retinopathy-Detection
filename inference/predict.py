import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import preprocess_image
from models.model import get_model

CLASSES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

def predict(image_path, model_path=None, model_name='mobilenetv3_small_100'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(model_name=model_name, num_classes=5, pretrained=False)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess
    processed_img = preprocess_image(image_path)

    # To tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_img = transform(processed_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

    print(f"Prediction Result for: {image_path}")
    print(f"Grade: {predicted_class} ({CLASSES[predicted_class]})")
    print(f"Confidence: {confidence:.4f}")
    print("\nAll Probabilities:")
    for idx, prob in enumerate(probabilities):
        print(f"  {CLASSES[idx]}: {prob:.4f}")

    return predicted_class, confidence, probabilities

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Edge-Deployable DR Detection Inference")
    parser.add_argument('--image', type=str, required=True, help='Path to fundus image')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained weights (.pth)')
    parser.add_argument('--model_name', type=str, default='mobilenetv3_small_100', help='Backbone model name')

    args = parser.parse_args()
    predict(args.image, args.model_path, args.model_name)
