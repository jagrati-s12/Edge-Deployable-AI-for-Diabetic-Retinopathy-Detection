import torch
from dr_model import DREfficientNetB4, GradCAM, generate_heatmap_overlay, preprocess_image
import cv2
import numpy as np

def test_inference_pipeline():
    device = torch.device('cpu')
    model = DREfficientNetB4(num_classes=5)

    # Load our trained harmonized weights
    if os.path.exists('best_harmonized_dr_model.pth'):
        model.load_state_dict(torch.load('best_harmonized_dr_model.pth', map_location=device))
        print("Loaded harmonized model weights successfully.")
    else:
        print("Weights not found, using initialized model.")

    model.to(device)
    model.eval()

    # Pick a sample image from processed_data/set1/0/
    sample_dir = 'processed_data/set1/0'
    if os.path.exists(sample_dir):
        files = os.listdir(sample_dir)
        if files:
            img_path = os.path.join(sample_dir, files[0])
            print(f"Testing inference and Grad-CAM on: {img_path}")

            tensor_img, raw_rgb = preprocess_image(img_path)
            tensor_img = tensor_img.to(device)

            grad_cam = GradCAM(model, model.get_target_layer())
            heatmap, pred_class, probs = grad_cam(tensor_img)

            print(f"Predicted DR Grade: {pred_class}")
            print(f"Probabilities: {probs}")
            print("Inference and Grad-CAM test passed successfully!")

if __name__ == '__main__':
    import os
    test_inference_pipeline()
