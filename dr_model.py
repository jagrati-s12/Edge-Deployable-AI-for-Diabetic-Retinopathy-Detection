import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ----------------------------------------------------
# 1. Image Preprocessing Pipeline
# ----------------------------------------------------
def crop_image_from_gray(img, tol=7):
    """
    Automatically crops the fundus region to remove black borders.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3], axis=-1)
        return img

def preprocess_image(image_path, size=512):
    """
    Prepares the fundus image capturing:
    1. Border cropping
    2. Green channel extraction and CLAHE enhancement
    3. Resizing and normalization
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Auto crop fundus region
    img = crop_image_from_gray(img)

    # Extract green channel for microaneurysm/hemorrhage visibility
    green = img[:, :, 1]

    # Adaptive CLAHE Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_clahe = clahe.apply(green)

    # Reintegrate enhanced green channel
    img[:, :, 1] = green_clahe

    # Resize
    img = cv2.resize(img, (size, size))

    # Keep an unnormalized RGB copy for Grad-CAM overlay
    visual_img = img.copy()

    # Normalize to [0, 1] for the model
    img = img.astype(np.float32) / 255.0

    # ImageNet normalization standard for pre-trained weights
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # Channels first: (H, W, C) -> (C, H, W)
    img = np.transpose(img, (2, 0, 1))

    return torch.tensor(img).unsqueeze(0), visual_img

# ----------------------------------------------------
# 2. Attention-Enhanced Model Architecture
# ----------------------------------------------------
class DREfficientNetB4(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.4):
        super(DREfficientNetB4, self).__init__()

        # Load pretrained EfficientNet-B4 (contains Squeeze-and-Excitation blocks natively)
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        self.base_model = models.efficientnet_b4(weights=weights)

        in_features = self.base_model.classifier[1].in_features

        # Custom 3 dense layer classification head with dropout and label smoothing readiness
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

    def get_target_layer(self):
        """Returns the final convolutional layer for Grad-CAM"""
        return self.base_model.features[-1]

# ----------------------------------------------------
# 3. Ordinal-Aware Loss Function
# ----------------------------------------------------
class OrdinalAwareLoss(nn.Module):
    def __init__(self, num_classes=5, label_smoothing=0.1, ordinal_weight=1.0):
        super(OrdinalAwareLoss, self).__init__()
        self.num_classes = num_classes
        self.ordinal_weight = ordinal_weight
        # Cross Entropy with built-in label smoothing (calibrated confidence)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)

        probs = F.softmax(logits, dim=1)
        class_indices = torch.arange(self.num_classes, device=logits.device, dtype=torch.float32)

        # Calculate expected prediction (soft ordinal prediction)
        expected_pred = torch.sum(probs * class_indices, dim=1)
        true_labels = targets.float()

        # Penalize further distances more heavily (e.g. predicting grade 4 for grade 0 patient)
        ordinal_penalty = torch.mean((expected_pred - true_labels) ** 2)

        return ce_loss + (self.ordinal_weight * ordinal_penalty)

# ----------------------------------------------------
# 4. Explainability (Grad-CAM)
# ----------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, class_idx]
        # Retain graph needed for backward
        score.backward(retain_graph=True)

        # Global average pooling on gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weight the activation maps
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.cpu().detach().numpy(), class_idx, F.softmax(logits, dim=1)[0].cpu().detach().numpy()

def generate_heatmap_overlay(img_rgb, cam_heatmap):
    heatmap = cv2.resize(cam_heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    superimposed_img = float(0.5) * heatmap + float(0.5) * img_rgb
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

if __name__ == "__main__":
    print("Diabetic Retinopathy Pipeline Initialized.")
    print("Model: EfficientNet-B4 + Squeeze & Excitation")
    print("Optimization: Ordinal-Aware Loss + Label Smoothing")
    print("XAI: Grad-CAM")
