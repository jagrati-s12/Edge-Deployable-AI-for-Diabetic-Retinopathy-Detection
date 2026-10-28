import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from dr_model import DREfficientNetB4, OrdinalAwareLoss, preprocess_image
from sklearn.model_selection import train_test_split

class MultiSiteDRDataset(Dataset):
    """
    Unified dataset loader for multi-site fundus images.
    Supports two dataset structures:
    - set1: CSV mapping (id_code, diagnosis) + images in train_images/
    - set2: CSV mapping (Image name, Retinopathy grade) + images in structured directories
    """
    def __init__(self, data_dir, transform=None, target_resolution=512):
        self.samples = []
        self.transform = transform
        self.target_resolution = target_resolution

        data_dir = os.path.abspath(data_dir)

        # Detect dataset structure by checking for known files
        if os.path.exists(os.path.join(data_dir, 'train.csv')):
            # set1 structure
            print(f"Loading set1 dataset from {data_dir}")
            train_csv = os.path.join(data_dir, 'train.csv')
            train_img_dir = os.path.join(data_dir, 'train_images')

            if not os.path.exists(train_img_dir):
                raise FileNotFoundError(f"Train image directory not found: {train_img_dir}")

            df = pd.read_csv(train_csv)
            for _, row in df.iterrows():
                img_name = f"{row['id_code']}.png"
                img_path = os.path.join(train_img_dir, img_name)
                if os.path.exists(img_path):
                    self.samples.append((img_path, int(row['diagnosis']), 'set1'))
                else:
                    print(f"Warning: Image not found for {img_name}")

        elif os.path.exists(os.path.join(data_dir, 'B. Disease Grading', '2. Groundtruths', 'a. IDRiD_Disease Grading_Training Labels.csv')):
            # set2 structure
            print(f"Loading set2 (IDRiD) dataset from {data_dir}")
            # Use set2's 2. Disease Grading structure as primary focus
            base_dir = os.path.join(data_dir, 'B. Disease Grading', '1. Original Images', 'a. Training Set')
            labels_csv = os.path.join(data_dir, 'B. Disease Grading', '2. Groundtruths', 'a. IDRiD_Disease Grading_Training Labels.csv')

            if not os.path.exists(base_dir) or not os.path.exists(labels_csv):
                raise FileNotFoundError(f"Missing set2 training data or labels")

            df = pd.read_csv(labels_csv)
            for _, row in df.iterrows():
                img_name = f"{row['Image name']}.jpg"
                img_path = os.path.join(base_dir, img_name)
                if os.path.exists(img_path):
                    self.samples.append((img_path, int(row['Retinopathy grade']), 'set2'))
                else:
                    print(f"Warning: Image not found for {img_name}")
        else:
            raise ValueError(f"Unknown dataset structure in {data_dir}")

        if len(self.samples) == 0:
            raise ValueError(f"No samples loaded from {data_dir}")

        print(f"Dataset loaded: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, site_id = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error loading image {img_path}: {e}")

        # Apply preprocessing pipeline from dr_model.py to maintain consistency
        tensor_img, visual_img = preprocess_image(img_path)

        # For training, apply data augmentations
        if self.transform:
            image = self.transform(image)

        return tensor_img, torch.tensor(label, dtype=torch.long)

def get_train_transform(size=512):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def compute_class_weights(dataset, num_classes=5):
    labels = [s[1] for s in dataset.samples]
    label_array = np.array(labels)
    class_counts = np.bincount(label_array, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights * (len(class_weights) / class_weights.sum())
    return torch.FloatTensor(class_weights)

def train_multi_site_model(data_dir, epochs=10, batch_size=16, lr=1e-4, save_path='weights_int8.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Multi-Site DR Training on Device: {device} ===")

    # Load unified dataset
    train_dataset = MultiSiteDRDataset(data_dir, transform=get_train_transform())

    # Compute class weights for balanced sampling
    class_weights = compute_class_weights(train_dataset)
    print(f"Class weights: {class_weights}")

    # Create weighted sampler to handle class imbalance
    sampler = WeightedRandomSampler(
        weights=class_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2
    )

    # Initialize model with OrdinalAwareLoss
    model = DREfficientNetB4(num_classes=5).to(device)
    criterion = OrdinalAwareLoss(num_classes=5, label_smoothing=0.1, ordinal_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    model.train()
    print(f"\nTraining started for {epochs} epochs...")

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        scheduler.step(epoch_loss)

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_weights': class_weights,
        'loss_function': 'OrdinalAwareLoss',
        'dataset_path': data_dir,
    }, save_path)

    print(f"\nModel saved to {save_path}")
    print(f"Class weights saved for inference on unseen sites")

if __name__ == "__main__":
    import sys
    # Default to local data directory
    data_path = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\KIIT\Documents\FreeClaude\res\data'

    if not os.path.exists(data_path):
        print(f"Error: Data directory not found: {data_path}")
        print("Available directories:")
        base_dir = r'C:\Users\KIIT\Documents\FreeClaude\res'
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                print(f"  - {item}")
        sys.exit(1)

    train_multi_site_model(data_path)
