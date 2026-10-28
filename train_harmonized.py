import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from dr_model import DREfficientNetB4, OrdinalAwareLoss
from sklearn.metrics import cohen_kappa_score

class ProcessedDRDataset(Dataset):
    def __init__(self, data_dirs, transform=None):
        self.samples = []
        self.transform = transform
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        for d_dir in data_dirs:
            d_path = os.path.abspath(d_dir)
            if not os.path.exists(d_path):
                continue
            for grade in range(5):
                grade_dir = os.path.join(d_path, str(grade))
                if os.path.exists(grade_dir):
                    for img_name in os.listdir(grade_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(grade_dir, img_name), grade))
        print(f"Loaded {len(self.samples)} samples from {data_dirs}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, torch.tensor(label, dtype=torch.long)

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)), # Smaller for fast CPU training demonstration
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train_harmonized_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    dataset = ProcessedDRDataset(['processed_data/set1', 'processed_data/set2'], transform=get_train_transform())
    if len(dataset) == 0:
        print("No training images found!")
        return

    # Use a small subset for fast verification (e.g., 200 samples)
    from torch.utils.data import Subset
    indices = np.random.choice(len(dataset), size=min(200, len(dataset)), replace=False)
    subset = Subset(dataset, indices)

    train_size = int(0.8 * len(subset))
    val_size = len(subset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(subset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = DREfficientNetB4(num_classes=5).to(device)
    criterion = OrdinalAwareLoss(num_classes=5, label_smoothing=0.1, ordinal_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(1): # 1 epoch test run
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            print(f"Batch {i+1}/{len(train_loader)} Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch 1 Complete | Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), 'best_harmonized_dr_model.pth')
    print("Saved test model checkpoint successfully.")

if __name__ == '__main__':
    train_harmonized_model()
