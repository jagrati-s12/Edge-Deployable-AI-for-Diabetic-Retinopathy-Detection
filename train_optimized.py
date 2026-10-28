"""
Optimized DR Model Training - Combines baseline + advanced augmentation
Targets: 75-80% accuracy baseline, +1-2% with augmentation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
import numpy as np
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))

from training.train import (
    DRDataset,
    get_augmentation_transforms,
    get_class_weights,
    EarlyStopping,
    MetricsTracker,
    Trainer
)
from dr_model import DREfficientNetB4
from models.dr_model_complete import OrdinalAwareLoss

def train_model(
    data_dir='processed_data/set1',
    epochs=50,
    batch_size=16,
    learning_rate=1e-3,
    use_augmentation=True,
    checkpoint_path='dr_model_best.pth'
):
    """
    Main training function.

    Args:
        data_dir: Path to preprocessed data
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        use_augmentation: Use advanced augmentation
        checkpoint_path: Where to save best model
    """

    print("\n" + "="*80)
    print("DR MODEL TRAINING - OPTIMIZED PIPELINE")
    print("="*80)
    print(f"Data: {data_dir}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {learning_rate}")
    print(f"Augmentation: {'Advanced (RandAug+MixUp)' if use_augmentation else 'Standard'}")
    print("="*80 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("[STEP 1] Loading data...")
    dataset = DRDataset(data_dir, augment=False)
    print(f"  Total images: {len(dataset)}\n")

    # 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Apply transforms
    train_dataset.dataset.transform = get_augmentation_transforms(augment=True)
    val_dataset.dataset.transform = get_augmentation_transforms(augment=False)

    # Class weights for imbalance
    train_labels = [dataset.labels[i] for i in train_dataset.indices]
    weights = get_class_weights(dataset)
    class_weights = weights / weights.sum() * len(weights)
    sample_weights = class_weights[[dataset.labels[i] for i in train_dataset.indices]]

    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"  Batches per epoch: {len(train_loader)}\n")

    # ========================================================================
    # MODEL & OPTIMIZER
    # ========================================================================
    print("[STEP 2] Creating model...")
    model = DREfficientNetB4(num_classes=5).to(device)
    criterion = OrdinalAwareLoss(num_classes=5, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    early_stop = EarlyStopping(patience=7, min_delta=1e-4, model_path=checkpoint_path)
    metrics = MetricsTracker()

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model size: {sum(p.numel() for p in model.parameters()) * 4 / 1e6:.1f} MB\n")

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    print("[STEP 3] Training...\n")
    start_time = datetime.now()

    for epoch in range(1, epochs + 1):
        # Train phase
        model.train()
        train_loss, train_preds, train_labels_list = 0, [], []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())

            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                print(f"  Epoch {epoch}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_loss /= len(train_dataset)
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels_list)) * 100

        # Validation phase
        model.eval()
        val_loss, val_preds, val_labels_list = 0, [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_loss /= len(val_dataset)
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels_list)) * 100

        # QWK metric
        from sklearn.metrics import cohen_kappa_score
        val_qwk = cohen_kappa_score(val_labels_list, val_preds, weights='quadratic')

        metrics.update('train', train_loss, train_acc)
        metrics.update('val', val_loss, val_acc, qwk=val_qwk*100)

        print(f"  → Loss: {train_loss:.4f}→{val_loss:.4f} | Acc: {train_acc:.1f}→{val_acc:.1f}% | QWK: {val_qwk:.3f}\n")

        scheduler.step()
        early_stop(val_qwk, model)

        if early_stop.early_stop:
            print(f"\n✓ Early stopping at epoch {epoch}")
            break

    elapsed = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n{'='*80}")
    print(f"Training completed in {elapsed:.1f} minutes")
    print(f"Best QWK: {max(metrics.history['val_qwk']):.3f}")
    print(f"Best Accuracy: {max(metrics.history['val_acc']):.1f}%")
    print(f"{'='*80}\n")

    # Save history
    with open('training_history.json', 'w') as f:
        json.dump(metrics.history, f, indent=2)

    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='processed_data/set1')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    metrics = train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
