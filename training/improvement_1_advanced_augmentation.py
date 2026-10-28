"""
Implementation 1: Advanced Augmentation with RandAugment + MixUp
- Expected improvement: +1–2% AUC
- Effort: Low (1 day)
- Inference cost: 0x (training only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm.data import RandAugment, Mixup
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)


class AdvancedAugmentationTransform:
    """
    Combines RandAugment + Mixup for improved training robustness.

    RandAugment: Automatically applies 2 random augmentation operations
    Mixup: Interpolates between pairs of images/labels

    Why this works:
    - RandAugment explores augmentation space automatically
    - Reduces need for manual augmentation tuning
    - Mixup: prevents overfitting by training on mixed examples
    - Model learns to generalize between feature spaces
    """

    def __init__(self, num_ops=2, magnitude=9, mixup_alpha=0.2, cutmix_alpha=1.0):
        """
        Args:
            num_ops: Number of random augmentation ops to apply (2–4)
            magnitude: Magnitude of augmentation (0–10, higher = stronger)
            mixup_alpha: MixUp alpha (0.2 = beta(0.2, 0.2))
            cutmix_alpha: CutMix alpha for region mixing
        """
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

        # RandAugment policy
        self.rand_augment = RandAugment(
            num_ops=num_ops,
            magnitude=magnitude,
            num_layers=1,
            prob=1.0
        )

    def __call__(self, image):
        """Apply RandAugment to image (PIL)."""
        return self.rand_augment(image)


def get_advanced_augmentation_transforms(augment=True):
    """
    Get training and validation transforms with advanced augmentation.
    """
    if augment:
        # Advanced training augmentation
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            # RandAugment: 2–3 random ops at magnitude 9
            RandAugment(num_ops=2, magnitude=9),
            # Standard augmentation (complementary to RandAugment)
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            # Color jitter
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return train_transform
    else:
        # Validation: minimal augmentation
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return val_transform


class MixupCutmixCollator:
    """
    Applies Mixup and CutMix to batches during training.

    Mixup:  x_mixed = λ*x_i + (1-λ)*x_j, y_mixed = λ*y_i + (1-λ)*y_j
    CutMix: Copy rectangle region from one image to another, interpolate labels

    Why this works:
    - Prevents model from memorizing training data
    - Encourages linear behavior between training examples
    - Improves generalization to unseen data
    """

    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob_mixup=0.7, prob_cutmix=0.3):
        """
        Args:
            mixup_alpha: Alpha for Beta distribution (higher = more mixing)
            cutmix_alpha: Alpha for CutMix Beta distribution
            prob_mixup: Probability of applying Mixup (vs CutMix)
            prob_cutmix: Probability of applying CutMix
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob_mixup = prob_mixup
        self.prob_cutmix = prob_cutmix

        # Use timm's built-in Mixup
        self.mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            cutmix_minmax=None,
            prob=prob_mixup,
            switch_prob=prob_cutmix,
            mode='batch',
            label_smoothing=0.1
        )

    def __call__(self, batch):
        """
        Apply Mixup/CutMix to batch.

        Args:
            batch: List of (image, label) tuples from DataLoader

        Returns:
            (mixed_images, mixed_labels)
        """
        images = torch.stack([x[0] for x in batch])
        labels = torch.tensor([x[1] for x in batch])

        # Apply Mixup/CutMix
        images, labels = self.mixup_fn(images, labels)

        return images, labels


class TrainerWithAdvancedAugmentation:
    """
    Training loop using advanced augmentation strategies.
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)

    def train_with_advanced_augmentation(self, train_loader, val_loader,
                                         epochs=50, lr=1e-3):
        """
        Train with RandAugment + Mixup.

        Key differences from baseline:
        1. RandAugment replaces manual augmentation tuning
        2. MixUp applied at batch level (during training)
        3. Expected: +1–2% AUC improvement
        """

        # Setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        # MixUp collator
        mixup_collator = MixupCutmixCollator(
            mixup_alpha=0.2,
            cutmix_alpha=1.0,
            prob_mixup=0.7,
            prob_cutmix=0.3
        )

        # Recreate loaders with MixUp collator
        train_loader_mixup = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
            collate_fn=mixup_collator,
            pin_memory=True
        )

        best_auc = 0

        for epoch in range(1, epochs + 1):
            # Training with RandAugment + MixUp
            self.model.train()
            train_loss = 0

            for images, labels in train_loader_mixup:
                # Note: images already have RandAugment applied (in transform)
                # Note: labels already mixed (via mixup_collator)

                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader_mixup)

            # Validation
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.model(images)
                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss /= len(val_loader)

            # Metrics
            from sklearn.metrics import roc_auc_score, accuracy_score
            val_acc = accuracy_score(all_labels, all_preds)

            try:
                val_auc = roc_auc_score(
                    all_labels,
                    torch.softmax(self.model(torch.from_numpy(np.array([x for x in val_loader.dataset])).to(self.device).float()), dim=1).cpu().numpy(),
                    multi_class='ovr',
                    average='weighted'
                )
            except:
                val_auc = 0

            logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                       f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

            scheduler.step()

        return best_auc


if __name__ == "__main__":
    logger.info("Advanced Augmentation Module (RandAugment + MixUp) Loaded")
    logger.info("Expected improvement: +1–2% AUC")
    logger.info("Training cost: Same as baseline (no inference overhead)")
