"""
Implementation 3: Multi-Task Learning
Tasks: DR Severity (main) + Lesion Detection + Image Quality

Expected improvement: +2–4% AUC on main DR grading task.
Single model — still edge-deployable.

How it works:
- Shared EfficientNet-B4 backbone learns richer retinal features
- Three task-specific heads branch off the shared features
- Each task provides regularization signal for the others
- At inference: use only DR severity head (no overhead)

Why it improves DR grading:
- Lesion detection forces backbone to attend to microaneurysms, hemorrhages
- Image quality task teaches the model what "good" retinal structure looks like
- Both act as auxiliary supervision signals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, cohen_kappa_score, accuracy_score

logger = logging.getLogger(__name__)


# ============================================================================
# MULTI-TASK MODEL
# ============================================================================

class MultiTaskDRModel(nn.Module):
    """
    Multi-task DR detection model.

    Architecture:
        Shared backbone (EfficientNet-B4)
            ├── DR Severity Head     → 5 classes (0-4 ICDR)
            ├── Lesion Detection Head → 1 output (sigmoid: lesion present/absent)
            └── Image Quality Head   → 3 classes (good/acceptable/poor)

    At inference: only DR head is used. No added cost.
    """

    def __init__(self, num_dr_classes=5, pretrained=True):
        super().__init__()

        # Shared backbone
        backbone = models.efficientnet_b4(pretrained=pretrained)
        in_features = backbone.classifier[1].in_features  # 1792 for B4
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.backbone_features = in_features

        # Shared intermediate projection (improves feature quality for all heads)
        self.shared_proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Task 1: DR Severity — main task
        self.dr_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_dr_classes)
        )

        # Task 2: Lesion Detection — binary (any lesion present vs. normal)
        # Lesion: microaneurysms, hemorrhages, exudates, cotton wool spots
        self.lesion_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)   # BCEWithLogitsLoss (sigmoid applied in loss)
        )

        # Task 3: Image Quality — 3 classes
        # 0: Good (sharp, well-illuminated, clear disc/macula)
        # 1: Acceptable (mild blur or uneven illumination, still gradable)
        # 2: Poor (severe blur, missing structures, not gradable)
        self.quality_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

    def forward(self, x, return_all=True):
        """
        Args:
            x: (batch, 3, 512, 512) preprocessed retinal images
            return_all: If True, return all task outputs (training).
                        If False, return only DR logits (inference).

        Returns (training):
            dict with keys 'dr', 'lesion', 'quality'
        Returns (inference):
            dr_logits: (batch, 5)
        """
        features = self.backbone(x)           # (batch, 1792)
        shared = self.shared_proj(features)    # (batch, 512)

        dr_logits = self.dr_head(shared)       # (batch, 5)

        if not return_all:
            return dr_logits

        lesion_logits = self.lesion_head(shared)   # (batch, 1)
        quality_logits = self.quality_head(shared) # (batch, 3)

        return {
            'dr': dr_logits,
            'lesion': lesion_logits,
            'quality': quality_logits
        }

    def predict_dr(self, x):
        """Inference-only DR prediction — no overhead from auxiliary heads."""
        return self.forward(x, return_all=False)


# ============================================================================
# MULTI-TASK LOSS
# ============================================================================

class MultiTaskLoss(nn.Module):
    """
    Weighted combination of task-specific losses.

    L_total = w_dr * L_dr + w_lesion * L_lesion + w_quality * L_quality

    Default weights (0.6 / 0.2 / 0.2) prioritize DR grading while still
    benefiting from auxiliary tasks.
    """

    def __init__(self, w_dr=0.6, w_lesion=0.2, w_quality=0.2,
                 label_smoothing=0.1):
        super().__init__()
        self.w_dr = w_dr
        self.w_lesion = w_lesion
        self.w_quality = w_quality

        # Task-specific loss functions
        self.dr_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.lesion_loss_fn = nn.BCEWithLogitsLoss()
        self.quality_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with keys 'dr', 'lesion', 'quality'
            targets: dict with keys 'dr', 'lesion', 'quality'
                     lesion target: float tensor (0.0 or 1.0)
                     dr/quality: long tensor (class indices)

        Returns:
            Scalar total loss, plus component losses for logging
        """
        l_dr = self.dr_loss_fn(outputs['dr'], targets['dr'])
        l_lesion = self.lesion_loss_fn(
            outputs['lesion'].squeeze(1),
            targets['lesion'].float()
        )
        l_quality = self.quality_loss_fn(outputs['quality'], targets['quality'])

        total = self.w_dr * l_dr + self.w_lesion * l_lesion + self.w_quality * l_quality

        return total, {'dr': l_dr.item(), 'lesion': l_lesion.item(), 'quality': l_quality.item()}


# ============================================================================
# MULTI-TASK DATASET WRAPPER
# ============================================================================

class MultiTaskDRDataset(torch.utils.data.Dataset):
    """
    Dataset that returns DR grade + lesion label + quality label per image.

    For images where lesion/quality labels are not available:
    - Lesion: auto-derived from DR grade (Grade 0 = no lesion, Grade 1+ = lesion)
    - Quality: defaults to 0 (good) — override with real annotations if available

    This lets you start multi-task training immediately with only DR labels,
    then add richer annotations progressively.
    """

    def __init__(self, base_dataset, quality_labels=None):
        """
        Args:
            base_dataset: DRDataset instance with images and DR labels
            quality_labels: Optional dict {image_path: quality_class} or None
        """
        self.base = base_dataset
        self.quality_labels = quality_labels or {}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, dr_label = self.base[idx]
        dr_label = dr_label.long()

        # Lesion: derived from DR grade (grade 0 = no lesion, grade 1+ = lesion)
        # shortcut: derived from grade, not pixel-level annotation; upgrade when annotations available
        lesion_label = torch.tensor(1.0 if dr_label.item() > 0 else 0.0)

        # Quality: use annotation if available, else default to 0 (good)
        img_path = self.base.images[idx]
        quality = self.quality_labels.get(img_path, 0)
        quality_label = torch.tensor(quality, dtype=torch.long)

        return image, {
            'dr': dr_label,
            'lesion': lesion_label,
            'quality': quality_label
        }


def multitask_collate_fn(batch):
    """Custom collate for multi-task batches."""
    images = torch.stack([item[0] for item in batch])
    targets = {
        'dr':      torch.stack([item[1]['dr']      for item in batch]),
        'lesion':  torch.stack([item[1]['lesion']  for item in batch]),
        'quality': torch.stack([item[1]['quality'] for item in batch]),
    }
    return images, targets


# ============================================================================
# MULTI-TASK TRAINER
# ============================================================================

class MultiTaskTrainer:
    """Full training loop for multi-task DR model."""

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def train(self, train_loader, val_loader,
              epochs=50, lr=1e-3,
              w_dr=0.6, w_lesion=0.2, w_quality=0.2,
              save_path='checkpoints/multitask_dr_model.pth'):
        """
        Train multi-task model.

        Args:
            train_loader: DataLoader returning (images, targets_dict)
            val_loader:   DataLoader returning (images, targets_dict)
            w_dr/w_lesion/w_quality: Task loss weights (must sum to 1.0)
            save_path: Where to save best checkpoint
        """
        import os
        os.makedirs('checkpoints', exist_ok=True)

        model = MultiTaskDRModel(num_dr_classes=5, pretrained=True).to(self.device)
        criterion = MultiTaskLoss(w_dr=w_dr, w_lesion=w_lesion, w_quality=w_quality)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        best_qwk = -1
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # --- Training ---
            model.train()
            total_loss = 0
            component_losses = {'dr': 0, 'lesion': 0, 'quality': 0}

            for images, targets in tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]"):
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                optimizer.zero_grad()
                outputs = model(images, return_all=True)
                loss, components = criterion(outputs, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                for k in component_losses:
                    component_losses[k] += components[k]

            n = len(train_loader)
            logger.info(
                f"Epoch {epoch:3d} | Total: {total_loss/n:.4f} | "
                f"DR: {component_losses['dr']/n:.4f} | "
                f"Lesion: {component_losses['lesion']/n:.4f} | "
                f"Quality: {component_losses['quality']/n:.4f}"
            )

            # --- Validation (DR task only) ---
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(self.device)
                    dr_logits = model.predict_dr(images)
                    preds = torch.argmax(dr_logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(targets['dr'].numpy())

            qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
            acc = accuracy_score(all_labels, all_preds)
            logger.info(f"           Val DR → Acc: {acc:.4f} | QWK: {qwk:.4f}")

            if qwk > best_qwk + 1e-4:
                best_qwk = qwk
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                logger.info(f"  → Best model saved (QWK={best_qwk:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    logger.info("Early stopping triggered")
                    break

            scheduler.step()

        logger.info(f"Multi-task training complete. Best QWK: {best_qwk:.4f}")
        return save_path


if __name__ == "__main__":
    logger.info("Multi-Task Learning Module loaded")
    logger.info("Tasks: DR Severity + Lesion Detection + Image Quality")
    logger.info("Expected improvement: +2–4% AUC on DR grading")
    logger.info("Deployed model: single model, DR head only at inference")
