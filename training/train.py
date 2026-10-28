"""
Training Guide for Edge-Deployable Diabetic Retinopathy Detection Model

This module provides:
1. Complete training pipeline with validation
2. Model improvement strategies
3. Hyperparameter tuning recommendations
4. Monitoring and visualization utilities
5. Cross-dataset evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import cohen_kappa_score
import cv2
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 1: DATASET LOADING AND AUGMENTATION
# ============================================================================

class DRDataset(Dataset):
    """
    Custom Dataset for Diabetic Retinopathy images.

    Expected directory structure:
    data/
    ├── 0/  (No DR)
    │   ├── img1.jpg
    │   ├── img2.jpg
    ├── 1/  (Mild)
    ├── 2/  (Moderate)
    ├── 3/  (Severe)
    └── 4/  (Proliferative)
    """

    def __init__(self, root_dir: str, transform=None, augment=True):
        """
        Args:
            root_dir: Path to dataset root
            transform: Torchvision transforms
            augment: Apply augmentation (True for training, False for validation)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.augment = augment
        self.images = []
        self.labels = []

        # Load image paths and labels
        for class_idx in range(5):
            class_dir = self.root_dir / str(class_idx)
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
                for img_path in class_dir.glob('*.png'):
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)

        logger.info(f"Loaded {len(self.images)} images from {root_dir}")
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Print class balance statistics."""
        distribution = Counter(self.labels)
        total = len(self.labels)
        logger.info("Class Distribution:")
        for class_idx in range(5):
            count = distribution[class_idx]
            pct = (count / total) * 100
            logger.info(f"  Grade {class_idx}: {count} images ({pct:.1f}%)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"Failed to load {img_path}")
            return torch.zeros(3, 512, 512), torch.tensor(label)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))

        # Apply transform
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, torch.tensor(label, dtype=torch.long)


def get_augmentation_transforms(augment=True):
    """
    Get training and validation transforms.

    Training: aggressive augmentation to prevent overfitting
    Validation: minimal transforms (only normalization)
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        return train_transform
    else:
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        return val_transform


def get_class_weights(dataset: DRDataset):
    """
    Compute class weights to handle imbalanced data.
    Rarer classes (severe/proliferative DR) get higher weights.
    """
    from sklearn.utils.class_weight import compute_class_weight

    labels = np.array(dataset.labels)
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    weights = torch.from_numpy(weights).float()

    logger.info("Class weights for loss function:")
    for idx, w in enumerate(weights):
        logger.info(f"  Grade {idx}: {w:.3f}")

    return weights


# ============================================================================
# SECTION 2: TRAINING UTILITIES AND CALLBACKS
# ============================================================================

class EarlyStopping:
    """
    Stop training when validation metric stops improving.
    Saves best model checkpoint.
    """

    def __init__(self, patience=5, min_delta=1e-4, model_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.model_path = model_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric, model):
        if self.best_score is None:
            self.best_score = val_metric
            self.save_checkpoint(model)
        elif val_metric > self.best_score + self.min_delta:
            self.best_score = val_metric
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")

    def save_checkpoint(self, model):
        logger.info(f"Saving best model to {self.model_path}")
        torch.save(model.state_dict(), self.model_path)


class MetricsTracker:
    """
    Track training metrics: loss, accuracy, AUC, etc.
    Generate plots and reports.
    """

    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_auc': [],
            'val_auc': [],
            'val_qwk': []  # Quadratic Weighted Kappa
        }
        self.best_metrics = {}

    def update(self, phase, loss, acc, auc=None, qwk=None):
        if phase == 'train':
            self.history['train_loss'].append(loss)
            self.history['train_acc'].append(acc)
            if auc is not None:
                self.history['train_auc'].append(auc)
        elif phase == 'val':
            self.history['val_loss'].append(loss)
            self.history['val_acc'].append(acc)
            if auc is not None:
                self.history['val_auc'].append(auc)
            if qwk is not None:
                self.history['val_qwk'].append(qwk)

    def plot_history(self, save_path='training_history.png'):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].grid(alpha=0.3)

        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history['val_acc'], label='Val', linewidth=2)
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].set_title('Training & Validation Accuracy')
        axes[0, 1].grid(alpha=0.3)

        # AUC
        if self.history['train_auc']:
            axes[1, 0].plot(self.history['train_auc'], label='Train', linewidth=2)
            axes[1, 0].plot(self.history['val_auc'], label='Val', linewidth=2)
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].legend()
            axes[1, 0].set_title('Training & Validation AUC')
            axes[1, 0].grid(alpha=0.3)

        # QWK
        if self.history['val_qwk']:
            axes[1, 1].plot(self.history['val_qwk'], linewidth=2, color='purple')
            axes[1, 1].set_ylabel('Quadratic Weighted Kappa')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_title('Validation QWK')
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        logger.info(f"Training history saved to {save_path}")
        plt.close()

    def save_json(self, save_path='metrics.json'):
        """Save metrics to JSON for later analysis."""
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Metrics saved to {save_path}")


# ============================================================================
# SECTION 3: TRAINING LOOP
# ============================================================================

class Trainer:
    """
    Main training class.
    Handles: forward pass, backward pass, validation, checkpoint saving.
    """

    def __init__(self, model, device='cuda', use_ordinal_loss=True):
        self.model = model
        self.device = device
        self.model.to(device)

        self.use_ordinal_loss = use_ordinal_loss
        self.metrics_tracker = MetricsTracker()

    def setup_training(self, train_loader, val_loader, class_weights=None,
                       lr=1e-3, ordinal_weight=1.0, label_smoothing=0.1):
        """
        Setup optimizer, loss function, and scheduler.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            class_weights: Tensor of class weights (for imbalanced data)
            lr: Initial learning rate
            ordinal_weight: Weight for ordinal penalty (higher = stricter ordinal enforcement)
            label_smoothing: Label smoothing factor (0.0-1.0)
        """
        # Loss function
        if self.use_ordinal_loss:
            from models.dr_model_complete import OrdinalAwareLoss
            self.criterion = OrdinalAwareLoss(
                num_classes=5,
                ordinal_weights=self._get_ordinal_matrix(5, ordinal_weight),
                label_smoothing=label_smoothing
            )
            logger.info("Using Ordinal-Aware Loss with label smoothing")
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing
            )
            logger.info("Using Cross-Entropy Loss with label smoothing")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Initial restart at epoch 10
            T_mult=2,  # Double restart interval each time
            eta_min=1e-6
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=5, model_path='best_dr_model.pth')

        self.train_loader = train_loader
        self.val_loader = val_loader

        logger.info(f"Training setup complete. LR={lr}, Ordinal Weight={ordinal_weight}")

    @staticmethod
    def _get_ordinal_matrix(num_classes=5, weight=1.0):
        """Create ordinal penalty matrix."""
        matrix = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                matrix[i, j] = float(abs(i - j)) ** 2 * weight
        return matrix

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [TRAIN]")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        acc = np.mean(np.array(all_preds) == np.array(all_labels))

        return avg_loss, acc

    def validate(self, epoch):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [VAL]")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)

                total_loss += loss.item()
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        avg_loss = total_loss / len(self.val_loader)
        acc = np.mean(all_preds == all_labels)

        # AUC (one-vs-rest for each class)
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        except:
            auc = 0.0

        # Quadratic Weighted Kappa (clinical metric)
        qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

        return avg_loss, acc, auc, qwk, all_preds, all_labels

    def train(self, epochs=50):
        """Complete training loop."""
        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, val_auc, val_qwk, val_preds, val_labels = self.validate(epoch)

            # Log metrics
            self.metrics_tracker.update('train', train_loss, train_acc, auc=0)
            self.metrics_tracker.update('val', val_loss, val_acc, auc=val_auc, qwk=val_qwk)

            logger.info(f"Epoch {epoch:3d} | "
                       f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                       f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, QWK: {val_qwk:.4f}")

            # Update learning rate
            self.scheduler.step()

            # Early stopping
            self.early_stopping(val_qwk, self.model)
            if self.early_stopping.early_stop:
                logger.info("Training stopped by early stopping")
                break

        # Plot and save
        self.metrics_tracker.plot_history('training_history.png')
        self.metrics_tracker.save_json('metrics.json')

        return self.metrics_tracker


# ============================================================================
# SECTION 4: MODEL IMPROVEMENT STRATEGIES
# ============================================================================

class ModelImprovementStrategies:
    """
    Collection of techniques to improve model performance.
    """

    @staticmethod
    def strategy_1_ensemble():
        """
        Strategy 1: Ensemble Multiple Models

        Train 3–5 models with different random seeds and average predictions.
        Reduces variance and improves robustness.

        Pros:
        - Simple to implement
        - Consistent improvement (1–3% AUC boost)

        Cons:
        - Increases inference cost (3–5x slower)
        - Larger model files

        Implementation:
        - Train models with seeds: 42, 123, 456, 789, 999
        - Average softmax probabilities during inference
        - Use weighted averaging based on individual model validation AUC
        """
        return {
            'name': 'Ensemble Models',
            'improvement_expected': '1–3% AUC',
            'inference_cost_increase': '3–5x',
            'deployment_feasibility': 'Medium (larger binary, longer latency)',
            'implementation_complexity': 'Low'
        }

    @staticmethod
    def strategy_2_advanced_augmentation():
        """
        Strategy 2: Advanced Data Augmentation (RandAugment, MixUp, CutMix)

        Use automatic augmentation policies that search for optimal transforms.

        Pros:
        - Improves generalization (1–2% AUC)
        - No inference cost

        Cons:
        - Longer training time
        - Requires tuning augmentation strength

        Implementation:
        - Add RandAugment (num_ops=2, magnitude=9)
        - Add MixUp (alpha=0.2) during training
        - Add CutMix (alpha=1.0) for selective region removal
        """
        return {
            'name': 'Advanced Augmentation (RandAugment + MixUp)',
            'improvement_expected': '1–2% AUC',
            'inference_cost_increase': '0x (training only)',
            'deployment_feasibility': 'High (no change to deployed model)',
            'implementation_complexity': 'Medium',
            'libraries_needed': ['timm', 'albumentations']
        }

    @staticmethod
    def strategy_3_knowledge_distillation():
        """
        Strategy 3: Knowledge Distillation

        Train a larger model (teacher), then compress knowledge into smaller model (student).

        Pros:
        - Improves small model performance (2–3% AUC)
        - Model remains small for edge deployment

        Cons:
        - Requires training two models
        - Moderate complexity

        Implementation:
        - Train teacher: EfficientNet-B5 or ResNet-50
        - Train student: EfficientNet-B4 (our current model)
        - Use distillation loss: α*CE(student, labels) + (1-α)*KL(student, teacher)
        """
        return {
            'name': 'Knowledge Distillation (Large teacher → Small student)',
            'improvement_expected': '2–3% AUC',
            'inference_cost_increase': '0x (same model size)',
            'deployment_feasibility': 'High',
            'implementation_complexity': 'Medium-High',
            'recommended': True
        }

    @staticmethod
    def strategy_4_self_supervised_pretraining():
        """
        Strategy 4: Self-Supervised Pretraining

        Pretrain model on unlabeled retinal images using contrastive learning (SimCLR, MoCo).
        Then fine-tune on labeled DR data.

        Pros:
        - Leverages unlabeled data (if available)
        - Significant improvement (3–5% AUC)

        Cons:
        - Requires large unlabeled dataset
        - Higher computational cost
        - Moderate complexity

        Implementation:
        - Use SimCLR or MoCo on EyePACS unlabeled images
        - Freeze backbone, fine-tune classifier
        """
        return {
            'name': 'Self-Supervised Pretraining (SimCLR)',
            'improvement_expected': '3–5% AUC',
            'data_requirement': 'Large unlabeled dataset (~100K images)',
            'deployment_feasibility': 'High',
            'implementation_complexity': 'High'
        }

    @staticmethod
    def strategy_5_multi_task_learning():
        """
        Strategy 5: Multi-Task Learning

        Train model on DR grading + lesion detection + image quality simultaneously.

        Pros:
        - Leverages related tasks to improve main task
        - Better representation learning (2–4% AUC)

        Cons:
        - Requires additional annotations (lesion masks, quality labels)
        - More complex training

        Implementation:
        - Branch 1: DR severity (5 classes)
        - Branch 2: Lesion detection (presence/absence)
        - Branch 3: Image quality (good/acceptable/poor)
        - Shared backbone, task-specific heads
        - Weight losses: L_total = 0.6*L_dr + 0.2*L_lesion + 0.2*L_quality
        """
        return {
            'name': 'Multi-Task Learning (DR + Lesion Detection + Quality)',
            'improvement_expected': '2–4% AUC',
            'annotation_requirement': 'Lesion masks + quality labels',
            'deployment_feasibility': 'High (single model)',
            'implementation_complexity': 'High',
            'recommended': True
        }

    @staticmethod
    def strategy_6_ordinal_classification_refinement():
        """
        Strategy 6: Ordinal Classification Refinement

        Currently using ordinal-aware loss. Further improvements:

        Option A: Ordinal Regression
        - Train on cumulative labels: [1,0,0,0,0] for grade 0, [1,1,0,0,0] for grade 1
        - Decode by finding first 0
        - Better captures ordinal structure

        Option B: Ranker-based approach
        - Train pairwise ranking comparisons (A > B, B > C, etc.)
        - More flexible but complex

        Pros:
        - Better captures ordinal structure (1–2% AUC)
        - Reduces severe errors further

        Cons:
        - Requires careful implementation
        """
        return {
            'name': 'Ordinal Regression with Cumulative Labels',
            'improvement_expected': '1–2% AUC + fewer severe errors',
            'inference_cost_increase': '0x',
            'deployment_feasibility': 'High',
            'implementation_complexity': 'Medium'
        }

    @staticmethod
    def strategy_7_domain_adaptation():
        """
        Strategy 7: Domain Adaptation for Low-Quality Images

        Train with explicit domain shift awareness.

        Approaches:
        1. Adversarial Domain Adaptation (DANN)
           - Shared encoder + class classifier + domain classifier
           - Domain classifier tries to distinguish clean vs. degraded images
           - Encoder learns domain-invariant features

        2. Continuous Domain Randomization
           - During training, randomly blur/noise images to varying degrees
           - Model learns to handle any degradation level

        Pros:
        - Explicit robustness to image degradation (2–3% on degraded images)
        - Better real-world performance

        Cons:
        - More complex training
        - Requires careful balancing
        """
        return {
            'name': 'Domain Adaptation (DANN or Continuous Randomization)',
            'improvement_expected': '2–3% on degraded images',
            'deployment_feasibility': 'High',
            'implementation_complexity': 'High',
            'recommended': True
        }

    @staticmethod
    def strategy_8_hyperparameter_optimization():
        """
        Strategy 8: Systematic Hyperparameter Tuning

        Use Bayesian optimization to find best hyperparameters.

        Key hyperparameters to tune:
        - Learning rate: [1e-4, 1e-2]
        - Batch size: [16, 32, 64]
        - Dropout: [0.1, 0.5]
        - Ordinal weight: [0.5, 2.0]
        - Label smoothing: [0.0, 0.3]
        - Weight decay: [0, 1e-4, 1e-3]

        Pros:
        - Potentially significant improvement (1–2% AUC)
        - Systematic approach

        Cons:
        - Computationally expensive (requires 50–100 trials)
        """
        return {
            'name': 'Bayesian Hyperparameter Optimization',
            'improvement_expected': '1–2% AUC',
            'computational_cost': '50–100 full training runs',
            'deployment_feasibility': 'N/A (affects final weights)',
            'implementation_complexity': 'High (but frameworks exist: Optuna, Ray Tune)',
            'libraries_needed': ['optuna', 'ray[tune]']
        }


def print_improvement_strategies():
    """Print all improvement strategies with details."""
    strategies = [
        ModelImprovementStrategies.strategy_1_ensemble(),
        ModelImprovementStrategies.strategy_2_advanced_augmentation(),
        ModelImprovementStrategies.strategy_3_knowledge_distillation(),
        ModelImprovementStrategies.strategy_4_self_supervised_pretraining(),
        ModelImprovementStrategies.strategy_5_multi_task_learning(),
        ModelImprovementStrategies.strategy_6_ordinal_classification_refinement(),
        ModelImprovementStrategies.strategy_7_domain_adaptation(),
        ModelImprovementStrategies.strategy_8_hyperparameter_optimization(),
    ]

    logger.info("=" * 80)
    logger.info("MODEL IMPROVEMENT STRATEGIES FOR DR DETECTION")
    logger.info("=" * 80)

    for i, strategy in enumerate(strategies, 1):
        logger.info(f"\n{i}. {strategy['name']}")
        for key, value in strategy.items():
            if key != 'name':
                logger.info(f"   {key}: {value}")


# ============================================================================
# SECTION 5: EVALUATION AND ANALYSIS
# ============================================================================

def generate_evaluation_report(predictions, labels, model_name='DR Model'):
    """
    Generate comprehensive evaluation report.
    """
    logger.info("=" * 80)
    logger.info(f"EVALUATION REPORT: {model_name}")
    logger.info("=" * 80)

    # Overall metrics
    acc = np.mean(predictions == labels)
    logger.info(f"\nOverall Accuracy: {acc:.4f}")

    # Per-class metrics
    logger.info("\nPer-Class Metrics:")
    report = classification_report(labels, predictions,
                                   target_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'],
                                   digits=4)
    logger.info(report)

    # Quadratic Weighted Kappa
    qwk = cohen_kappa_score(labels, predictions, weights='quadratic')
    logger.info(f"\nQuadratic Weighted Kappa (QWK): {qwk:.4f}")
    logger.info("  (Gold standard for ordinal classification in medical AI)")

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    logger.info("\nConfusion Matrix:")
    logger.info(cm)

    # Error analysis
    errors = predictions != labels
    error_types = np.abs(predictions[errors] - labels[errors])
    logger.info(f"\nError Analysis:")
    logger.info(f"  Total errors: {np.sum(errors)} ({np.mean(errors)*100:.2f}%)")
    logger.info(f"  Adjacent class errors (±1): {np.sum(error_types == 1)} ({np.sum(error_types == 1)/len(error_types)*100:.2f}%)")
    logger.info(f"  Moderate errors (±2): {np.sum(error_types == 2)} ({np.sum(error_types == 2)/len(error_types)*100:.2f}%)")
    logger.info(f"  Severe errors (±3 or more): {np.sum(error_types >= 3)} ({np.sum(error_types >= 3)/len(error_types)*100:.2f}%)")


if __name__ == "__main__":
    logger.info("DR Detection Training Module Initialized")
    print_improvement_strategies()
