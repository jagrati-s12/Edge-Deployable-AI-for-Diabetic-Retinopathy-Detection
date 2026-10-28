"""
Implementation 2: Knowledge Distillation
Teacher: EfficientNet-B5 (larger, higher accuracy)
Student: EfficientNet-B4 (our edge model, same size)

Expected improvement: +2–3% AUC with no increase in deployed model size.

How it works:
1. Train a large "teacher" model (EfficientNet-B5, ~30M params)
2. Teacher learns soft probability distributions over DR grades
3. Student (EfficientNet-B4, ~19M params) learns from BOTH:
   - Hard labels (ground truth grades)
   - Soft targets (teacher's probability distributions)
4. Soft targets carry inter-class relationship info the hard labels lose
   e.g., teacher says Grade 2 is 70% confident but 20% Grade 3 — student
   learns that moderate/severe are related, hard labels can't teach that.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, cohen_kappa_score

logger = logging.getLogger(__name__)


# ============================================================================
# TEACHER MODEL (EfficientNet-B5, trained first)
# ============================================================================

class TeacherModel(nn.Module):
    """
    EfficientNet-B5 — larger backbone for knowledge distillation teacher.
    ~30M parameters. Not deployed; only used for distillation training.
    """

    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        from torchvision import models
        self.backbone = models.efficientnet_b5(pretrained=pretrained)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# DISTILLATION LOSS
# ============================================================================

class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.

    L_total = alpha * L_distill + (1 - alpha) * L_CE

    L_distill: KL divergence between student and teacher soft targets
               (softened by temperature T — higher T = softer distribution)
    L_CE:      Standard cross-entropy against true labels

    Temperature T=4 works well in practice:
    - Makes probability distributions softer (less peaked)
    - Transfers more information about inter-class relationships
    - alpha=0.7 means teacher guidance dominates early, true labels anchor it
    """

    def __init__(self, temperature=4.0, alpha=0.7, label_smoothing=0.1):
        """
        Args:
            temperature: Softening temperature (higher = softer, more transfer)
            alpha: Weight for distillation vs CE loss (0.7 = 70% distillation)
            label_smoothing: Smoothing for CE component
        """
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, true_labels):
        """
        Args:
            student_logits: (batch, 5) — student model output
            teacher_logits: (batch, 5) — teacher model output (no grad)
            true_labels:    (batch,)   — ground truth DR grades

        Returns:
            Scalar loss
        """
        # Distillation: match teacher's softened distribution
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
        # Scale by T² to preserve gradient magnitudes after temperature scaling
        distill_loss = self.kl(student_soft, teacher_soft) * (self.T ** 2)

        # Supervised: match ground truth
        ce_loss = self.ce(student_logits, true_labels)

        return self.alpha * distill_loss + (1 - self.alpha) * ce_loss


# ============================================================================
# DISTILLATION TRAINER
# ============================================================================

class DistillationTrainer:
    """
    Two-phase training pipeline:
      Phase 1 — Train teacher (EfficientNet-B5) to convergence.
      Phase 2 — Distill into student (EfficientNet-B4) using teacher soft labels.
    """

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------
    # PHASE 1: Train teacher
    # ------------------------------------------------------------------
    def train_teacher(self, train_loader, val_loader,
                      epochs=50, lr=5e-4,
                      save_path='checkpoints/teacher_b5.pth'):
        """
        Train EfficientNet-B5 teacher model.
        Larger model → higher accuracy ceiling → better soft labels for student.
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Training Teacher (EfficientNet-B5)")
        logger.info("=" * 60)

        teacher = TeacherModel(num_classes=5, pretrained=True).to(self.device)

        optimizer = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_qwk = -1
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            teacher.train()
            epoch_loss = 0
            for images, labels in tqdm(train_loader, desc=f"Teacher Epoch {epoch}"):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = teacher(images)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            # Validate
            teacher.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    preds = torch.argmax(teacher(images), dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
            logger.info(f"Teacher Epoch {epoch:3d} | Loss: {epoch_loss/len(train_loader):.4f} | QWK: {qwk:.4f}")

            if qwk > best_qwk + 1e-4:
                best_qwk = qwk
                patience_counter = 0
                torch.save(teacher.state_dict(), save_path)
                logger.info(f"  → Best teacher saved (QWK={best_qwk:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    logger.info("Teacher early stopping triggered")
                    break

            scheduler.step()

        logger.info(f"Teacher training complete. Best QWK: {best_qwk:.4f}")
        logger.info(f"Saved to: {save_path}")
        return save_path

    # ------------------------------------------------------------------
    # PHASE 2: Distill into student
    # ------------------------------------------------------------------
    def distill_to_student(self, train_loader, val_loader,
                           teacher_path='checkpoints/teacher_b5.pth',
                           epochs=50, lr=1e-3,
                           temperature=4.0, alpha=0.7,
                           save_path='checkpoints/student_b4_distilled.pth'):
        """
        Distill teacher knowledge into EfficientNet-B4 student.

        The student:
        - Stays the same size as the original model (19M params, 19MB INT8)
        - Learns from teacher's soft probability distributions
        - Retains full edge deployability
        """
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.dr_model_complete import EfficientNetB4WithSE

        logger.info("=" * 60)
        logger.info("PHASE 2: Distilling Teacher → Student (EfficientNet-B4)")
        logger.info(f"  Temperature: {temperature}, Alpha: {alpha}")
        logger.info("=" * 60)

        # Load frozen teacher
        teacher = TeacherModel(num_classes=5, pretrained=False).to(self.device)
        teacher.load_state_dict(torch.load(teacher_path, map_location=self.device))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        logger.info("Teacher loaded and frozen.")

        # Student model (same as deployment model)
        student = EfficientNetB4WithSE(num_classes=5, pretrained=True).to(self.device)

        optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        criterion = DistillationLoss(
            temperature=temperature,
            alpha=alpha,
            label_smoothing=0.1
        )

        best_qwk = -1
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            student.train()
            epoch_loss = 0

            for images, labels in tqdm(train_loader, desc=f"Distill Epoch {epoch}"):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                student_logits = student(images)

                # Get teacher soft labels (no gradient)
                with torch.no_grad():
                    teacher_logits = teacher(images)

                loss = criterion(student_logits, teacher_logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            # Validate
            student.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    preds = torch.argmax(student(images), dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
            logger.info(f"Student Epoch {epoch:3d} | Loss: {epoch_loss/len(train_loader):.4f} | QWK: {qwk:.4f}")

            if qwk > best_qwk + 1e-4:
                best_qwk = qwk
                patience_counter = 0
                torch.save(student.state_dict(), save_path)
                logger.info(f"  → Best student saved (QWK={best_qwk:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    logger.info("Student early stopping triggered")
                    break

            scheduler.step()

        logger.info(f"Distillation complete. Best Student QWK: {best_qwk:.4f}")
        logger.info(f"Student saved to: {save_path}")
        logger.info("Student model is same size as original → fully edge-deployable.")
        return save_path

    def run_full_pipeline(self, train_loader, val_loader):
        """Run Phase 1 (teacher) then Phase 2 (distillation) end-to-end."""
        import os
        os.makedirs('checkpoints', exist_ok=True)

        teacher_path = self.train_teacher(train_loader, val_loader)
        student_path = self.distill_to_student(train_loader, val_loader,
                                               teacher_path=teacher_path)
        return student_path


if __name__ == "__main__":
    logger.info("Knowledge Distillation Module (EfficientNet-B5 → B4)")
    logger.info("Expected improvement: +2–3% AUC")
    logger.info("Deployed model size: unchanged (19MB INT8)")
