"""
Unified Improvement Runner
Chains all 3 improvements in order of increasing complexity.

Usage:
    # Run all improvements sequentially:
    python training/run_improvements.py --data_dir ./data --all

    # Run specific improvements:
    python training/run_improvements.py --data_dir ./data --augment
    python training/run_improvements.py --data_dir ./data --distill
    python training/run_improvements.py --data_dir ./data --multitask

    # Evaluate a specific checkpoint:
    python training/run_improvements.py --data_dir ./data --eval --checkpoint checkpoints/student_b4_distilled.pth
"""

import argparse
import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import (
    DRDataset,
    get_augmentation_transforms,
    get_class_weights,
    Trainer,
    generate_evaluation_report
)
from training.improvement_1_advanced_augmentation import (
    get_advanced_augmentation_transforms,
    MixupCutmixCollator
)
from training.improvement_2_knowledge_distillation import DistillationTrainer
from training.improvement_3_multitask_learning import (
    MultiTaskDRDataset,
    MultiTaskTrainer,
    multitask_collate_fn
)
from models.dr_model_complete import EfficientNetB4WithSE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s'
)
logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

def build_splits(data_dir, val_frac=0.2, seed=42):
    """Return (train_dataset, val_dataset) as plain DRDataset instances."""
    full = DRDataset(data_dir, augment=False)  # no transform yet
    n_val = int(len(full) * val_frac)
    n_train = len(full) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=gen)
    return full, train_ds, val_ds


def evaluate_model(model_path, dataset, device, label='Model'):
    """Load a checkpoint and print full evaluation metrics."""
    device = torch.device(device)
    model = EfficientNetB4WithSE(num_classes=5, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = torch.argmax(model(images), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy() if not isinstance(labels, np.ndarray) else labels)

    generate_evaluation_report(np.array(all_preds), np.array(all_labels), label)


# ── improvement 1 ────────────────────────────────────────────────────────────

def run_augmentation(data_dir, epochs, batch_size, lr, device):
    logger.info("=" * 70)
    logger.info("IMPROVEMENT 1 — Advanced Augmentation (RandAugment + MixUp)")
    logger.info("Expected gain: +1–2% AUC | Inference overhead: none")
    logger.info("=" * 70)

    full_ds, train_ds, val_ds = build_splits(data_dir)

    # Apply advanced transforms to training split
    train_ds.dataset.transform = get_advanced_augmentation_transforms(augment=True)
    val_ds.dataset.transform   = get_augmentation_transforms(augment=False)

    mixup_collator = MixupCutmixCollator(mixup_alpha=0.2, cutmix_alpha=1.0)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        collate_fn=mixup_collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    model = EfficientNetB4WithSE(num_classes=5, pretrained=True)
    class_weights = get_class_weights(full_ds)

    trainer = Trainer(model, device=device, use_ordinal_loss=True)
    trainer.setup_training(
        train_loader, val_loader,
        class_weights=class_weights,
        lr=lr,
        ordinal_weight=1.0,
        label_smoothing=0.1
    )
    # Override early-stopping save path
    trainer.early_stopping.model_path = 'checkpoints/augmented_model.pth'

    trainer.train(epochs=epochs)
    logger.info("Improvement 1 complete → checkpoints/augmented_model.pth")


# ── improvement 2 ────────────────────────────────────────────────────────────

def run_distillation(data_dir, epochs, batch_size, lr, device):
    logger.info("=" * 70)
    logger.info("IMPROVEMENT 2 — Knowledge Distillation (B5 teacher → B4 student)")
    logger.info("Expected gain: +2–3% AUC | Deployed model size: unchanged")
    logger.info("=" * 70)

    full_ds, train_ds, val_ds = build_splits(data_dir)
    train_ds.dataset.transform = get_advanced_augmentation_transforms(augment=True)
    val_ds.dataset.transform   = get_augmentation_transforms(augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    trainer = DistillationTrainer(device=device)
    trainer.run_full_pipeline(train_loader, val_loader)
    logger.info("Improvement 2 complete → checkpoints/student_b4_distilled.pth")


# ── improvement 3 ────────────────────────────────────────────────────────────

def run_multitask(data_dir, epochs, batch_size, lr, device):
    logger.info("=" * 70)
    logger.info("IMPROVEMENT 3 — Multi-Task Learning (DR + Lesion + Quality)")
    logger.info("Expected gain: +2–4% AUC | Single model, DR head at inference")
    logger.info("=" * 70)

    full_ds, train_ds, val_ds = build_splits(data_dir)
    train_ds.dataset.transform = get_advanced_augmentation_transforms(augment=True)
    val_ds.dataset.transform   = get_augmentation_transforms(augment=False)

    # Wrap in multi-task dataset
    mt_train = MultiTaskDRDataset(train_ds.dataset)
    mt_val   = MultiTaskDRDataset(val_ds.dataset)

    train_loader = DataLoader(
        mt_train, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        collate_fn=multitask_collate_fn
    )
    val_loader = DataLoader(
        mt_val, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
        collate_fn=multitask_collate_fn
    )

    trainer = MultiTaskTrainer(device=device)
    trainer.train(
        train_loader, val_loader,
        epochs=epochs, lr=lr,
        w_dr=0.6, w_lesion=0.2, w_quality=0.2,
        save_path='checkpoints/multitask_dr_model.pth'
    )
    logger.info("Improvement 3 complete → checkpoints/multitask_dr_model.pth")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Run DR Model Improvements')

    parser.add_argument('--data_dir',   type=str,   default='./data')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--device',     type=str,   default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--all',      action='store_true', help='Run all 3 improvements sequentially')
    parser.add_argument('--augment',  action='store_true', help='Run improvement 1 (RandAugment + MixUp)')
    parser.add_argument('--distill',  action='store_true', help='Run improvement 2 (Knowledge Distillation)')
    parser.add_argument('--multitask',action='store_true', help='Run improvement 3 (Multi-Task Learning)')
    parser.add_argument('--eval',     action='store_true', help='Evaluate a checkpoint (requires --checkpoint)')
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)

    if args.eval:
        if not args.checkpoint:
            logger.error("--eval requires --checkpoint <path>")
            return
        full_ds, _, val_ds = build_splits(args.data_dir)
        val_ds.dataset.transform = get_augmentation_transforms(augment=False)
        evaluate_model(args.checkpoint, val_ds, args.device, label=args.checkpoint)
        return

    run_any = args.all or args.augment or args.distill or args.multitask
    if not run_any:
        parser.print_help()
        return

    if args.all or args.augment:
        run_augmentation(args.data_dir, args.epochs, args.batch_size, args.lr, args.device)

    if args.all or args.distill:
        run_distillation(args.data_dir, args.epochs, args.batch_size, args.lr, args.device)

    if args.all or args.multitask:
        run_multitask(args.data_dir, args.epochs, args.batch_size, args.lr, args.device)

    logger.info("=" * 70)
    logger.info("ALL IMPROVEMENTS COMPLETE")
    logger.info("Checkpoints saved in ./checkpoints/")
    logger.info("  augmented_model.pth        (Improvement 1)")
    logger.info("  student_b4_distilled.pth   (Improvement 2)")
    logger.info("  multitask_dr_model.pth     (Improvement 3)")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
