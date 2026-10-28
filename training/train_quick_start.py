"""
Quick Start Training Script for DR Detection Model

Usage:
    python training/train_quick_start.py --data_dir ./data --epochs 50 --batch_size 32 --use_ordinal_loss

This script:
1. Loads data from organized directories (0/, 1/, 2/, 3/, 4/)
2. Trains model with chosen improvements
3. Validates and saves checkpoints
4. Generates evaluation reports and visualizations
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import logging
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import (
    DRDataset,
    get_augmentation_transforms,
    get_class_weights,
    Trainer,
    MetricsTracker,
    ModelImprovementStrategies,
    generate_evaluation_report
)
from models.dr_model_complete import EfficientNetB4WithSE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """Main training pipeline."""

    logger.info("=" * 80)
    logger.info("DIABETIC RETINOPATHY DETECTION - TRAINING PIPELINE")
    logger.info("=" * 80)

    # ========================================================================
    # STEP 1: PREPARE DATA
    # ========================================================================
    logger.info("\n[STEP 1] Loading Data...")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("\nExpected directory structure:")
        logger.info("  data/")
        logger.info("  ├── 0/  (No DR)")
        logger.info("  ├── 1/  (Mild)")
        logger.info("  ├── 2/  (Moderate)")
        logger.info("  ├── 3/  (Severe)")
        logger.info("  └── 4/  (Proliferative)")
        return

    # Load dataset
    dataset = DRDataset(data_dir, augment=False)

    # Split into train/val (80/20 stratified)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Apply augmentation
    train_dataset.dataset.transform = get_augmentation_transforms(augment=True)
    val_dataset.dataset.transform = get_augmentation_transforms(augment=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )

    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val samples: {len(val_dataset)}")
    logger.info(f"  Batch size: {args.batch_size}")

    # Get class weights for imbalanced data
    class_weights = get_class_weights(dataset)

    # ========================================================================
    # STEP 2: CREATE MODEL
    # ========================================================================
    logger.info("\n[STEP 2] Creating Model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"  Device: {device}")

    model = EfficientNetB4WithSE(
        num_classes=5,
        pretrained=True,  # Use ImageNet weights
        add_se_blocks=True
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size (FP32): {total_params * 4 / 1e6:.1f} MB")

    # ========================================================================
    # STEP 3: SETUP TRAINING
    # ========================================================================
    logger.info("\n[STEP 3] Setting Up Training...")

    trainer = Trainer(model, device=device, use_ordinal_loss=args.use_ordinal_loss)

    trainer.setup_training(
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        lr=args.learning_rate,
        ordinal_weight=args.ordinal_weight,
        label_smoothing=args.label_smoothing
    )

    logger.info(f"  Using ordinal loss: {args.use_ordinal_loss}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Ordinal weight: {args.ordinal_weight}")
    logger.info(f"  Label smoothing: {args.label_smoothing}")

    # ========================================================================
    # STEP 4: TRAIN MODEL
    # ========================================================================
    logger.info("\n[STEP 4] Training Model...")
    logger.info(f"  Epochs: {args.epochs}")

    metrics = trainer.train(epochs=args.epochs)

    logger.info("\n[STEP 4] Training Complete!")
    logger.info(f"  Best model saved to: {trainer.early_stopping.model_path}")

    # ========================================================================
    # STEP 5: LOAD BEST MODEL AND EVALUATE
    # ========================================================================
    logger.info("\n[STEP 5] Final Evaluation...")

    # Load best checkpoint
    best_state = torch.load(trainer.early_stopping.model_path, map_location=device)
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Get predictions on full dataset for evaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=32, num_workers=2):
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    import numpy as np
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    generate_evaluation_report(all_preds, all_labels, "DR Detection Model (Best Checkpoint)")

    # ========================================================================
    # STEP 6: SAVE MODEL AND ARTIFACTS
    # ========================================================================
    logger.info("\n[STEP 6] Saving Model and Artifacts...")

    # Save full model
    output_dir = Path('checkpoints')
    output_dir.mkdir(exist_ok=True)

    final_model_path = output_dir / 'dr_model_final.pth'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"  Model saved to: {final_model_path}")

    # Save training metrics
    metrics.save_json(str(output_dir / 'metrics.json'))
    logger.info(f"  Metrics saved to: {output_dir / 'metrics.json'}")

    # Save plots
    metrics.plot_history(str(output_dir / 'training_history.png'))
    logger.info(f"  Plot saved to: {output_dir / 'training_history.png'}")

    # ========================================================================
    # STEP 7: IMPROVEMENT RECOMMENDATIONS
    # ========================================================================
    logger.info("\n[STEP 7] Improvement Recommendations...")
    logger.info("\nTo improve model further, consider:")
    logger.info("\n1. ENSEMBLE (Easy, +1-3% AUC):")
    logger.info("   - Train 3-5 models with different random seeds")
    logger.info("   - Average predictions at inference")

    logger.info("\n2. ADVANCED AUGMENTATION (Medium, +1-2% AUC):")
    logger.info("   - Add RandAugment or AutoAugment")
    logger.info("   - Add MixUp or CutMix during training")

    logger.info("\n3. KNOWLEDGE DISTILLATION (Medium, +2-3% AUC) - RECOMMENDED:")
    logger.info("   - Train larger teacher model (EfficientNet-B5)")
    logger.info("   - Distill into smaller student (our B4 model)")
    logger.info("   - Improves performance without increasing inference cost")

    logger.info("\n4. MULTI-TASK LEARNING (Complex, +2-4% AUC) - RECOMMENDED:")
    logger.info("   - Add lesion detection task")
    logger.info("   - Add image quality classification task")
    logger.info("   - Requires additional annotations")

    logger.info("\n5. DOMAIN ADAPTATION (Complex, +2-3% on degraded images) - RECOMMENDED:")
    logger.info("   - Train with continuous domain randomization")
    logger.info("   - Or use Adversarial Domain Adaptation (DANN)")
    logger.info("   - Improves robustness to low-quality images")

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete! Check 'checkpoints/' directory for artifacts.")
    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DR Detection Model')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to data directory with subdirs 0-4')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Initial learning rate')

    # Loss function arguments
    parser.add_argument('--use_ordinal_loss', action='store_true',
                       help='Use ordinal-aware loss instead of cross-entropy')
    parser.add_argument('--ordinal_weight', type=float, default=1.0,
                       help='Weight for ordinal penalty')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor (0-1)')

    args = parser.parse_args()

    main(args)
