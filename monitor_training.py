"""
Monitor training and stop at epoch 25, then generate report.
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from dr_model import DREfficientNetB4, preprocess_image, generate_heatmap_overlay
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
import cv2
import base64

def wait_for_checkpoint(target_epoch=25, check_interval=30, timeout=7200):
    """
    Wait for training to reach target epoch and save checkpoint.

    Args:
        target_epoch: Stop at this epoch
        check_interval: Check every N seconds
        timeout: Max wait time in seconds
    """
    print(f"\n{'='*80}")
    print(f"MONITORING TRAINING - WILL STOP AT EPOCH {target_epoch}")
    print(f"{'='*80}\n")

    checkpoint_path = 'dr_model_best.pth'
    history_path = 'training_history.json'

    start_time = time.time()
    epoch_reached = False

    while not epoch_reached:
        elapsed = time.time() - start_time

        # Check if history file exists and has data
        if Path(history_path).exists():
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)

                current_epochs = len(history.get('val_acc', []))

                if current_epochs > 0:
                    last_epoch = current_epochs
                    last_train_loss = history['train_loss'][-1] if history['train_loss'] else None
                    last_val_loss = history['val_loss'][-1] if history['val_loss'] else None
                    last_train_acc = history['train_acc'][-1] if history['train_acc'] else None
                    last_val_acc = history['val_acc'][-1] if history['val_acc'] else None
                    last_qwk = history['val_qwk'][-1] if history['val_qwk'] else None

                    print(f"[{time.strftime('%H:%M:%S')}] Epoch {last_epoch}/{target_epoch}")
                    print(f"  Train Loss: {last_train_loss:.4f} → Val Loss: {last_val_loss:.4f}")
                    print(f"  Train Acc: {last_train_acc:.1f}% → Val Acc: {last_val_acc:.1f}%")
                    if last_qwk is not None:
                        print(f"  QWK: {last_qwk:.3f}")
                    print()

                    if last_epoch >= target_epoch:
                        epoch_reached = True
                        print(f"✓ Reached epoch {target_epoch}!")
                        break

            except Exception as e:
                pass

        # Check timeout
        if elapsed > timeout:
            print(f"⚠ Timeout reached ({elapsed/60:.1f} min). Stopping.")
            break

        time.sleep(check_interval)

    return history if Path(history_path).exists() else None

def generate_midpoint_report(history, checkpoint_path='dr_model_best.pth'):
    """
    Generate evaluation report at epoch 25 midpoint.
    """
    if history is None:
        print("⚠ No training history found")
        return

    print(f"\n{'='*80}")
    print("EPOCH 25 MIDPOINT REPORT")
    print(f"{'='*80}\n")

    # Training curves
    train_losses = history.get('train_loss', [])
    val_losses = history.get('val_loss', [])
    train_accs = history.get('train_acc', [])
    val_accs = history.get('val_acc', [])
    val_qwks = history.get('val_qwk', [])

    epochs_completed = len(train_accs)

    print(f"Epochs completed: {epochs_completed}/50")
    print(f"Training progress: {epochs_completed/50*100:.1f}%\n")

    print("LOSS PROGRESSION:")
    print(f"  Initial Train Loss: {train_losses[0]:.4f}")
    print(f"  Final Train Loss: {train_losses[-1]:.4f}")
    print(f"  Improvement: {(train_losses[0] - train_losses[-1])/train_losses[0]*100:.1f}%\n")

    print("ACCURACY PROGRESSION:")
    print(f"  Initial Train Acc: {train_accs[0]:.1f}%")
    print(f"  Final Train Acc: {train_accs[-1]:.1f}%")
    print(f"  Initial Val Acc: {val_accs[0]:.1f}%")
    print(f"  Final Val Acc: {val_accs[-1]:.1f}%")
    print(f"  Improvement: {val_accs[-1] - val_accs[0]:.1f}% points\n")

    if val_qwks:
        print(f"QUADRATIC WEIGHTED KAPPA:")
        print(f"  Initial QWK: {val_qwks[0]:.3f}")
        print(f"  Final QWK: {val_qwks[-1]:.3f}\n")

    print("TREND ANALYSIS:")
    if len(train_accs) > 5:
        recent_improvement = train_accs[-1] - train_accs[-6]
        print(f"  Last 5 epochs improvement: {recent_improvement:.1f}%")

        if recent_improvement > 0.5:
            print(f"  Status: ✓ Model still improving strongly")
        elif recent_improvement > 0.1:
            print(f"  Status: ⚠ Model improving slowly")
        else:
            print(f"  Status: ⚠ Model convergence may be near")

    print(f"\n{'='*80}\n")

    # Save summary
    summary = {
        'epochs_completed': epochs_completed,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'final_qwk': val_qwks[-1] if val_qwks else None,
        'improvement_from_baseline': (val_accs[-1] - val_accs[0]) if val_accs else 0
    }

    with open('epoch_25_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("✓ Summary saved to epoch_25_summary.json\n")

if __name__ == '__main__':
    history = wait_for_checkpoint(target_epoch=25)
    generate_midpoint_report(history)
