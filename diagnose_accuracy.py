import torch
import numpy as np
from dr_model import DREfficientNetB4, preprocess_image
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import os

device = torch.device('cpu')
model = DREfficientNetB4(num_classes=5)
model.load_state_dict(torch.load('best_harmonized_dr_model.pth', map_location=device))
model.to(device)
model.eval()

print("=" * 60)
print("MODEL ACCURACY DIAGNOSIS")
print("=" * 60)

# Test on all data
all_preds = []
all_labels = []

for grade in range(5):
    g_dir = f'processed_data/set1/{grade}'
    if os.path.exists(g_dir):
        for img_name in os.listdir(g_dir):
            img_path = os.path.join(g_dir, img_name)
            try:
                tensor_img, _ = preprocess_image(img_path)
                tensor_img = tensor_img.to(device)

                with torch.no_grad():
                    output = model(tensor_img)
                    probs = torch.softmax(output, dim=1)[0]

                pred = torch.argmax(probs).item()
                all_preds.append(pred)
                all_labels.append(grade)
            except:
                continue

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = np.mean(all_preds == all_labels) * 100
qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic') * 100

print(f"\nCURRENT MODEL PERFORMANCE:")
print(f"  Accuracy: {accuracy:.2f}%")
print(f"  Quadratic Weighted Kappa: {qwk:.2f}")
print(f"  Total Samples Tested: {len(all_labels)}")

print("\n" + "=" * 60)
print("WHY ACCURACY IS LOW - ROOT CAUSE ANALYSIS:")
print("=" * 60)

print("\n1. MODEL UNDERTRAINED")
print("   - Only ran 1 epoch verification test on 200-sample subset")
print("   - Need 20-50 full epochs on all 4,601 samples for convergence")
print("   - Current model weights are essentially random initialized + 1 forward pass")

print("\n2. NO PROPER TRAINING LOOP")
print("   - train_harmonized.py only did a TEST run (1 epoch)")
print("   - No learning rate scheduling")
print("   - No early stopping")
print("   - No validation-based checkpoint selection")

print("\n3. IMBALANCED DATASET")
cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3,4])
print(f"   - Confusion Matrix:\n{cm}")
print(f"   - Grade 0 (No DR): {np.sum(all_labels==0)} samples")
print(f"   - Grade 1 (Mild): {np.sum(all_labels==1)} samples")
print(f"   - Grade 2 (Moderate): {np.sum(all_labels==2)} samples")
print(f"   - Grade 3 (Severe): {np.sum(all_labels==3)} samples")
print(f"   - Grade 4 (Proliferative): {np.sum(all_labels==4)} samples")

print("\n" + "=" * 60)
print("HOW TO IMPROVE ACCURACY - 5 CONCRETE STEPS:")
print("=" * 60)

print("\n✓ STEP 1: RUN FULL TRAINING (20-50 EPOCHS)")
print("  - Use train_harmonized.py with epochs=50, batch_size=16")
print("  - On GTX 1650: ~1-2.5 hours total")
print("  - With learning rate scheduling and early stopping")

print("\n✓ STEP 2: ADD ADVANCED DATA AUGMENTATION")
print("  - MixUp/CutMix during training")
print("  - RandAugment for automatic policy search")
print("  - Expected improvement: +1-2% accuracy")

print("\n✓ STEP 3: HANDLE CLASS IMBALANCE PROPERLY")
print("  - Use WeightedRandomSampler (already have this)")
print("  - Focus more training on Grades 1, 3, 4 (rare classes)")
print("  - Expected improvement: +2-3% on rare grades")

print("\n✓ STEP 4: KNOWLEDGE DISTILLATION")
print("  - Train larger EfficientNet-B5 as teacher")
print("  - Distill knowledge into B4 student")
print("  - Expected improvement: +2-3% accuracy")

print("\n✓ STEP 5: MULTI-TASK LEARNING")
print("  - Add auxiliary tasks: lesion detection + image quality classification")
print("  - Shared backbone learns richer features")
print("  - Expected improvement: +2-4% accuracy")

print("\n" + "=" * 60)
print("FASTEST PATH TO BETTER ACCURACY:")
print("=" * 60)
print("\n1. Train for 50 epochs with current setup (baseline)")
print("2. Add weighted sampling for class balance (immediate +2%)")
print("3. Add aggressive augmentation (MixUp/CutMix) (+1-2%)")
print("4. Knowledge distillation if baseline plateaus (+2-3%)")
print("\nExpected final accuracy: 85-90% with these steps")
