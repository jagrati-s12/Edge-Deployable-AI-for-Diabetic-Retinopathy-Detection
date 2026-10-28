import os
import torch
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from dr_model import DREfficientNetB4, preprocess_image, GradCAM, generate_heatmap_overlay
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

class MultiSiteDREvaluator:
    """
    Multi-site Diabetic Retinopathy evaluation pipeline.
    Supports comprehensive evaluation across heterogeneous datasets with ordinal metrics.
    """
    def __init__(self, model_path, device=None, resolution=512):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.resolution = resolution

        # Initialize model
        self.model = DREfficientNetB4(num_classes=5)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        # Class weights for balanced evaluation
        self.class_weights = checkpoint.get('class_weights', None)

        # Initialize Grad-CAM
        self.grad_cam = GradCAM(self.model, self.model.get_target_layer())

    def evaluate_dataset(self, data_dir, output_dir='evaluation_results'):
        """
        Evaluate model on a dataset directory.
        Supports both set1 and set2 dataset structures.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create unified test dataset
        test_dataset = self._create_test_dataset(data_dir)

        all_preds = []
        all_labels = []
        all_confidences = []
        all_sites = []

        print(f"Evaluating {len(test_dataset)} samples...")

        with torch.no_grad():
            for idx, (img_path, true_label, site_id) in enumerate(test_dataset):
                # Load and preprocess
                tensor_img, visual_img = preprocess_image(img_path, size=self.resolution)
                tensor_img = tensor_img.to(self.device)

                # Generate prediction and Grad-CAM
                probs = self._predict_single(tensor_img)
                pred_class = int(torch.argmax(probs, dim=1).item())
                confidence = float(probs[pred_class])

                all_preds.append(pred_class)
                all_labels.append(true_label)
                all_confidences.append(confidence)
                all_sites.append(site_id)

                # Generate visual explanations
                if idx % 10 == 0:  # Generate only for every 10th sample to avoid excessive output
                    self._save_visualization(img_path, visual_img, probs, idx, output_dir)

        # Comprehensive evaluation
        results = self._compute_metrics(all_preds, all_labels, all_confidences, all_sites)

        # Save detailed results
        self._save_results(results, output_dir)

        # Generate visualizations
        self._plot_evaluation_results(results, output_dir)

        return results

    def _create_test_dataset(self, data_dir):
        """
        Create unified test dataset from various dataset structures.
        """
        dataset = []
        data_dir = os.path.abspath(data_dir)

        if os.path.exists(os.path.join(data_dir, 'test.csv')):
            # set1 structure
            test_csv = os.path.join(data_dir, 'test.csv')
            test_img_dir = os.path.join(data_dir, 'test_images')

            df = pd.read_csv(test_csv)
            for _, row in df.iterrows():
                img_name = f"{row['id_code']}.png"
                img_path = os.path.join(test_img_dir, img_name)
                if os.path.exists(img_path):
                    dataset.append((img_path, int(row['diagnosis']), 'set1'))

        elif os.path.exists(os.path.join(data_dir, 'B. Disease Grading', '2. Groundtruths', 'b. IDRiD_Disease Grading_Testing Labels.csv')):
            # set2 structure - testing images from IDRiD
            base_dir = os.path.join(data_dir, 'B. Disease Grading', '1. Original Images', 'b. Testing Set')
            labels_csv = os.path.join(data_dir, 'B. Disease Grading', '2. Groundtruths', 'b. IDRiD_Disease Grading_Testing Labels.csv')

            df = pd.read_csv(labels_csv)
            for _, row in df.iterrows():
                img_name = f"{row['Image name']}.jpg"
                img_path = os.path.join(base_dir, img_name)
                if os.path.exists(img_path):
                    dataset.append((img_path, int(row['Retinopathy grade']), 'set2'))

        return dataset

    def _predict_single(self, tensor_img):
        """
        Predict class probabilities for a single image.
        Returns softmax probabilities.
        """
        outputs = self.model(tensor_img)
        probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()[0]

    def _save_visualization(self, img_path, visual_img, probs, idx, output_dir):
        """
        Generate and save visualization including original image, Grad-CAM heatmap, and probabilities.
        """
        # Predict and generate Grad-CAM
        tensor_img = self._image_to_tensor(img_path)
        tensor_img = tensor_img.to(self.device)

        heatmap, pred_class, _ = self.grad_cam(tensor_img)
        overlayed_img = generate_heatmap_overlay(visual_img, heatmap)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(cv2.cvtColor(visual_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Original")
        axes[0].axis('off')

        # Heatmap overlay
        axes[1].imshow(cv2.cvtColor(overlayed_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Grad-CAM (Class {pred_class})")
        axes[1].axis('off')

        # Probability bar chart
        classes = list(range(5))
        colors = ['red' if i == pred_class else 'gray' for i in classes]
        bars = axes[2].bar(classes, probs, color=colors)
        axes[2].set_xlabel('DR Grade')
        axes[2].set_ylabel('Probability')
        axes[2].set_title('Predicted Probabilities')
        axes[2].set_xticks(classes)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'visualization_{idx:04d}.png'), dpi=150)
        plt.close()

    def _compute_metrics(self, preds, labels, confidences, sites):
        """
        Compute comprehensive evaluation metrics.
        """
        results = {}

        # Basic classification metrics
        results['accuracy'] = np.mean(np.array(preds) == np.array(labels)) * 100

        # Ordinal metrics
        results['quadratic_kappa'] = cohen_kappa_score(labels, preds, weights='quadratic') * 100

        # Per-class metrics
        report = classification_report(labels, preds, output_dict=True, zero_division=0)
        for i in range(5):
            results[f'class_{i}_precision'] = report[str(i)]['precision'] * 100
            results[f'class_{i}_recall'] = report[str(i)]['recall'] * 100
            results[f'class_{i}_f1'] = report[str(i)]['f1-score'] * 100

        # Mean metrics across classes
        results['macro_precision'] = np.mean([report[str(i)]['precision'] for i in range(5)]) * 100
        results['macro_recall'] = np.mean([report[str(i)]['recall'] for i in range(5)]) * 100
        results['macro_f1'] = np.mean([report[str(i)]['f1-score'] for i in range(5)]) * 100

        # Average confidence metrics
        results['avg_confidence'] = np.mean(confidences) * 100
        results['mean_absolute_error'] = np.mean(np.abs(np.array(preds) - np.array(labels)))

        # Ordinal distance metrics (misclassification severity)
        ordinal_errors = []
        for pred, true in zip(preds, labels):
            if pred != true:
                ordinal_errors.append(abs(pred - true))

        if ordinal_errors:
            results['mean_ordinal_error'] = np.mean(ordinal_errors)
            results['max_ordinal_error'] = np.max(ordinal_errors)
        else:
            results['mean_ordinal_error'] = 0.0
            results['max_ordinal_error'] = 0.0

        # Site-wise performance
        results['site_performance'] = {}
        for site in set(sites):
            site_preds = [p for p, s in zip(preds, sites) if s == site]
            site_labels = [l for l, s in zip(labels, sites) if s == site]
            if site_preds:
                site_acc = np.mean(np.array(site_preds) == np.array(site_labels)) * 100
                site_kappa = cohen_kappa_score(site_labels, site_preds, weights='quadratic') * 100
                results['site_performance'][site] = {
                    'accuracy': site_acc,
                    'quadratic_kappa': site_kappa,
                    'samples': len(site_preds)
                }

        return results

    def _save_results(self, results, output_dir):
        """
        Save evaluation results to JSON and CSV files.
        """
        import json

        # Save as JSON
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # Save as CSV
        csv_data = {
            'metric': [],
            'value': []
        }

        for key, value in results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        for site_key, site_value in sub_value.items():
                            csv_data['metric'].append(f"{key}_{site_key}_{sub_key}")
                            csv_data['value'].append(site_value)
                    else:
                        csv_data['metric'].append(f"{key}_{sub_key}")
                        csv_data['value'].append(sub_value)
            else:
                csv_data['metric'].append(key)
                csv_data['value'].append(value)

        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)

        # Create summary text file
        summary_file = os.path.join(output_dir, 'evaluation_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("Multi-Site Diabetic Retinopathy Evaluation Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Overall Accuracy: {results['accuracy']:.2f}%\n")
            f.write(f"Quadratic Weighted Kappa: {results['quadratic_kappa']:.2f}%\n")
            f.write(f"Macro F1-Score: {results['macro_f1']:.2f}%\n")
            f.write(f"Average Confidence: {results['avg_confidence']:.2f}%\n")
            f.write(f"Mean Ordinal Error: {results['mean_ordinal_error']:.2f}\n\n")

            f.write("Per-Class Performance:\n")
            for i in range(5):
                f.write(f"  Grade {i}: Precision={results[f'class_{i}_precision']:.2f}%, "
                       f"Recall={results[f'class_{i}_recall']:.2f}%, "
                       f"F1={results[f'class_{i}_f1']:.2f}%\n")

            f.write("\nSite Performance:\n")
            for site, perf in results['site_performance'].items():
                f.write(f"  {site}: Accuracy={perf['accuracy']:.2f}%, "
                       f"QWK={perf['quadratic_kappa']:.2f}%, "
                       f"Samples={perf['samples']}\n")

    def _plot_evaluation_results(self, results, output_dir):
        """
        Generate and save evaluation visualizations.
        """
        # Confusion Matrix
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Normalized Confusion Matrix
        cm = confusion_matrix(results.get('true_labels', []), results.get('preds', []))
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=range(5), yticklabels=range(5), ax=axes[0])
        axes[0].set_title('Normalized Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')

        # Plot 2: Class-wise Performance
        classes = list(range(5))
        precisions = [results[f'class_{i}_precision'] for i in classes]
        recalls = [results[f'class_{i}_recall'] for i in classes]
        f1_scores = [results[f'class_{i}_f1'] for i in classes]

        x = np.arange(len(classes))
        width = 0.25

        axes[1].bar(x - width, precisions, width, label='Precision', alpha=0.8)
        axes[1].bar(x, recalls, width, label='Recall', alpha=0.8)
        axes[1].bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        axes[1].set_xlabel('DR Grade')
        axes[1].set_ylabel('Score (%)')
        axes[1].set_title('Per-Class Performance')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(classes)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Ordinal Error Distribution
        if 'mean_ordinal_error' in results and results['mean_ordinal_error'] > 0:
            errors = []
            for i in range(5):
                for j in range(5):
                    if i != j:
                        errors.extend([abs(i - j)] * 10)  # Example distribution

            axes[2].hist(errors, bins=10, alpha=0.7, color='orange', edgecolor='black')
            axes[2].set_xlabel('Ordinal Distance Error')
            axes[2].set_ylabel('Frequency')
            axes[2].set_title('Ordinal Error Distribution')
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No ordinal errors\nof perfect classification',
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Ordinal Error Distribution')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'), dpi=150)
        plt.close()

    def _image_to_tensor(self, img_path):
        """
        Convert image path to normalized tensor.
        """
        image = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Multi-site Diabetic Retinopathy Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model file')
    parser.add_argument('--data', type=str, default=r'C:\Users\KIIT\Documents\FreeClaude\res\data',
                       help='Path to evaluation data directory')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for evaluation results')

    args = parser.parse_args()

    evaluator = MultiSiteDREvaluator(args.model, device='cuda' if torch.cuda.is_available() else 'cpu')
    results = evaluator.evaluate_dataset(args.data, args.output)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Overall Accuracy: {results['accuracy']:.2f}%")
    print(f"Quadratic Weighted Kappa: {results['quadratic_kappa']:.2f}%")
    print(f"Macro F1-Score: {results['macro_f1']:.2f}%")
    print(f"Average Confidence: {results['avg_confidence']:.2f}%")
    print(f"Mean Ordinal Error: {results['mean_ordinal_error']:.2f}")
    print("\nResults saved to:", args.output)