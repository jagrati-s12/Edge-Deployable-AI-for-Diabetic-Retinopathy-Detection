"""
Unified Data Preprocessor for Multi-Site Diabetic Retinopathy Detection
Handles set1 (Kaggle-like) and set2 (IDRiD) dataset structures
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import json
import shutil
from typing import List, Tuple, Dict, Optional
import hashlib
from collections import Counter

class UnifiedDataPreprocessor:
    """
    Unified data preprocessor that handles multiple dataset structures
    and creates standardized input for training.
    """

    def __init__(self, output_dir: str = "data", verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.dataset_stats = {}

        # Create standardized directory structure
        self.std_structure = {
            '0': 'No DR (0)',
            '1': 'Mild NPDR (1)',
            '2': 'Moderate NPDR (2)',
            '3': 'Severe NPDR (3)',
            '4': 'Proliferative DR (4)'
        }

    def create_standard_dataset(self, source_dir: str, dataset_name: str,
                               output_dir: Optional[str] = None) -> Dict:
        """
        Create standardized dataset from a source directory.

        Args:
            source_dir: Path to source dataset
            dataset_name: Name of the dataset (set1, set2, etc.)
            output_dir: Custom output directory (optional)

        Returns:
            Dictionary with dataset statistics
        """
        if output_dir is None:
            output_dir = self.output_dir / dataset_name
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        for grade in self.std_structure.keys():
            (output_dir / grade).mkdir(parents=True, exist_ok=True)

        stats = {
            'dataset_name': dataset_name,
            'source_dir': source_dir,
            'output_dir': str(output_dir),
            'classes': {},
            'total_images': 0,
            'class_distribution': {},
            'preprocessing_stats': {}
        }

        # Detect dataset structure and process accordingly
        if self._is_kaggle_style(source_dir):
            stats = self._process_kaggle_style(source_dir, output_dir, stats)
        elif self._is_idrid_style(source_dir):
            stats = self._process_idrid_style(source_dir, output_dir, stats)
        else:
            raise ValueError(f"Unknown dataset structure in {source_dir}")

        self._update_global_stats(stats)
        return stats

    def _is_kaggle_style(self, source_dir: str) -> bool:
        """Check if dataset is Kaggle-style (has train.csv)."""
        csv_path = Path(source_dir) / 'train.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return 'diagnosis' in df.columns and 'image' in df.columns
        return False

    def _is_idrid_style(self, source_dir: str) -> bool:
        """Check if dataset is IDRiD-style (has structured directories)."""
        disease_dir = Path(source_dir) / 'B. Disease Grading' / '2. Groundtruths'
        if disease_dir.exists():
            csv_files = list(disease_dir.glob('*.csv'))
            if csv_files:
                df = pd.read_csv(csv_files[0])
                return 'Retinopathy grade' in df.columns
        return False

    def _process_kaggle_style(self, source_dir: str, output_dir: Path,
                            stats: Dict) -> Dict:
        """Process Kaggle-style dataset (set1)."""
        csv_path = Path(source_dir) / 'train.csv'
        img_dir = Path(source_dir) / 'train_images'

        if self.verbose:
            print(f"Processing Kaggle-style dataset from {source_dir}")

        df = pd.read_csv(csv_path)

        for idx, row in df.iterrows():
            try:
                # Handle image paths (could be just filename or path)
                img_name = str(row['image'])
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_name = f"{img_name}.png"

                src_path = img_dir / img_name if img_dir.exists() else Path(source_dir) / img_name

                if not src_path.exists():
                    if self.verbose:
                        print(f"Warning: Image not found: {src_path}")
                    continue

                # Load and preprocess image
                processed_img = self._preprocess_image(str(src_path))

                # Save to standardized structure
                grade = str(int(row['diagnosis']))
                if grade not in self.std_structure:
                    if self.verbose:
                        print(f"Warning: Unknown diagnosis grade {grade}, skipping")
                    continue

                # Generate unique filename to avoid conflicts
                img_hash = hashlib.md5(str(src_path).encode()).hexdigest()[:8]
                filename = f"{img_hash}_{Path(src_path).name}"

                dest_path = output_dir / grade / filename
                cv2.imwrite(str(dest_path), processed_img)

                # Update stats
                if grade not in stats['classes']:
                    stats['classes'][grade] = 0
                stats['classes'][grade] += 1
                stats['total_images'] += 1

            except Exception as e:
                if self.verbose:
                    print(f"Error processing row {idx}: {e}")
                continue

        stats['class_distribution'] = {
            grade: stats['classes'].get(grade, 0)
            for grade in self.std_structure.keys()
        }

        return stats

    def _process_idrid_style(self, source_dir: str, output_dir: Path,
                           stats: Dict) -> Dict:
        """Process IDRiD-style dataset (set2)."""
        # Use the disease grading directory structure
        disease_dir = Path(source_dir) / 'B. Disease Grading'
        img_dir = disease_dir / '1. Original Images' / 'a. Training Set'
        labels_csv = disease_dir / '2. Groundtruths' / 'a. IDRiD_Disease Grading_Training Labels.csv'

        if self.verbose:
            print(f"Processing IDRiD-style dataset from {source_dir}")

        df = pd.read_csv(labels_csv)

        for idx, row in df.iterrows():
            try:
                img_name = f"{row['Image name']}.jpg"
                src_path = img_dir / img_name

                if not src_path.exists():
                    if self.verbose:
                        print(f"Warning: Image not found: {src_path}")
                    continue

                # Load and preprocess image
                processed_img = self._preprocess_image(str(src_path))

                # Save to standardized structure
                grade = str(int(row['Retinopathy grade']))
                if grade not in self.std_structure:
                    if self.verbose:
                        print(f"Warning: Unknown grade {grade}, skipping")
                    continue

                # Generate unique filename
                img_hash = hashlib.md5(str(src_path).encode()).hexdigest()[:8]
                filename = f"{img_hash}_{row['Image name']}.jpg"

                dest_path = output_dir / grade / filename
                cv2.imwrite(str(dest_path), processed_img)

                # Update stats
                if grade not in stats['classes']:
                    stats['classes'][grade] = 0
                stats['classes'][grade] += 1
                stats['total_images'] += 1

                # Store additional metadata (risk score, etc.)
                if 'Risk of macular edema' in row:
                    risk = int(row['Risk of macular edema'])
                    stats['risk_stats'] = stats.get('risk_stats', {0: 0, 1: 0})
                    stats['risk_stats'][risk] += 1

            except Exception as e:
                if self.verbose:
                    print(f"Error processing row {idx}: {e}")
                continue

        stats['class_distribution'] = {
            grade: stats['classes'].get(grade, 0)
            for grade in self.std_structure.keys()
        }

        return stats

    def _preprocess_image(self, img_path: str) -> np.ndarray:
        """
        Apply consistent preprocessing pipeline to all images.
        Same pipeline as in dr_model.py for consistency.
        """
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")

            # Convert to RGB (dr_model.py expects RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Auto crop dark borders (same as dr_model.py)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            tol = 7
            mask = gray > tol

            if img.ndim == 3:
                check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
                if check_shape == 0:
                    return img
                else:
                    img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
                    img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
                    img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
                    img = np.stack([img1,img2,img3], axis=-1)

            # Extract green channel for microaneurysm/hemorrhage visibility
            green = img[:, :, 1]

            # Adaptive CLAHE Enhancement (same parameters as dr_model.py)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            green_clahe = clahe.apply(green)

            # Reintegrate enhanced green channel
            img[:, :, 1] = green_clahe

            # Resize to standard size (same as dr_model.py)
            img = cv2.resize(img, (512, 512))

            return img

        except Exception as e:
            print(f"Error preprocessing {img_path}: {e}")
            # Return a blank image as fallback
            return np.zeros((512, 512, 3), dtype=np.uint8)

    def create_train_test_split(self, dataset_dir: str, output_dir: str,
                                train_ratio: float = 0.8,
                                stratify_by_class: bool = True) -> Dict:
        """
        Create train/test split with stratification.

        Args:
            dataset_dir: Directory with class folders
            output_dir: Output directory for split datasets
            train_ratio: Train/test ratio
            stratify_by_class: Whether to stratify by class

        Returns:
            Dictionary with split statistics
        """
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)

        # Collect all images and labels
        images = []
        labels = []

        for class_dir in dataset_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name not in self.std_structure:
                continue

            class_label = int(class_dir.name)
            for img_path in class_dir.glob('*'):
                if img_path.is_file() and img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    images.append(str(img_path))
                    labels.append(class_label)

        # Create stratified split
        if stratify_by_class and len(set(labels)) > 1:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                images, labels, test_size=1-train_ratio,
                random_state=42, stratify=labels
            )
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                images, labels, test_size=1-train_ratio,
                random_state=42
            )

        # Create directory structure for split datasets
        train_output = Path(output_dir) / 'train'
        val_output = Path(output_dir) / 'val'

        for split_dir, split_images, split_labels in [
            (train_output, X_train, y_train),
            (val_output, X_val, y_val)
        ]:
            for class_label in set(split_labels):
                (split_dir / str(class_label)).mkdir(parents=True, exist_ok=True)

            for img_path, label in zip(split_images, split_labels):
                # Copy image to split directory
                dest_name = Path(img_path).name
                dest_path = split_dir / str(label) / dest_name
                shutil.copy2(img_path, dest_path)

        # Save split metadata
        metadata = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'class_distribution': {
                'train': dict(Counter(y_train)),
                'val': dict(Counter(y_val))
            },
            'train_ratio': train_ratio,
            'stratified': stratify_by_class
        }

        with open(Path(output_dir) / 'split_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def _update_global_stats(self, stats: Dict):
        """Update global dataset statistics."""
        if not hasattr(self, 'global_stats'):
            self.global_stats = {}

        dataset_name = stats['dataset_name']
        self.global_stats[dataset_name] = stats

        if self.verbose:
            print(f"\n=== Dataset {dataset_name} Statistics ===")
            print(f"Total images: {stats['total_images']}")
            print(f"Class distribution: {stats['class_distribution']}")

    def save_configuration(self, save_path: str):
        """Save preprocessor configuration."""
        config = {
            'std_structure': self.std_structure,
            'global_stats': self.global_stats if hasattr(self, 'global_stats') else {}
        }

        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)

        if self.verbose:
            print(f"Configuration saved to {save_path}")

    def load_configuration(self, load_path: str):
        """Load preprocessor configuration."""
        with open(load_path, 'r') as f:
            config = json.load(f)

        self.std_structure = config['std_structure']
        self.global_stats = config.get('global_stats', {})

        if self.verbose:
            print(f"Configuration loaded from {load_path}")
if __name__ == "__main__":
    # Example usage
    print("=== Unified Data Preprocessor for DR Detection ===")

    # Initialize preprocessor
    preprocessor = UnifiedDataPreprocessor(verbose=True)

    # Process both datasets
    print("\nProcessing set1 (Kaggle-style)...")
    set1_stats = preprocessor.create_standard_dataset(
        source_dir=r"C:\Users\KIIT\Documents\FreeClaude\res\data\set1",
        dataset_name="set1"
    )

    print("\nProcessing set2 (IDRiD-style)...")
    set2_stats = preprocessor.create_standard_dataset(
        source_dir=r"C:\Users\KIIT\Documents\FreeClaude\res\data\set2",
        dataset_name="set2"
    )

    # Save configuration
    preprocessor.save_configuration("preprocessor_config.json")

    print("\n=== Preprocessing Complete ===")
    print(f"Set1: {set1_stats['total_images']} images")
    print(f"Set2: {set2_stats['total_images']} images")
    print(f"Total: {set1_stats['total_images'] + set2_stats['total_images']} images")