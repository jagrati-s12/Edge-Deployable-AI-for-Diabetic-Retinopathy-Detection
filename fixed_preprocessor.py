import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import shutil
from collections import Counter
import hashlib

class FixedUnifiedDataPreprocessor:
    """
    Fixed Unified Data Preprocessor supporting set1 (APTOS style: id_code, diagnosis)
    and set2 (IDRiD style: Image name, Retinopathy grade).
    """
    def __init__(self, output_dir: str = "data", verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.std_structure = {
            '0': 'No DR (0)',
            '1': 'Mild NPDR (1)',
            '2': 'Moderate NPDR (2)',
            '3': 'Severe NPDR (3)',
            '4': 'Proliferative DR (4)'
        }
        self.global_stats = {}

    def create_standard_dataset(self, source_dir: str, dataset_name: str) -> dict:
        source_dir = Path(source_dir)
        output_dir = self.output_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for grade in self.std_structure.keys():
            (output_dir / grade).mkdir(parents=True, exist_ok=True)

        stats = {
            'dataset_name': dataset_name,
            'source_dir': str(source_dir),
            'output_dir': str(output_dir),
            'classes': {},
            'total_images': 0
        }

        # Check set1 vs set2
        if (source_dir / 'train.csv').exists():
            stats = self._process_set1(source_dir, output_dir, stats)
        elif (source_dir / 'B. Disease Grading').exists():
            stats = self._process_set2(source_dir, output_dir, stats)
        else:
            raise ValueError(f"Unknown dataset structure in {source_dir}")

        self.global_stats[dataset_name] = stats
        return stats

    def _process_set1(self, source_dir: Path, output_dir: Path, stats: dict) -> dict:
        csv_path = source_dir / 'train.csv'
        img_dir = source_dir / 'train_images'
        df = pd.read_csv(csv_path)

        print(f"Processing set1 from {source_dir} ({len(df)} records)")
        for _, row in df.iterrows():
            try:
                img_name = f"{row['id_code']}.png"
                src_path = img_dir / img_name
                if not src_path.exists():
                    # try .jpg
                    src_path = img_dir / f"{row['id_code']}.jpg"
                if not src_path.exists():
                    continue

                grade = str(int(row['diagnosis']))
                if grade not in self.std_structure:
                    continue

                processed_img = self._preprocess_image(str(src_path))
                filename = f"set1_{row['id_code']}.jpg"
                dest_path = output_dir / grade / filename
                cv2.imwrite(str(dest_path), cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))

                stats['classes'][grade] = stats['classes'].get(grade, 0) + 1
                stats['total_images'] += 1
            except Exception as e:
                continue

        stats['class_distribution'] = {g: stats['classes'].get(g, 0) for g in self.std_structure.keys()}
        return stats

    def _process_set2(self, source_dir: Path, output_dir: Path, stats: dict) -> dict:
        img_dir = source_dir / 'B. Disease Grading' / '1. Original Images' / 'a. Training Set'
        labels_csv = source_dir / 'B. Disease Grading' / '2. Groundtruths' / 'a. IDRiD_Disease Grading_Training Labels.csv'
        df = pd.read_csv(labels_csv)

        print(f"Processing set2 (IDRiD) from {source_dir} ({len(df)} records)")
        for _, row in df.iterrows():
            try:
                img_name = f"{row['Image name']}.jpg"
                src_path = img_dir / img_name
                if not src_path.exists():
                    continue

                grade = str(int(row['Retinopathy grade']))
                if grade not in self.std_structure:
                    continue

                processed_img = self._preprocess_image(str(src_path))
                filename = f"set2_{row['Image name']}.jpg"
                dest_path = output_dir / grade / filename
                cv2.imwrite(str(dest_path), cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))

                stats['classes'][grade] = stats['classes'].get(grade, 0) + 1
                stats['total_images'] += 1
            except Exception as e:
                continue

        stats['class_distribution'] = {g: stats['classes'].get(g, 0) for g in self.std_structure.keys()}
        return stats

    def _preprocess_image(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Crop dark borders
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > 7
        if mask.any():
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)

        # Green channel CLAHE
        green = img[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img[:, :, 1] = clahe.apply(green)

        img = cv2.resize(img, (512, 512))
        return img

if __name__ == "__main__":
    print("=== Running Fixed Unified Data Preprocessor ===")
    preprocessor = FixedUnifiedDataPreprocessor(output_dir="processed_data")

    set1_stats = preprocessor.create_standard_dataset(
        source_dir=r"C:\Users\KIIT\Documents\FreeClaude\res\data\set1",
        dataset_name="set1"
    )

    set2_stats = preprocessor.create_standard_dataset(
        source_dir=r"C:\Users\KIIT\Documents\FreeClaude\res\data\set2",
        dataset_name="set2"
    )

    print("\nPreprocessing Complete!")
    print(f"Set1 processed: {set1_stats['total_images']} images")
    print(f"Set2 processed: {set2_stats['total_images']} images")
