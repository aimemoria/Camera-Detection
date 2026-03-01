"""
TinyML Person Detection - Dataset Preprocessing Pipeline

This script:
1. Resizes images to target resolution (96x96 or 64x64)
2. Normalizes pixel values to [-1, 1] range
3. Applies augmentation (flip, brightness, rotation, noise)
4. Converts to grayscale (optional)
5. Saves preprocessed dataset with train/val/test splits

Memory Tradeoff Analysis:
- RGB (96x96x3): 27,648 bytes per image, better accuracy, color info
- Grayscale (96x96x1): 9,216 bytes per image, 3x less memory
- Grayscale (64x64x1): 4,096 bytes per image, 6.75x less memory than RGB 96x96

Recommendation: Start with RGB, convert to grayscale only if Arduino OOM.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import argparse
import shutil


class DatasetPreprocessor:
    """Preprocessing pipeline for TinyML person detection"""
    
    def __init__(self, 
                 input_dir,
                 output_dir,
                 target_size=(96, 96),
                 grayscale=False,
                 normalize=True,
                 augmentation=True):
        """
        Args:
            input_dir: Root directory with person/ and no_person/ subdirs
            output_dir: Output directory for preprocessed data
            target_size: (height, width) tuple, e.g., (96, 96) or (64, 64)
            grayscale: If True, convert to grayscale
            normalize: If True, normalize to [-1, 1]
            augmentation: If True, apply augmentation to training set
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.grayscale = grayscale
        self.normalize = normalize
        self.augmentation = augmentation
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            for cls in ['person', 'no_person']:
                (self.output_dir / split / cls).mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'test_images': 0,
            'augmented_images': 0,
            'person_images': 0,
            'no_person_images': 0
        }
    
    def load_and_preprocess_image(self, image_path, augment=False):
        """
        Load and preprocess a single image
        
        Returns:
            List of preprocessed images (1 original + N augmented if augment=True)
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read {image_path}")
            return []
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale if needed
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
        
        # Normalize to [-1, 1]
        if self.normalize:
            img = (img.astype(np.float32) / 127.5) - 1.0
        else:
            img = img.astype(np.float32) / 255.0  # [0, 1] range
        
        images = [img]
        
        # Apply augmentation if requested
        if augment and self.augmentation:
            images.extend(self._augment_image(img))
        
        return images
    
    def _augment_image(self, img):
        """
        Apply augmentation to create additional training samples
        
        Args:
            img: Normalized image array
        
        Returns:
            List of augmented images
        """
        augmented = []
        
        # Denormalize for OpenCV operations if needed
        if self.normalize:
            img_uint8 = ((img + 1.0) * 127.5).astype(np.uint8)
        else:
            img_uint8 = (img * 255).astype(np.uint8)
        
        # 1. Horizontal flip
        flipped = cv2.flip(img_uint8, 1)
        augmented.append(self._normalize_image(flipped))
        
        # 2. Brightness adjustment (±20%)
        bright = cv2.convertScaleAbs(img_uint8, alpha=1.2, beta=0)
        augmented.append(self._normalize_image(bright))
        
        dark = cv2.convertScaleAbs(img_uint8, alpha=0.8, beta=0)
        augmented.append(self._normalize_image(dark))
        
        # 3. Rotation (±10 degrees)
        h, w = img_uint8.shape[:2]
        center = (w // 2, h // 2)
        
        for angle in [10, -10]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_uint8, M, (w, h), 
                                     borderMode=cv2.BORDER_REFLECT)
            augmented.append(self._normalize_image(rotated))
        
        # 4. Gaussian noise
        noisy = img_uint8.copy().astype(np.float32)
        noise = np.random.normal(0, 5, noisy.shape)  # σ=5 for uint8
        noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
        augmented.append(self._normalize_image(noisy))
        
        return augmented
    
    def _normalize_image(self, img_uint8):
        """Normalize uint8 image to [-1, 1] or [0, 1]"""
        if self.normalize:
            return (img_uint8.astype(np.float32) / 127.5) - 1.0
        else:
            return img_uint8.astype(np.float32) / 255.0
    
    def collect_image_paths(self):
        """
        Collect all image paths from input directory
        
        Expected structure:
        input_dir/
            person/*.jpg
            no_person/*.jpg
        
        Returns:
            dict: {'person': [...paths...], 'no_person': [...paths...]}
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        paths = {'person': [], 'no_person': []}
        
        for class_name in ['person', 'no_person']:
            class_dir = self.input_dir / class_name
            if not class_dir.exists():
                raise ValueError(f"Directory not found: {class_dir}")
            
            for ext in image_extensions:
                paths[class_name].extend(list(class_dir.glob(f'*{ext}')))
                paths[class_name].extend(list(class_dir.glob(f'*{ext.upper()}')))
        
        print(f"Found {len(paths['person'])} person images")
        print(f"Found {len(paths['no_person'])} no_person images")
        
        return paths
    
    def split_dataset(self, paths, val_ratio=0.15, test_ratio=0.15):
        """
        Split dataset into train/val/test
        
        Args:
            paths: Dict with 'person' and 'no_person' lists
            val_ratio: Validation set ratio (0.15 = 15%)
            test_ratio: Test set ratio (0.15 = 15%)
        
        Returns:
            dict: {'train': [...], 'val': [...], 'test': [...]}
        """
        splits = {'train': [], 'val': [], 'test': []}
        
        for class_name, class_paths in paths.items():
            # First split: separate test set
            train_val, test = train_test_split(
                class_paths, 
                test_size=test_ratio, 
                random_state=42
            )
            
            # Second split: separate validation from training
            val_ratio_adjusted = val_ratio / (1 - test_ratio)
            train, val = train_test_split(
                train_val,
                test_size=val_ratio_adjusted,
                random_state=42
            )
            
            # Store with class labels
            splits['train'].extend([(p, class_name) for p in train])
            splits['val'].extend([(p, class_name) for p in val])
            splits['test'].extend([(p, class_name) for p in test])
        
        # Shuffle splits
        np.random.seed(42)
        for split in splits.values():
            np.random.shuffle(split)
        
        print(f"\nDataset split:")
        print(f"  Train: {len(splits['train'])} images")
        print(f"  Val:   {len(splits['val'])} images")
        print(f"  Test:  {len(splits['test'])} images")
        
        return splits
    
    def process_and_save(self, splits):
        """
        Process all images and save to output directory
        
        Args:
            splits: Dict from split_dataset()
        """
        for split_name, split_data in splits.items():
            print(f"\nProcessing {split_name} set...")
            
            # Apply augmentation only to training set
            apply_augment = (split_name == 'train') and self.augmentation
            
            for img_path, class_name in tqdm(split_data):
                # Process image (returns list: [original] or [original, aug1, aug2, ...])
                processed_images = self.load_and_preprocess_image(
                    img_path, 
                    augment=apply_augment
                )
                
                if not processed_images:
                    continue
                
                # Save each processed image
                base_name = img_path.stem
                for i, proc_img in enumerate(processed_images):
                    # Generate output filename
                    if i == 0:
                        out_name = f"{base_name}.npy"
                    else:
                        out_name = f"{base_name}_aug{i}.npy"
                    
                    out_path = self.output_dir / split_name / class_name / out_name
                    
                    # Save as numpy array for fast loading
                    np.save(out_path, proc_img)
                    
                    # Update stats
                    self.stats['total_images'] += 1
                    self.stats[f'{split_name}_images'] += 1
                    if i > 0:
                        self.stats['augmented_images'] += 1
                    self.stats[f'{class_name.replace("_", "")}_images'] += 1
        
        # Save configuration and stats
        self._save_metadata()
    
    def _save_metadata(self):
        """Save preprocessing configuration and statistics"""
        metadata = {
            'config': {
                'target_size': self.target_size,
                'grayscale': self.grayscale,
                'normalize': self.normalize,
                'augmentation': self.augmentation,
                'channels': 1 if self.grayscale else 3
            },
            'stats': self.stats,
            'memory_per_image_bytes': (
                self.target_size[0] * self.target_size[1] * 
                (1 if self.grayscale else 3)
            ),
            'input_shape': [
                self.target_size[0], 
                self.target_size[1], 
                1 if self.grayscale else 3
            ]
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nPreprocessing complete!")
        print(f"Total images processed: {self.stats['total_images']}")
        print(f"  - Original: {self.stats['total_images'] - self.stats['augmented_images']}")
        print(f"  - Augmented: {self.stats['augmented_images']}")
        print(f"\nPerson images: {self.stats['personimages']}")
        print(f"No-person images: {self.stats['nopersonimages']}")
        print(f"\nMetadata saved to {self.output_dir / 'metadata.json'}")
    
    def run(self):
        """Execute full preprocessing pipeline"""
        print("="*60)
        print("TinyML Person Detection - Dataset Preprocessing")
        print("="*60)
        print(f"\nConfiguration:")
        print(f"  Target size: {self.target_size}")
        print(f"  Color mode: {'Grayscale' if self.grayscale else 'RGB'}")
        print(f"  Normalization: {'[-1, 1]' if self.normalize else '[0, 1]'}")
        print(f"  Augmentation: {self.augmentation}")
        
        # Memory analysis
        mem_per_img = (self.target_size[0] * self.target_size[1] * 
                       (1 if self.grayscale else 3))
        print(f"\nMemory per image: {mem_per_img:,} bytes ({mem_per_img/1024:.2f} KB)")
        
        # Step 1: Collect image paths
        paths = self.collect_image_paths()
        
        # Step 2: Split dataset
        splits = self.split_dataset(paths)
        
        # Step 3: Process and save
        self.process_and_save(splits)
        
        return self.output_dir / 'metadata.json'


def verify_dataset(preprocessed_dir):
    """
    Verify preprocessed dataset integrity
    
    Args:
        preprocessed_dir: Path to preprocessed dataset
    """
    print("\n" + "="*60)
    print("Verifying preprocessed dataset...")
    print("="*60)
    
    preprocessed_dir = Path(preprocessed_dir)
    
    # Load metadata
    with open(preprocessed_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"\nConfiguration:")
    print(f"  Input shape: {metadata['input_shape']}")
    print(f"  Memory per image: {metadata['memory_per_image_bytes']:,} bytes")
    
    # Check each split
    for split in ['train', 'val', 'test']:
        for cls in ['person', 'no_person']:
            split_dir = preprocessed_dir / split / cls
            npy_files = list(split_dir.glob('*.npy'))
            
            if npy_files:
                # Load one sample to verify shape
                sample = np.load(npy_files[0])
                print(f"\n{split}/{cls}:")
                print(f"  Files: {len(npy_files)}")
                print(f"  Shape: {sample.shape}")
                print(f"  Dtype: {sample.dtype}")
                print(f"  Range: [{sample.min():.3f}, {sample.max():.3f}]")
            else:
                print(f"\nWarning: No files in {split}/{cls}")
    
    print("\n" + "="*60)
    print("Verification complete!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess dataset for TinyML person detection'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing person/ and no_person/ subdirectories'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='preprocessed_dataset',
        help='Output directory for preprocessed data'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=96,
        help='Target image size (square). Options: 64, 96 (default: 96)'
    )
    parser.add_argument(
        '--grayscale',
        action='store_true',
        help='Convert to grayscale (reduces memory by 3x)'
    )
    parser.add_argument(
        '--no-augmentation',
        action='store_true',
        help='Disable data augmentation'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify preprocessed dataset after creation'
    )
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=(args.size, args.size),
        grayscale=args.grayscale,
        normalize=True,
        augmentation=not args.no_augmentation
    )
    
    # Run preprocessing
    metadata_path = preprocessor.run()
    
    # Verify if requested
    if args.verify:
        verify_dataset(args.output_dir)
    
    print(f"\n✓ Preprocessed dataset ready at: {args.output_dir}")
    print(f"✓ Use this dataset for training with the input shape from metadata.json")
