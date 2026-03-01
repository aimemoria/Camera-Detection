#!/usr/bin/env python3
"""
Download and organize the Olivetti Faces dataset for training.
This dataset is built into sklearn and downloads automatically.

- 40 people × 10 images each = 400 images
- 64×64 grayscale images
- We'll select 5 for training, 3 for unknown testing
"""

import os
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from PIL import Image
from pathlib import Path

def main():
    print("=" * 50)
    print("Downloading Olivetti Faces Dataset")
    print("=" * 50)
    
    # Download dataset (sklearn caches it automatically)
    print("\nFetching dataset from sklearn...")
    data = fetch_olivetti_faces()
    
    images = data.images  # Shape: (400, 64, 64)
    labels = data.target  # Shape: (400,) - person IDs 0-39
    
    print(f"Downloaded {len(images)} images")
    print(f"Image shape: {images[0].shape}")
    print(f"Number of people: {len(np.unique(labels))}")
    
    # Create directory structure
    base_dir = Path("dataset")
    
    dirs = [
        base_dir / "stage_a" / "person",
        base_dir / "stage_a" / "no_person",
        base_dir / "stage_b" / "person1",
        base_dir / "stage_b" / "person2",
        base_dir / "stage_b" / "person3",
        base_dir / "stage_b" / "person4",
        base_dir / "stage_b" / "person5",
        base_dir / "unknown_test" / "other_person1",
        base_dir / "unknown_test" / "other_person2",
        base_dir / "unknown_test" / "other_person3",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    print("\nCreated directory structure")
    
    # Select 5 people for known faces (persons 0-4)
    # Select 3 people for unknown testing (persons 5-7)
    known_persons = [0, 1, 2, 3, 4]
    unknown_persons = [5, 6, 7]
    
    # Map dataset person IDs to our person names
    person_mapping = {
        0: "person1",
        1: "person2", 
        2: "person3",
        3: "person4",
        4: "person5",
    }
    
    unknown_mapping = {
        5: "other_person1",
        6: "other_person2",
        7: "other_person3",
    }
    
    print("\n--- Saving Known Persons (Stage B) ---")
    
    for person_id in known_persons:
        person_name = person_mapping[person_id]
        person_images = images[labels == person_id]
        
        # Save to stage_b folder
        for i, img in enumerate(person_images):
            # Convert to uint8 (0-255)
            img_uint8 = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode='L')
            
            # Resize to 96x96 (our target size)
            pil_img = pil_img.resize((96, 96), Image.Resampling.LANCZOS)
            
            # Save
            filename = f"{person_name}_{i+1:03d}.png"
            pil_img.save(base_dir / "stage_b" / person_name / filename)
        
        # Also save to stage_a/person folder
        for i, img in enumerate(person_images):
            img_uint8 = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode='L')
            pil_img = pil_img.resize((96, 96), Image.Resampling.LANCZOS)
            
            filename = f"{person_name}_{i+1:03d}.png"
            pil_img.save(base_dir / "stage_a" / "person" / filename)
        
        print(f"  {person_name}: {len(person_images)} images saved")
    
    print("\n--- Saving Unknown Persons ---")
    
    for person_id in unknown_persons:
        person_name = unknown_mapping[person_id]
        person_images = images[labels == person_id]
        
        for i, img in enumerate(person_images):
            img_uint8 = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode='L')
            pil_img = pil_img.resize((96, 96), Image.Resampling.LANCZOS)
            
            filename = f"{person_name}_{i+1:03d}.png"
            pil_img.save(base_dir / "unknown_test" / person_name / filename)
        
        print(f"  {person_name}: {len(person_images)} images saved")
    
    print("\n--- Generating Background (No Person) Images ---")
    
    # Create synthetic backgrounds by:
    # 1. Random noise
    # 2. Blurred/scrambled face regions
    # 3. Gradient patterns
    
    num_backgrounds = 100
    
    for i in range(num_backgrounds):
        if i % 3 == 0:
            # Random noise
            img = np.random.randint(0, 256, (96, 96), dtype=np.uint8)
        elif i % 3 == 1:
            # Gradient
            gradient = np.linspace(0, 255, 96).astype(np.uint8)
            if i % 2 == 0:
                img = np.tile(gradient, (96, 1))
            else:
                img = np.tile(gradient.reshape(-1, 1), (1, 96))
        else:
            # Uniform gray
            gray_value = np.random.randint(50, 200)
            img = np.full((96, 96), gray_value, dtype=np.uint8)
            # Add some noise
            noise = np.random.randint(-20, 20, (96, 96))
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        pil_img = Image.fromarray(img, mode='L')
        filename = f"background_{i+1:03d}.png"
        pil_img.save(base_dir / "stage_a" / "no_person" / filename)
    
    print(f"  Generated {num_backgrounds} background images")
    
    # Summary
    print("\n" + "=" * 50)
    print("Dataset Setup Complete!")
    print("=" * 50)
    
    print("\nDataset structure:")
    print(f"  dataset/stage_a/person/     : {len(list((base_dir / 'stage_a' / 'person').glob('*')))} images")
    print(f"  dataset/stage_a/no_person/  : {len(list((base_dir / 'stage_a' / 'no_person').glob('*')))} images")
    print(f"  dataset/stage_b/person1/    : {len(list((base_dir / 'stage_b' / 'person1').glob('*')))} images")
    print(f"  dataset/stage_b/person2/    : {len(list((base_dir / 'stage_b' / 'person2').glob('*')))} images")
    print(f"  dataset/stage_b/person3/    : {len(list((base_dir / 'stage_b' / 'person3').glob('*')))} images")
    print(f"  dataset/stage_b/person4/    : {len(list((base_dir / 'stage_b' / 'person4').glob('*')))} images")
    print(f"  dataset/stage_b/person5/    : {len(list((base_dir / 'stage_b' / 'person5').glob('*')))} images")
    print(f"  dataset/unknown_test/       : {len(list((base_dir / 'unknown_test').rglob('*.png')))} images")
    
    print("\nNext step: Run the training pipeline")
    print("  ./run_pipeline.sh")

if __name__ == "__main__":
    main()
