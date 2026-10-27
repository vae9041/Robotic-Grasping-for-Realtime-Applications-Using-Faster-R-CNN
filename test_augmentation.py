#!/usr/bin/env python3
"""
Test script for data augmentation pipeline
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from data_augmentation import *

def test_cornell_loading():
    """Test loading Cornell dataset samples"""
    cornell_path = "./cornell_dataset"
    if os.path.exists(cornell_path):
        dataset = CornellDataset(cornell_path)
        print(f"Found {len(dataset.samples)} Cornell samples")
        
        if len(dataset.samples) > 0:
            sample = dataset.samples[0]
            print(f"First sample: {sample}")
            
            # Try loading grasps
            grasps = dataset.load_grasps(sample['pos_grasps_path'], sample.get('neg_grasps_path'))
            print(f"Loaded {len(grasps)} grasps")
            
            return True
    else:
        print("Cornell dataset not found")
        return False

def test_jacquard_loading():
    """Test loading Jacquard dataset samples"""
    jacquard_path = "./jacquard_dataset"
    if os.path.exists(jacquard_path):
        dataset = JacquardDataset(jacquard_path)
        print(f"Found {len(dataset.samples)} Jacquard samples")
        
        if len(dataset.samples) > 0:
            sample = dataset.samples[0]
            print(f"First sample: {sample}")
            
            # Try loading grasps
            grasps = dataset.load_grasps(sample['grasps_path'])
            print(f"Loaded {len(grasps)} grasps")
            
            return True
    else:
        print("Jacquard dataset not found")
        return False

def test_augmentation():
    """Test augmentation pipeline"""
    config = AugmentationConfig(num_augmentations=2)
    augmenter = GraspAugmenter(config)
    
    # Create dummy data
    rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth_image = np.random.randint(0, 65535, (480, 640), dtype=np.uint16)
    
    # Create dummy grasp
    grasp = GraspRectangle(320, 240, 45, 100, 50, 1.0)
    grasps = [grasp]
    
    print("Testing augmentation...")
    aug_rgb, aug_depth, aug_grasps = augmenter.augment_sample(rgb_image, depth_image, grasps)
    
    print(f"Original image shape: {rgb_image.shape}")
    print(f"Augmented image shape: {aug_rgb.shape}")
    print(f"Original grasps: {len(grasps)}")
    print(f"Augmented grasps: {len(aug_grasps)}")
    
    return True

def run_small_test():
    """Run augmentation on a small subset of data"""
    print("Running small test...")
    
    # Test configuration with minimal augmentations
    config = AugmentationConfig(
        num_augmentations=2,
        rotation_limit=30,
        crop_ratio_min=0.8,
        crop_ratio_max=1.0
    )
    
    # Create output directory
    output_dir = "./test_augmented"
    Path(output_dir).mkdir(exist_ok=True)
    
    cornell_path = "./cornell_dataset"
    jacquard_path = "./jacquard_dataset"
    
    if os.path.exists(cornell_path):
        print("Testing Cornell dataset...")
        cornell_dataset = CornellDataset(cornell_path)
        augmenter = GraspAugmenter(config)
        
        # Process first sample only
        if len(cornell_dataset.samples) > 0:
            sample = cornell_dataset.samples[0]
            try:
                # Load images
                rgb_image = cv2.imread(sample['rgb_path'])
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                depth_image = cv2.imread(sample['depth_path'], cv2.IMREAD_UNCHANGED)
                
                # Load grasps
                grasps = cornell_dataset.load_grasps(
                    sample['pos_grasps_path'], 
                    sample.get('neg_grasps_path')
                )
                
                print(f"Loaded image: {rgb_image.shape}, grasps: {len(grasps)}")
                
                # Generate one augmentation
                aug_rgb, aug_depth, aug_grasps = augmenter.augment_sample(
                    rgb_image, depth_image, grasps
                )
                
                print(f"Augmented: {aug_rgb.shape}, grasps: {len(aug_grasps)}")
                
                # Save test result
                save_augmented_data(
                    Path(output_dir), "test_cornell", 0,
                    aug_rgb, aug_depth, aug_grasps, 'cornell', source_id=sample['sample_id']
                )
                
                print("Cornell test completed successfully!")
                
            except Exception as e:
                print(f"Error in Cornell test: {e}")
    
    print(f"Test results saved in: {output_dir}")

if __name__ == "__main__":
    print("Testing data augmentation pipeline...")
    
    print("\n1. Testing Cornell dataset loading...")
    test_cornell_loading()
    
    print("\n2. Testing Jacquard dataset loading...")
    test_jacquard_loading()
    
    print("\n3. Testing augmentation pipeline...")
    test_augmentation()
    
    print("\n4. Running small augmentation test...")
    run_small_test()
    
    print("\nAll tests completed!")