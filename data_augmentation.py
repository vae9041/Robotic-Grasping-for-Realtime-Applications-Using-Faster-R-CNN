"""
Data Augmentation for Grasp Detection using Cornell and Jacquard Datasets
Implements random crops, zooms, and rotations with proper grasp annotation transformations.
"""

import os
import cv2
import numpy as np
import random
from typing import List, Tuple, Dict, Union
import json
import argparse
from pathlib import Path
import shutil
from dataclasses import dataclass
import math

# Try to import albumentations, fallback to OpenCV if not available
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    print("Warning: albumentations not available. Using OpenCV-based augmentations.")
    ALBUMENTATIONS_AVAILABLE = False


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation parameters"""
    rotation_limit: int = 45
    crop_ratio_min: float = 0.7
    crop_ratio_max: float = 1.0
    zoom_min: float = 0.8
    zoom_max: float = 1.2
    num_augmentations: int = 5
    output_size: Tuple[int, int] = (640, 480)
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2


class GraspRectangle:
    """Represents a grasp rectangle with center, angle, and dimensions"""
    
    def __init__(self, center_x: float, center_y: float, angle: float, width: float, height: float, quality: float = 1.0):
        self.center_x = center_x
        self.center_y = center_y
        self.angle = angle  # in degrees
        self.width = width
        self.height = height
        self.quality = quality
    
    def get_corners(self) -> np.ndarray:
        """Get the four corner points of the grasp rectangle"""
        # Convert angle to radians
        angle_rad = np.radians(self.angle)
        
        # Half dimensions
        hw = self.width / 2
        hh = self.height / 2
        
        # Corner points relative to center (before rotation)
        corners = np.array([
            [-hw, -hh],
            [hw, -hh],
            [hw, hh],
            [-hw, hh]
        ])
        
        # Rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Rotate corners
        rotated_corners = np.dot(corners, rotation_matrix.T)
        
        # Translate to center position
        rotated_corners[:, 0] += self.center_x
        rotated_corners[:, 1] += self.center_y
        
        return rotated_corners
    
    @classmethod
    def from_corners(cls, corners: np.ndarray, quality: float = 1.0):
        """Create GraspRectangle from four corner points"""
        # Calculate center
        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])
        
        # Calculate width and height
        width = np.linalg.norm(corners[1] - corners[0])
        height = np.linalg.norm(corners[2] - corners[1])
        
        # Calculate angle
        dx = corners[1, 0] - corners[0, 0]
        dy = corners[1, 1] - corners[0, 1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        return cls(center_x, center_y, angle, width, height, quality)
    
    def is_valid(self, image_width: int, image_height: int, min_size: int = 10) -> bool:
        """Check if grasp rectangle is valid (within bounds and reasonable size)"""
        corners = self.get_corners()
        
        # Check if all corners are within image bounds
        if (np.any(corners[:, 0] < 0) or np.any(corners[:, 0] >= image_width) or
            np.any(corners[:, 1] < 0) or np.any(corners[:, 1] >= image_height)):
            return False
        
        # Check minimum size
        if self.width < min_size or self.height < min_size:
            return False
        
        return True


class CornellDataset:
    """Handler for Cornell Grasp Dataset"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples from Cornell dataset"""
        samples = []
        
        for folder in sorted(self.dataset_path.iterdir()):
            if folder.is_dir() and folder.name != 'backgrounds':
                for file in folder.glob('pcd*.txt'):
                    if 'cpos' not in file.name and 'cneg' not in file.name:
                        base_name = file.stem
                        rgb_file = folder / f"{base_name}r.png"
                        depth_file = folder / f"{base_name}d.png"
                        pos_file = folder / f"{base_name}cpos.txt"
                        neg_file = folder / f"{base_name}cneg.txt"
                        
                        if all(f.exists() for f in [rgb_file, depth_file, pos_file]):
                            samples.append({
                                'rgb_path': str(rgb_file),
                                'depth_path': str(depth_file),
                                'pos_grasps_path': str(pos_file),
                                'neg_grasps_path': str(neg_file) if neg_file.exists() else None,
                                'dataset': 'cornell'
                            })
        
        return samples
    
    def load_grasps(self, pos_path: str, neg_path: str = None) -> List[GraspRectangle]:
        """Load grasp rectangles from Cornell format files"""
        grasps = []
        
        # Load positive grasps
        if os.path.exists(pos_path):
            with open(pos_path, 'r') as f:
                lines = f.readlines()
                for i in range(0, len(lines), 4):
                    if i + 3 < len(lines):
                        corners = []
                        for j in range(4):
                            x, y = map(float, lines[i + j].strip().split())
                            corners.append([x, y])
                        corners = np.array(corners)
                        grasp = GraspRectangle.from_corners(corners, quality=1.0)
                        grasps.append(grasp)
        
        # Load negative grasps
        if neg_path and os.path.exists(neg_path):
            with open(neg_path, 'r') as f:
                lines = f.readlines()
                for i in range(0, len(lines), 4):
                    if i + 3 < len(lines):
                        corners = []
                        for j in range(4):
                            x, y = map(float, lines[i + j].strip().split())
                            corners.append([x, y])
                        corners = np.array(corners)
                        grasp = GraspRectangle.from_corners(corners, quality=0.0)
                        grasps.append(grasp)
        
        return grasps


class JacquardDataset:
    """Handler for Jacquard Grasp Dataset"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples from Jacquard dataset"""
        samples = []
        
        for folder in sorted(self.dataset_path.iterdir()):
            if folder.is_dir():
                for subfolder in folder.iterdir():
                    if subfolder.is_dir():
                        for rgb_file in subfolder.glob('*_RGB.png'):
                            base_name = rgb_file.name.replace('_RGB.png', '')
                            depth_file = subfolder / f"{base_name}_perfect_depth.tiff"
                            grasps_file = subfolder / f"{base_name}_grasps.txt"
                            
                            if all(f.exists() for f in [rgb_file, depth_file, grasps_file]):
                                samples.append({
                                    'rgb_path': str(rgb_file),
                                    'depth_path': str(depth_file),
                                    'grasps_path': str(grasps_file),
                                    'dataset': 'jacquard'
                                })
        
        return samples
    
    def load_grasps(self, grasps_path: str) -> List[GraspRectangle]:
        """Load grasp rectangles from Jacquard format file"""
        grasps = []
        
        with open(grasps_path, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) >= 5:
                    x, y, theta, w, h = map(float, parts[:5])
                    # Convert theta from degrees to our format
                    grasp = GraspRectangle(x, y, theta, w, h, quality=1.0)
                    grasps.append(grasp)
        
        return grasps


class GraspAugmenter:
    """Handles augmentation of images and grasp annotations"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        
        if ALBUMENTATIONS_AVAILABLE:
            # Define augmentation pipeline using albumentations
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=config.rotation_limit, p=0.8),
                A.RandomResizedCrop(
                    size=(config.output_size[1], config.output_size[0]),
                    scale=(config.crop_ratio_min, config.crop_ratio_max),
                    ratio=(0.75, 1.33),
                    p=0.8
                ),
                A.ColorJitter(
                    brightness=config.brightness_limit,
                    contrast=config.contrast_limit,
                    p=0.5
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
        else:
            self.transform = None
    
    def _opencv_augment(self, image: np.ndarray, grasps: List[GraspRectangle]) -> Tuple[np.ndarray, List[GraspRectangle]]:
        """OpenCV-based augmentation fallback"""
        height, width = image.shape[:2]
        aug_image = image.copy()
        aug_grasps = []
        
        # Random rotation
        if random.random() < 0.8:
            angle = random.uniform(-self.config.rotation_limit, self.config.rotation_limit)
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation to image
            aug_image = cv2.warpAffine(aug_image, rotation_matrix, (width, height))
            
            # Apply rotation to grasps
            for grasp in grasps:
                corners = grasp.get_corners()
                # Add homogeneous coordinate
                corners_hom = np.column_stack([corners, np.ones(4)])
                # Apply rotation
                rotated_corners = np.dot(rotation_matrix, corners_hom.T).T
                
                new_grasp = GraspRectangle.from_corners(rotated_corners, grasp.quality)
                if new_grasp.is_valid(width, height):
                    aug_grasps.append(new_grasp)
        else:
            aug_grasps = grasps.copy()
        
        # Random crop and resize
        if random.random() < 0.8:
            crop_ratio = random.uniform(self.config.crop_ratio_min, self.config.crop_ratio_max)
            new_width = int(width * crop_ratio)
            new_height = int(height * crop_ratio)
            
            if new_width < width and new_height < height:
                # Random crop position
                x_start = random.randint(0, width - new_width)
                y_start = random.randint(0, height - new_height)
                
                # Crop image
                cropped_image = aug_image[y_start:y_start+new_height, x_start:x_start+new_width]
                
                # Resize to output size
                aug_image = cv2.resize(cropped_image, self.config.output_size)
                
                # Scale factor for grasps
                scale_x = self.config.output_size[0] / new_width
                scale_y = self.config.output_size[1] / new_height
                
                # Update grasps
                valid_grasps = []
                for grasp in aug_grasps:
                    # Translate grasp coordinates
                    new_center_x = (grasp.center_x - x_start) * scale_x
                    new_center_y = (grasp.center_y - y_start) * scale_y
                    new_width_g = grasp.width * scale_x
                    new_height_g = grasp.height * scale_y
                    
                    new_grasp = GraspRectangle(new_center_x, new_center_y, grasp.angle, 
                                             new_width_g, new_height_g, grasp.quality)
                    
                    if new_grasp.is_valid(self.config.output_size[0], self.config.output_size[1]):
                        valid_grasps.append(new_grasp)
                
                aug_grasps = valid_grasps
            else:
                # Just resize
                aug_image = cv2.resize(aug_image, self.config.output_size)
                scale_x = self.config.output_size[0] / width
                scale_y = self.config.output_size[1] / height
                
                for grasp in aug_grasps:
                    grasp.center_x *= scale_x
                    grasp.center_y *= scale_y
                    grasp.width *= scale_x
                    grasp.height *= scale_y
        else:
            # Just resize to output size
            aug_image = cv2.resize(aug_image, self.config.output_size)
            scale_x = self.config.output_size[0] / width
            scale_y = self.config.output_size[1] / height
            
            for grasp in aug_grasps:
                grasp.center_x *= scale_x
                grasp.center_y *= scale_y
                grasp.width *= scale_x
                grasp.height *= scale_y
        
        # Color augmentations
        if random.random() < 0.5:
            # Brightness
            brightness_factor = random.uniform(1 - self.config.brightness_limit, 1 + self.config.brightness_limit)
            aug_image = np.clip(aug_image * brightness_factor, 0, 255).astype(np.uint8)
            
            # Contrast
            contrast_factor = random.uniform(1 - self.config.contrast_limit, 1 + self.config.contrast_limit)
            mean = np.mean(aug_image)
            aug_image = np.clip((aug_image - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        
        # Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, random.uniform(10, 50), aug_image.shape).astype(np.int16)
            aug_image = np.clip(aug_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return aug_image, aug_grasps
    
    def augment_sample(self, rgb_image: np.ndarray, depth_image: np.ndarray, grasps: List[GraspRectangle]) -> Tuple[np.ndarray, np.ndarray, List[GraspRectangle]]:
        """Apply augmentation to image and grasps"""
        
        if ALBUMENTATIONS_AVAILABLE and len(grasps) > 0:
            # Convert grasps to keypoints for albumentations
            keypoints = []
            for grasp in grasps:
                corners = grasp.get_corners()
                for corner in corners:
                    keypoints.append(corner.tolist())
            
            # Apply transformation
            transformed = self.transform(image=rgb_image, keypoints=keypoints)
            aug_rgb = transformed['image']
            aug_keypoints = transformed['keypoints']
            
            # Apply same transformation to depth image (without color augmentations)
            depth_transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=self.config.rotation_limit, p=0.8),
                A.RandomResizedCrop(
                    size=(self.config.output_size[1], self.config.output_size[0]),
                    scale=(self.config.crop_ratio_min, self.config.crop_ratio_max),
                    ratio=(0.75, 1.33),
                    p=0.8
                ),
            ], additional_targets={'depth': 'image'})
            
            depth_transformed = depth_transform(image=rgb_image, depth=depth_image)
            aug_depth = depth_transformed['depth']
            
            # Reconstruct grasps from transformed keypoints
            aug_grasps = []
            for i in range(0, len(aug_keypoints), 4):
                if i + 3 < len(aug_keypoints):
                    corners = np.array(aug_keypoints[i:i+4])
                    original_grasp = grasps[i//4]
                    new_grasp = GraspRectangle.from_corners(corners, original_grasp.quality)
                    
                    # Validate grasp
                    if new_grasp.is_valid(aug_rgb.shape[1], aug_rgb.shape[0]):
                        aug_grasps.append(new_grasp)
            
            return aug_rgb, aug_depth, aug_grasps
        
        else:
            # Use OpenCV-based augmentation
            aug_rgb, aug_grasps = self._opencv_augment(rgb_image, grasps)
            aug_depth, _ = self._opencv_augment(depth_image, [])  # Don't augment depth colors
            
            return aug_rgb, aug_depth, aug_grasps


def save_augmented_data(output_dir: Path, sample_id: str, aug_id: int, 
                       rgb_image: np.ndarray, depth_image: np.ndarray, 
                       grasps: List[GraspRectangle], dataset_type: str):
    """Save augmented data in organized structure"""
    
    # Create output directories
    rgb_dir = output_dir / 'rgb'
    depth_dir = output_dir / 'depth'
    annotations_dir = output_dir / 'annotations'
    
    for dir_path in [rgb_dir, depth_dir, annotations_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filenames
    base_name = f"{sample_id}_aug_{aug_id}"
    
    # Save RGB image
    rgb_path = rgb_dir / f"{base_name}.png"
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    
    # Save depth image
    depth_path = depth_dir / f"{base_name}.png"
    if depth_image.dtype != np.uint16:
        depth_image = (depth_image * 65535).astype(np.uint16)
    cv2.imwrite(str(depth_path), depth_image)
    
    # Save annotations
    ann_path = annotations_dir / f"{base_name}.json"
    annotations = {
        'image_id': base_name,
        'dataset_source': dataset_type,
        'grasps': []
    }
    
    for grasp in grasps:
        corners = grasp.get_corners()
        annotations['grasps'].append({
            'center_x': float(grasp.center_x),
            'center_y': float(grasp.center_y),
            'angle': float(grasp.angle),
            'width': float(grasp.width),
            'height': float(grasp.height),
            'quality': float(grasp.quality),
            'corners': corners.tolist()
        })
    
    with open(ann_path, 'w') as f:
        json.dump(annotations, f, indent=2)


def augment_datasets(cornell_path: str, jacquard_path: str, output_path: str, config: AugmentationConfig):
    """Main function to augment Cornell dataset only (Jacquard has enough data)"""
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize only Cornell dataset - Jacquard has sufficient data already
    cornell_dataset = CornellDataset(cornell_path) if os.path.exists(cornell_path) else None
    jacquard_dataset = None  # Skip Jacquard dataset processing
    
    # Initialize augmenter
    augmenter = GraspAugmenter(config)
    
    total_samples = 0
    augmented_samples = 0
    
    # Process Cornell dataset
    if cornell_dataset:
        print(f"Processing Cornell dataset: {len(cornell_dataset.samples)} samples")
        
        for i, sample in enumerate(cornell_dataset.samples):
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
                
                # Generate augmentations
                for aug_id in range(config.num_augmentations):
                    aug_rgb, aug_depth, aug_grasps = augmenter.augment_sample(
                        rgb_image, depth_image, grasps
                    )
                    
                    if len(aug_grasps) > 0:  # Only save if valid grasps remain
                        save_augmented_data(
                            output_dir, f"cornell_{i:06d}", aug_id,
                            aug_rgb, aug_depth, aug_grasps, 'cornell'
                        )
                        augmented_samples += 1
                
                total_samples += 1
                if total_samples % 100 == 0:
                    print(f"Processed {total_samples} samples, generated {augmented_samples} augmentations")
                    
            except Exception as e:
                print(f"Error processing Cornell sample {i}: {e}")
                continue
    
    # Process Jacquard dataset - SKIPPED (already has sufficient data)
    print("Skipping Jacquard dataset processing - dataset already has sufficient samples")
    
    print(f"\nAugmentation complete!")
    print(f"Total samples processed: {total_samples}")
    print(f"Total augmented samples generated: {augmented_samples}")
    print(f"Output directory: {output_dir}")


def visualize_grasps(image: np.ndarray, grasps: List[GraspRectangle], output_path: str):
    """Visualize grasp rectangles on image for debugging"""
    vis_image = image.copy()
    
    for grasp in grasps:
        corners = grasp.get_corners().astype(int)
        
        # Draw grasp rectangle
        color = (0, 255, 0) if grasp.quality > 0.5 else (0, 0, 255)
        cv2.polylines(vis_image, [corners], True, color, 2)
        
        # Draw center point
        center = (int(grasp.center_x), int(grasp.center_y))
        cv2.circle(vis_image, center, 3, color, -1)
        
        # Draw orientation line
        angle_rad = np.radians(grasp.angle)
        end_x = int(grasp.center_x + 20 * np.cos(angle_rad))
        end_y = int(grasp.center_y + 20 * np.sin(angle_rad))
        cv2.line(vis_image, center, (end_x, end_y), color, 2)
    
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment Cornell grasp detection dataset (Jacquard skipped - has enough data)")
    parser.add_argument("--cornell_path", type=str, default="./cornell_dataset", 
                       help="Path to Cornell dataset")
    parser.add_argument("--jacquard_path", type=str, default="./jacquard_dataset",
                       help="Path to Jacquard dataset (will be skipped)")
    parser.add_argument("--output_path", type=str, default="./augmented_dataset",
                       help="Output path for augmented dataset")
    parser.add_argument("--num_augmentations", type=int, default=5,
                       help="Number of augmentations per Cornell sample")
    parser.add_argument("--rotation_limit", type=int, default=45,
                       help="Maximum rotation angle in degrees")
    parser.add_argument("--crop_ratio_min", type=float, default=0.7,
                       help="Minimum crop ratio")
    parser.add_argument("--crop_ratio_max", type=float, default=1.0,
                       help="Maximum crop ratio")
    parser.add_argument("--visualize", action="store_true",
                       help="Save visualization of first few samples")
    
    args = parser.parse_args()
    
    # Create configuration
    config = AugmentationConfig(
        rotation_limit=args.rotation_limit,
        crop_ratio_min=args.crop_ratio_min,
        crop_ratio_max=args.crop_ratio_max,
        num_augmentations=args.num_augmentations
    )
    
    # Run augmentation
    augment_datasets(
        args.cornell_path,
        args.jacquard_path,
        args.output_path,
        config
    )
    
    # Optional visualization
    if args.visualize:
        print("Creating visualizations...")
        # Add visualization code here if needed
