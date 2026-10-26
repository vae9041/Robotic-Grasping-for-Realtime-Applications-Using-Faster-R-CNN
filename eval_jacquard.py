#!/usr/bin/env python3
"""
Jacquard Dataset Evaluation Script for Faster R-CNN Grasp Detection

This script evaluates a trained Faster R-CNN model on the Jacquard dataset.
The Jacquard dataset uses a different grasp format compared to Cornell:
- Each line in grasps.txt: center_x;center_y;angle;width;height
- Multiple views per scene with different camera angles

Usage:
    python eval_jacquard.py --model_path path/to/model.pth --jacquard_path path/to/jacquard --output_dir results/
"""

import argparse
import logging
import os
import json
import time
import math
from pathlib import Path
import random
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from structured_pruning_resnet_18 import load_structured_pruned_state_dict

from train import GraspRectangle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _aabb_iou(box1, box2):
    """Compute IoU between two axis-aligned bounding boxes."""
    box1 = np.asarray(box1, dtype=float)
    box2 = np.asarray(box2, dtype=float)
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


class JacquardDataset:
    """Dataset loader for Jacquard grasp dataset."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize Jacquard dataset loader.
        
        Args:
            dataset_path: Path to Jacquard dataset root
        """
        self.dataset_path = Path(dataset_path)
        self.samples = self._load_samples()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples from the dataset."""
        samples = []
        
        # Jacquard is split into sub-folders like Jacquard_Dataset_0, ..., Jacquard_Dataset_11
        for folder in sorted(self.dataset_path.iterdir()):
            if not folder.is_dir():
                continue
            
            # Each folder contains scene directories
            for scene_dir in folder.iterdir():
                if not scene_dir.is_dir():
                    continue
                
                # Each scene contains multiple views (0-4)
                for rgb_file in scene_dir.glob("*_RGB.png"):
                    # Extract view number and scene ID from filename
                    filename_parts = rgb_file.stem.split('_')
                    if len(filename_parts) < 2:
                        continue
                    view_num = filename_parts[0]
                    scene_id = filename_parts[1]
                    
                    # Corresponding grasp file
                    grasp_file = scene_dir / f"{view_num}_{scene_id}_grasps.txt"
                    
                    if grasp_file.exists():
                        samples.append({
                            'image_path': str(rgb_file),
                            'grasp_path': str(grasp_file),
                            'scene_id': scene_id,
                            'view_num': int(view_num),
                            'folder': folder.name
                        })
        
        logger.info(f"Found {len(samples)} Jacquard samples")
        return samples
    
    def _load_grasps(self, grasp_path: str) -> List[GraspRectangle]:
        """Load grasps from Jacquard format file."""
        grasps = []
        
        try:
            with open(grasp_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse: center_x;center_y;angle;width;height
                    parts = line.split(';')
                    if len(parts) != 5:
                        continue
                    
                    center_x = float(parts[0])
                    center_y = float(parts[1])
                    angle = float(parts[2])  # degrees in Jacquard
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    grasp = GraspRectangle(
                        center_x=center_x,
                        center_y=center_y,
                        width=width,
                        height=height,
                        angle=angle,
                        format_type='jacquard'
                    )
                    grasps.append(grasp)
        
        except Exception as e:
            logger.warning(f"Error loading grasps from {grasp_path}: {e}")
        
        return grasps
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Load grasps
        grasps = self._load_grasps(sample['grasp_path'])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'grasps': grasps,
            'image_path': sample['image_path'],
            'scene_id': sample['scene_id'],
            'view_num': sample['view_num'],
            'folder': sample['folder']
        }


def get_model(num_classes: int = 2) -> nn.Module:
    """Create Faster R-CNN model with ResNet50 FPN backbone and oriented ROI head."""
    from train import get_model as get_oriented_model
    return get_oriented_model(num_classes=num_classes, freeze_backbone=False)


class JacquardEvaluator:
    """Evaluator for Jacquard dataset."""
    
    def __init__(self, model_path: str, jacquard_path: str, device: str = None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            jacquard_path: Path to Jacquard dataset
            device: Device to use for inference
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = get_model()
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            if isinstance(checkpoint, dict) and checkpoint.get("structured_pruning_meta"):
                load_structured_pruned_state_dict(self.model, checkpoint, self.device)
                logger.info("Loaded structured-pruned checkpoint with FPN wrapper metadata")
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)
            logger.info("Loaded model weights")
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded model on device: {self.device}")
        
        # Load dataset
        self.dataset = JacquardDataset(jacquard_path)
        logger.info(f"Found {len(self.dataset)} Jacquard samples for evaluation")
    
    def evaluate(self, confidence_threshold: float = 0.5, iou_threshold: float = 0.25,
                 angle_threshold: float = 30.0, max_samples: Optional[int] = None) -> Dict:
        """
        Evaluate model on Jacquard dataset using oriented IoU + angle.

        Args:
            confidence_threshold: Minimum confidence for predictions
            iou_threshold: IoU threshold for positive matches
            angle_threshold: Angle error threshold (degrees) for positive matches
            max_samples: Maximum number of samples to evaluate (None for all)

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Starting evaluation...")

        # Determine number of samples to evaluate
        num_samples = len(self.dataset)
        if max_samples is not None:
            num_samples = min(max_samples, num_samples)

        logger.info(f"Evaluating on {num_samples} samples...")

        # Metrics tracking
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_gt_grasps = 0
        total_predictions = 0
        iou_scores = []
        angle_errors = []

        # Sample tracking for detailed analysis
        sample_results = []

        # Evaluation loop
        for i in tqdm(range(num_samples), desc="Evaluating"):
            sample = self.dataset[i]
            image = sample['image'].unsqueeze(0).to(self.device)
            gt_grasps = sample['grasps']

            # Skip samples with no ground truth grasps
            if len(gt_grasps) == 0:
                continue

            # Get predictions
            with torch.no_grad():
                predictions = self.model(image)[0]

            # Filter predictions by confidence
            keep_indices = predictions['scores'] >= confidence_threshold
            pred_boxes = predictions['boxes'][keep_indices].cpu().numpy()
            pred_scores = predictions['scores'][keep_indices].cpu().numpy()
            pred_angles = predictions['angles'][keep_indices].cpu().numpy()

            # Convert predicted boxes to oriented grasp rectangles
            pred_grasps = []
            for box, angle in zip(pred_boxes, pred_angles):
                x_min, y_min, x_max, y_max = box
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                grasp = GraspRectangle(center_x, center_y, width, height, float(angle), 'cornell')
                pred_grasps.append(grasp)

            # Count ground truth grasps
            total_gt_grasps += len(gt_grasps)
            total_predictions += len(pred_grasps)

            # Calculate matches using oriented IoU and angle
            gt_matched = [False] * len(gt_grasps)
            pred_matched = [False] * len(pred_grasps)
            sample_ious = []
            sample_angle_errors = []

            # Find best matches
            for pred_idx, (pred_box, pred_grasp) in enumerate(zip(pred_boxes, pred_grasps)):
                best_iou = 0.0
                best_angle_error = 90.0
                best_gt_idx = -1

                for gt_idx, gt_grasp in enumerate(gt_grasps):
                    if gt_matched[gt_idx]:
                        continue

                    iou = _aabb_iou(pred_box, gt_grasp.to_bbox())
                    angle_error = pred_grasp.angle_error(gt_grasp)
                    if iou > best_iou or (iou == best_iou and angle_error < best_angle_error):
                        best_iou = iou
                        best_angle_error = angle_error
                        best_gt_idx = gt_idx

                # Check if match is good enough
                if best_iou > iou_threshold and best_angle_error <= angle_threshold and best_gt_idx != -1:
                    true_positives += 1
                    pred_matched[pred_idx] = True
                    gt_matched[best_gt_idx] = True
                    iou_scores.append(best_iou)
                    angle_errors.append(best_angle_error)
                    sample_ious.append(best_iou)
                    sample_angle_errors.append(best_angle_error)
                else:
                    false_positives += 1

            # Count unmatched ground truth as false negatives
            false_negatives += sum(1 for matched in gt_matched if not matched)

            # Store sample results
            sample_results.append({
                'scene_id': sample['scene_id'],
                'view_num': sample['view_num'],
                'folder': sample['folder'],
                'gt_grasps': len(gt_grasps),
                'predictions': len(pred_grasps),
                'true_positives': sum(pred_matched),
                'sample_ious': sample_ious,
                'sample_angle_errors': sample_angle_errors,
                'mean_iou': np.mean(sample_ious) if sample_ious else 0.0,
                'mean_angle_error': np.mean(sample_angle_errors) if sample_angle_errors else 0.0
            })

        # Calculate final metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = np.mean(iou_scores) if iou_scores else 0.0
        mean_angle_error = np.mean(angle_errors) if angle_errors else 0.0

        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mean_iou': mean_iou,
            'mean_angle_error': mean_angle_error,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_gt_grasps': total_gt_grasps,
            'total_predictions': total_predictions,
            'samples_evaluated': num_samples,
            'confidence_threshold': confidence_threshold,
            'iou_threshold': iou_threshold,
            'angle_threshold': angle_threshold,
            'sample_results': sample_results
        }

        return results
    
    def evaluate_success_rate(self, confidence_threshold: float = 0.5, iou_threshold: float = 0.25,
                              angle_threshold: float = 30.0, max_samples: Optional[int] = None) -> Dict:
        """
        Evaluate using oriented-grasp success rate.
        For each image, select the best predicted grasp and check if it has IoU > iou_threshold
        AND |angle difference| <= angle_threshold with any GT grasp.
        """
        # Determine number of samples to evaluate
        num_samples = len(self.dataset)
        if max_samples is not None:
            num_samples = min(max_samples, num_samples)

        logger.info(f"Evaluating oriented success rate on {num_samples} samples...")

        correct_predictions = 0
        total_images = 0
        detailed_results = []

        # Evaluation loop
        for i in tqdm(range(num_samples), desc="Evaluating Success Rate"):
            sample = self.dataset[i]
            image = sample['image'].unsqueeze(0).to(self.device)
            gt_grasps = sample['grasps']

            total_images += 1

            # Skip samples with no ground truth grasps
            if len(gt_grasps) == 0:
                detailed_results.append({
                    'sample_idx': i,
                    'scene_id': sample['scene_id'],
                    'view_num': sample['view_num'],
                    'folder': sample['folder'],
                    'success': False,
                    'reason': 'No ground truth grasps',
                    'best_iou': 0.0,
                    'best_angle_error': 0.0,
                    'num_predictions': 0,
                    'num_gt_grasps': 0
                })
                continue

            # Get predictions
            with torch.no_grad():
                predictions = self.model(image)[0]

            # Filter predictions by confidence
            keep_indices = predictions['scores'] >= confidence_threshold
            pred_boxes = predictions['boxes'][keep_indices].cpu().numpy()
            pred_scores = predictions['scores'][keep_indices].cpu().numpy()
            pred_angles = predictions['angles'][keep_indices].cpu().numpy()

            # Skip if no confident predictions
            if len(pred_boxes) == 0:
                detailed_results.append({
                    'sample_idx': i,
                    'scene_id': sample['scene_id'],
                    'view_num': sample['view_num'],
                    'folder': sample['folder'],
                    'success': False,
                    'reason': 'No confident predictions',
                    'best_iou': 0.0,
                    'best_angle_error': 0.0,
                    'num_predictions': 0,
                    'num_gt_grasps': len(gt_grasps)
                })
                continue

            # Select best prediction (highest confidence)
            best_pred_idx = int(np.argmax(pred_scores))
            best_pred_box = pred_boxes[best_pred_idx]
            best_pred_score = pred_scores[best_pred_idx]
            best_pred_angle = float(pred_angles[best_pred_idx])

            # Convert to oriented grasp rectangle
            best_pred_grasp = GraspRectangle.from_bbox(
                best_pred_box[0], best_pred_box[1],
                best_pred_box[2], best_pred_box[3], best_pred_angle
            )

            # Check IoU and angle with all ground truth grasps
            best_iou = 0.0
            best_angle_error = 90.0
            for gt_grasp in gt_grasps:
                iou = _aabb_iou(best_pred_box, gt_grasp.to_bbox())
                angle_error = best_pred_grasp.angle_error(gt_grasp)
                if iou > best_iou or (iou == best_iou and angle_error < best_angle_error):
                    best_iou = iou
                    best_angle_error = angle_error

            success = best_iou > iou_threshold and best_angle_error <= angle_threshold
            if success:
                correct_predictions += 1

            detailed_results.append({
                'sample_idx': i,
                'scene_id': sample['scene_id'],
                'view_num': sample['view_num'],
                'folder': sample['folder'],
                'success': success,
                'best_iou': best_iou,
                'best_angle_error': best_angle_error,
                'best_score': best_pred_score,
                'num_predictions': len(pred_boxes),
                'num_gt_grasps': len(gt_grasps),
                'reason': 'Success' if success else f'IoU {best_iou:.3f} / angle {best_angle_error:.1f}'
            })

        # Calculate final metrics
        accuracy = correct_predictions / total_images if total_images > 0 else 0.0

        results = {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_images': total_images,
            'confidence_threshold': confidence_threshold,
            'iou_threshold': iou_threshold,
            'angle_threshold': angle_threshold,
            'detailed_results': detailed_results
        }

        return results
    
    def visualize_predictions(self, num_visualizations: int = 20, 
                            confidence_threshold: float = 0.5,
                            output_dir: str = "visualizations") -> None:
        """
        Create visualizations of predictions vs ground truth.
        
        Args:
            num_visualizations: Number of samples to visualize
            confidence_threshold: Minimum confidence for visualization
            output_dir: Directory to save visualizations
        """
        logger.info("Generating visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample random indices for visualization
        indices = random.sample(range(len(self.dataset)), 
                              min(num_visualizations, len(self.dataset)))
        
        for idx in indices:
            sample = self.dataset[idx]
            image = sample['image']
            gt_grasps = sample['grasps']
            
            # Get predictions
            with torch.no_grad():
                image_tensor = image.unsqueeze(0).to(self.device)
                predictions = self.model(image_tensor)[0]
            
            # Filter predictions by confidence
            keep_indices = predictions['scores'] >= confidence_threshold
            pred_boxes = predictions['boxes'][keep_indices].cpu().numpy()
            pred_scores = predictions['scores'][keep_indices].cpu().numpy()
            
            # Convert tensor to PIL image
            image_pil = transforms.ToPILImage()(image)
            image_viz = image_pil.copy()
            draw = ImageDraw.Draw(image_viz)
            
            # Draw ground truth grasps in green
            for gt_grasp in gt_grasps:
                corners = gt_grasp.to_corners()
                # Close the polygon by adding first point at the end
                polygon_points = corners + [corners[0]]
                draw.polygon([(x, y) for x, y in polygon_points], outline='green', width=2)
            
            # Draw predictions in red
            for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
                x_min, y_min, x_max, y_max = box
                draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)
                
                # Add confidence score
                try:
                    font = ImageFont.load_default()
                    draw.text((x_min, y_min - 10), f'{score:.2f}', fill='red', font=font)
                except:
                    draw.text((x_min, y_min - 10), f'{score:.2f}', fill='red')
            
            # Add title
            try:
                font = ImageFont.load_default()
                title = f"Scene: {sample['scene_id']}, View: {sample['view_num']}"
                draw.text((10, 10), title, fill='blue', font=font)
                draw.text((10, 30), f"GT: {len(gt_grasps)}, Pred: {len(pred_boxes)}", 
                         fill='blue', font=font)
            except:
                pass
            
            # Save visualization
            filename = f"jacquard_{sample['folder']}_{sample['scene_id']}_view{sample['view_num']}.png"
            filepath = os.path.join(output_dir, filename)
            image_viz.save(filepath)
        
        logger.info(f"Saved {len(indices)} visualization images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Faster R-CNN on Jacquard dataset')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model (.pth file)')
    parser.add_argument('--jacquard_path', type=str, required=True,
                        help='Path to Jacquard dataset')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions')
    parser.add_argument('--iou_threshold', type=float, default=0.25,
                        help='IoU threshold for positive matches')
    parser.add_argument('--angle_threshold', type=float, default=30.0,
                        help='Angle error threshold (degrees) for positive matches')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--output_dir', type=str, default='jacquard_evaluation',
                        help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization images')
    parser.add_argument('--num_visualizations', type=int, default=20,
                        help='Number of visualization images to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = JacquardEvaluator(args.model_path, args.jacquard_path)
    
    # Run detailed evaluation (per-grasp metrics)
    logger.info("Starting detailed evaluation...")
    results = evaluator.evaluate(
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        angle_threshold=args.angle_threshold,
        max_samples=args.max_samples
    )

    # Run oriented success rate evaluation (per-image metrics)
    logger.info("Starting oriented success rate evaluation...")
    success_results = evaluator.evaluate_success_rate(
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        angle_threshold=args.angle_threshold,
        max_samples=args.max_samples
    )

    # Print results
    print("\n" + "=" * 60)
    print("JACQUARD DATASET EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.jacquard_path}")
    print(f"Samples evaluated: {results['samples_evaluated']}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Angle threshold: {args.angle_threshold} degrees")

    print("\n" + "="*20 + " PER-GRASP METRICS " + "="*20)
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"Mean Angle Error: {results['mean_angle_error']:.2f} degrees")
    print("-" * 60)
    print(f"True Positives: {results['true_positives']}")
    print(f"False Positives: {results['false_positives']}")
    print(f"False Negatives: {results['false_negatives']}")
    print(f"Total GT Grasps: {results['total_gt_grasps']}")
    print(f"Total Predictions: {results['total_predictions']}")

    print("\n" + "="*18 + " ORIENTED SUCCESS RATE " + "="*18)
    print(f"Success Rate (Accuracy): {success_results['accuracy']:.4f} ({success_results['accuracy']*100:.2f}%)")
    print(f"Correct Predictions: {success_results['correct_predictions']}")
    print(f"Total Images: {success_results['total_images']}")
    print("=" * 60)
    
    # Save detailed results (convert numpy types to JSON-serializable types)
    def convert_to_serializable(obj):
        """Convert numpy types to JSON-serializable types."""
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # Combine both result sets
    combined_results = {
        'per_grasp_metrics': convert_to_serializable(results),
        'success_rate_metrics': convert_to_serializable(success_results)
    }
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    logger.info(f"Detailed results saved to: {results_file}")
    
    # Generate visualizations if requested
    if args.visualize:
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        evaluator.visualize_predictions(
            num_visualizations=args.num_visualizations,
            confidence_threshold=args.confidence_threshold,
            output_dir=viz_dir
        )
    
    logger.info(f"Evaluation completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
