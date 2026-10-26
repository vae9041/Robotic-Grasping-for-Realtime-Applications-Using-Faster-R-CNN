"""
Faster R-CNN Training Script for Grasp Detection
Loads augmented Cornell, original Cornell, and Jacquard datasets
Uses Faster R-CNN with ResNet50 FPN backbone
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
import cv2
import numpy as np
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import logging
import math
from torch.utils.flop_counter import FlopCounterMode


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraspRectangle:
    """Represents an oriented grasp rectangle for IoU and angle comparisons."""

    def __init__(self, center_x: float, center_y: float, width: float, height: float,
                 angle: float = 0.0, format_type: str = 'jacquard'):
        """
        Initialize a grasp rectangle.

        Args:
            center_x: Center x coordinate
            center_y: Center y coordinate
            width: Width of grasp rectangle
            height: Height of grasp rectangle
            angle: Rotation angle in degrees (Jacquard) or radians (Cornell)
            format_type: 'jacquard' or 'cornell'
        """
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height

        if format_type == 'jacquard':
            self.angle = math.radians(angle)  # Convert degrees to radians
        else:
            self.angle = angle  # Already in radians for Cornell

        # Canonicalize to [0, pi) since a rectangle rotated by pi is identical.
        self.angle = self.angle % math.pi

    @classmethod
    def from_bbox(cls, x_min: float, y_min: float, x_max: float, y_max: float, angle: float = 0.0):
        """Create GraspRectangle from axis-aligned bounding box coordinates."""
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        return cls(center_x, center_y, width, height, angle, 'cornell')

    @classmethod
    def from_corners(cls, corners: np.ndarray, y_down: bool = True):
        """
        Build a GraspRectangle from 4 corner points in cyclic order.

        Args:
            corners: (4, 2) array of corner points ordered around the rectangle.
            y_down: If True, the y-axis points downward (standard image convention
                with origin top-left). If False, y points up.
        """
        assert corners.shape == (4, 2)
        corners = corners.astype(np.float32)
        center = corners.mean(axis=0)

        # Use the first side (corner 0 -> corner 1) as the orientation reference.
        # A rectangle is invariant to 180 degree flips, so we canonicalize later.
        edge = corners[1] - corners[0]
        ex, ey = float(edge[0]), float(edge[1])
        if y_down:
            ey = -ey
        angle = math.atan2(ey, ex)
        angle = angle % math.pi

        # Width and height are the two distinct side lengths of the polygon.
        edges = [corners[(i + 1) % 4] - corners[i] for i in range(4)]
        lengths = [float(np.linalg.norm(e)) for e in edges]
        # Keep width/height tied to the edge order used for the angle.
        w = lengths[0]
        h = lengths[1]
        if w == 0:
            w = 1.0
        if h == 0:
            h = 1.0

        return cls(float(center[0]), float(center[1]), w, h, angle, 'cornell')

    def to_corners(self) -> List[Tuple[float, float]]:
        """Convert center representation to 4 corner points."""
        # Half dimensions
        w_half = self.width / 2.0
        h_half = self.height / 2.0

        # Corner offsets in local coordinate system
        corners_local = [
            (-w_half, -h_half),  # Bottom-left
            (w_half, -h_half),   # Bottom-right
            (w_half, h_half),    # Top-right
            (-w_half, h_half)    # Top-left
        ]

        # Rotate and translate to global coordinates
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)

        corners_global = []
        for dx, dy in corners_local:
            # Rotate
            x_rot = dx * cos_angle - dy * sin_angle
            y_rot = dx * sin_angle + dy * cos_angle

            # Translate
            x_global = x_rot + self.center_x
            y_global = y_rot + self.center_y

            corners_global.append((x_global, y_global))

        return corners_global

    def to_bbox(self) -> Tuple[float, float, float, float]:
        """Convert to axis-aligned bounding box (x_min, y_min, x_max, y_max)."""
        corners = self.to_corners()
        x_coords = [corner[0] for corner in corners]
        y_coords = [corner[1] for corner in corners]

        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    def angle_error(self, other: 'GraspRectangle') -> float:
        """
        Return the absolute angle difference with another rectangle in degrees.
        Rectangles are equivalent every 180 degrees, so the error is wrapped to [0, 90].
        """
        delta = (self.angle - other.angle) % math.pi
        if delta > math.pi / 2.0:
            delta = math.pi - delta
        return math.degrees(delta)

    def intersection_over_union(self, other: 'GraspRectangle') -> float:
        """Calculate oriented-polygon IoU with another grasp rectangle."""
        try:
            corners1 = np.array(self.to_corners(), dtype=np.float32)
            corners2 = np.array(other.to_corners(), dtype=np.float32)

            intersection_area = cv2.intersectConvexConvex(corners1, corners2)[0]
            if intersection_area <= 0:
                return 0.0

            area1 = self.width * self.height
            area2 = other.width * other.height

            union_area = area1 + area2 - intersection_area
            if union_area <= 0:
                return 0.0

            return float(intersection_area / union_area)

        except Exception:
            # Fallback to bounding box IoU if cv2 fails for any reason.
            bbox1 = self.to_bbox()
            bbox2 = other.to_bbox()

            x_left = max(bbox1[0], bbox2[0])
            y_top = max(bbox1[1], bbox2[1])
            x_right = min(bbox1[2], bbox2[2])
            y_bottom = min(bbox1[3], bbox2[3])

            if x_right < x_left or y_bottom < y_top:
                return 0.0

            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            union_area = area1 + area2 - intersection_area

            if union_area <= 0:
                return 0.0

            return float(intersection_area / union_area)


class GraspDataset(Dataset):
    """Dataset class for grasp detection that handles multiple data sources"""
    
    def __init__(self, data_sources: List[Dict], transform=None, target_size=(640, 480), max_grasps_per_image=300):
        """
        Args:
            data_sources: List of dictionaries with dataset information
            transform: Optional transforms to apply to images
            target_size: Target image size (width, height)
            max_grasps_per_image: Cap the number of ground-truth grasps per image to avoid RPN OOM
        """
        self.data_sources = data_sources
        self.transform = transform
        self.target_size = target_size
        self.max_grasps_per_image = max_grasps_per_image
        self.samples = []
        
        # Load all samples from different sources
        self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} total samples from {len(data_sources)} data sources")
    
    def _load_samples(self):
        """Load samples from all data sources"""
        for source in self.data_sources:
            source_type = source['type']
            source_path = source['path']
            
            if source_type == 'augmented_cornell':
                self._load_augmented_cornell(source_path)
            elif source_type == 'original_cornell':
                self._load_original_cornell(source_path)
            elif source_type == 'original_jacquard':
                self._load_original_jacquard(source_path)
            else:
                logger.warning(f"Unknown source type: {source_type}")
    
    def _load_augmented_cornell(self, augmented_path):
        """Load augmented Cornell dataset"""
        augmented_dir = Path(augmented_path)
        rgb_dir = augmented_dir / 'rgb'
        depth_dir = augmented_dir / 'depth'
        ann_dir = augmented_dir / 'annotations'
        
        if not all(d.exists() for d in [rgb_dir, depth_dir, ann_dir]):
            logger.warning(f"Augmented Cornell directory structure incomplete: {augmented_path}")
            return
        
        for ann_file in ann_dir.glob('*.json'):
            base_name = ann_file.stem
            rgb_file = rgb_dir / f"{base_name}.png"
            depth_file = depth_dir / f"{base_name}.png"
            
            if rgb_file.exists() and depth_file.exists():
                self.samples.append({
                    'rgb_path': str(rgb_file),
                    'depth_path': str(depth_file),
                    'annotation_path': str(ann_file),
                    'source': 'augmented_cornell',
                    'format': 'json'
                })
        
        logger.info(f"Loaded {len([s for s in self.samples if s['source'] == 'augmented_cornell'])} augmented Cornell samples")
    
    def _load_original_cornell(self, cornell_path):
        """Load original Cornell dataset"""
        cornell_dir = Path(cornell_path)
        
        for folder in sorted(cornell_dir.iterdir()):
            if folder.is_dir() and folder.name != 'backgrounds':
                for file in folder.glob('pcd*.txt'):
                    if 'cpos' not in file.name and 'cneg' not in file.name:
                        base_name = file.stem
                        rgb_file = folder / f"{base_name}r.png"
                        depth_file = folder / f"{base_name}d.tiff"
                        pos_file = folder / f"{base_name}cpos.txt"
                        neg_file = folder / f"{base_name}cneg.txt"
                        
                        if all(f.exists() for f in [rgb_file, depth_file, pos_file]):
                            self.samples.append({
                                'rgb_path': str(rgb_file),
                                'depth_path': str(depth_file),
                                'pos_grasps_path': str(pos_file),
                                'neg_grasps_path': str(neg_file) if neg_file.exists() else None,
                                'source': 'original_cornell',
                                'format': 'cornell'
                            })
        
        logger.info(f"Loaded {len([s for s in self.samples if s['source'] == 'original_cornell'])} original Cornell samples")
    
    def _load_original_jacquard(self, jacquard_path):
        """Load original Jacquard dataset"""
        jacquard_dir = Path(jacquard_path)
        
        for folder in sorted(jacquard_dir.iterdir()):
            if folder.is_dir():
                for subfolder in folder.iterdir():
                    if subfolder.is_dir():
                        for rgb_file in subfolder.glob('*_RGB.png'):
                            base_name = rgb_file.name.replace('_RGB.png', '')
                            depth_file = subfolder / f"{base_name}_perfect_depth.tiff"
                            grasps_file = subfolder / f"{base_name}_grasps.txt"
                            
                            if all(f.exists() for f in [rgb_file, depth_file, grasps_file]):
                                self.samples.append({
                                    'rgb_path': str(rgb_file),
                                    'depth_path': str(depth_file),
                                    'grasps_path': str(grasps_file),
                                    'source': 'original_jacquard',
                                    'format': 'jacquard'
                                })
        
        logger.info(f"Loaded {len([s for s in self.samples if s['source'] == 'original_jacquard'])} original Jacquard samples")
    
    def _load_grasps_from_json(self, json_path):
        """Load grasps from augmented dataset JSON format."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        boxes = []
        labels = []
        angles = []

        for grasp in data['grasps']:
            # Convert grasp to oriented rectangle and take its AABB.
            corners = np.array(grasp['corners'])
            rect = GraspRectangle.from_corners(corners, y_down=True)
            x_min, y_min, x_max, y_max = rect.to_bbox()

            # Ensure valid bounding box
            if x_max > x_min and y_max > y_min:
                boxes.append([x_min, y_min, x_max, y_max])
                angles.append(rect.angle)
                # Use quality score to determine label (1 for good grasp, 0 for background)
                labels.append(1 if grasp['quality'] > 0.5 else 1)  # All grasps are positive for now

        return boxes, labels, angles
    
    def _load_grasps_from_cornell(self, pos_path, neg_path=None):
        """Load grasps from Cornell format files and return AABBs + oriented angles."""
        boxes = []
        labels = []
        angles = []

        def _process_cornell_file(path, is_positive):
            if not os.path.exists(path):
                return
            with open(path, 'r') as f:
                lines = f.readlines()
                for i in range(0, len(lines), 4):
                    if i + 3 < len(lines):
                        corners = []
                        for j in range(4):
                            x, y = map(float, lines[i + j].strip().split())
                            corners.append([x, y])
                        corners = np.array(corners)

                        rect = GraspRectangle.from_corners(corners, y_down=True)
                        x_min, y_min, x_max, y_max = rect.to_bbox()

                        if x_max > x_min and y_max > y_min:
                            boxes.append([x_min, y_min, x_max, y_max])
                            angles.append(rect.angle)
                            labels.append(1 if is_positive else 1)  # Treat all as positives for now

        _process_cornell_file(pos_path, is_positive=True)
        _process_cornell_file(neg_path, is_positive=False)

        return boxes, labels, angles
    
    def _load_grasps_from_jacquard(self, grasps_path):
        """Load grasps from Jacquard format file and return AABBs + oriented angles."""
        boxes = []
        labels = []
        angles = []

        with open(grasps_path, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) >= 5:
                    x, y, theta, w, h = map(float, parts[:5])

                    # Build the oriented rectangle and extract its AABB for the detector target.
                    rect = GraspRectangle(x, y, w, h, theta, format_type='jacquard')
                    x_min, y_min, x_max, y_max = rect.to_bbox()

                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        angles.append(rect.angle)
                        labels.append(1)  # All Jacquard grasps are positive

        return boxes, labels, angles
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load RGB image
        rgb_image = cv2.imread(sample['rgb_path'])
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # Resize image to target size
        rgb_image = cv2.resize(rgb_image, self.target_size)

        # Load grasps based on format
        if sample['format'] == 'json':
            boxes, labels, angles = self._load_grasps_from_json(sample['annotation_path'])
        elif sample['format'] == 'cornell':
            boxes, labels, angles = self._load_grasps_from_cornell(
                sample['pos_grasps_path'],
                sample.get('neg_grasps_path')
            )
        elif sample['format'] == 'jacquard':
            boxes, labels, angles = self._load_grasps_from_jacquard(sample['grasps_path'])
        else:
            boxes, labels, angles = [], [], []

        # Scale boxes to resized image (angles are invariant to scale).
        if len(boxes) > 0:
            # Get original image size for scaling
            original_image = cv2.imread(sample['rgb_path'])
            orig_h, orig_w = original_image.shape[:2]
            scale_x = self.target_size[0] / orig_w
            scale_y = self.target_size[1] / orig_h

            # Scale boxes
            scaled_boxes = []
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                scaled_box = [
                    x_min * scale_x,
                    y_min * scale_y,
                    x_max * scale_x,
                    y_max * scale_y
                ]
                # Clamp to image bounds
                scaled_box[0] = max(0, min(scaled_box[0], self.target_size[0] - 1))
                scaled_box[1] = max(0, min(scaled_box[1], self.target_size[1] - 1))
                scaled_box[2] = max(scaled_box[0] + 1, min(scaled_box[2], self.target_size[0]))
                scaled_box[3] = max(scaled_box[1] + 1, min(scaled_box[3], self.target_size[1]))
                scaled_boxes.append(scaled_box)
            boxes = scaled_boxes

            # Cap the number of ground-truth boxes to avoid RPN box_iou OOM on
            # samples with an extreme number of grasps (some Jacquard files have >1000).
            if len(boxes) > self.max_grasps_per_image:
                rng = np.random.default_rng(idx)
                keep = rng.choice(len(boxes), self.max_grasps_per_image, replace=False)
                keep = keep.tolist()
                boxes = [boxes[i] for i in keep]
                labels = [labels[i] for i in keep]
                angles = [angles[i] for i in keep]

        # Convert to tensors
        if len(boxes) == 0:
            # Handle case with no grasps
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            angles = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            angles = torch.tensor(angles, dtype=torch.float32)

        # Apply transforms if specified
        if self.transform:
            rgb_image = self.transform(rgb_image)
        else:
            # Default transform: convert to tensor and normalize
            rgb_image = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0

        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'angles': angles,
        }
        
        return rgb_image, target


class OrientedFastRCNNPredictor(nn.Module):
    """Faster R-CNN box predictor extended with a continuous angle head."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_features, num_classes)
        self.bbox_pred = nn.Linear(in_features, num_classes * 4)
        # Encode angle as 2-channel (sin, cos) to avoid wrap-around discontinuities.
        # Small MLP to give the angle branch enough capacity.
        self.angle_pred = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        if x.dim() == 4:
            assert x.shape[2:] == (1, 1), f"Expected last two dims to be (1,1), got {x.shape}"
        x = x.flatten(start_dim=1)
        class_logits = self.cls_score(x)
        bbox_regression = self.bbox_pred(x)
        angle = F.normalize(self.angle_pred(x), p=2, dim=1, eps=1e-6)
        return class_logits, bbox_regression, angle


class OrientedRoIHeads(RoIHeads):
    """ROI head that adds an angle regression branch to Faster R-CNN."""
    angle_loss_weight = 1.0

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Override to compute loss_angle on sin/cos encoded ground-truth angles.
        In training returns (result, losses); in evaluation returns result.
        """
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None in training")
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        # Extract box features
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression, angle_pred = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}

        if self.training:
            if labels is None or regression_targets is None or matched_idxs is None:
                raise ValueError("Training samples not generated")

            # Standard classification and bbox regression losses
            labels_all = torch.cat(labels, dim=0)
            regression_targets_all = torch.cat(regression_targets, dim=0)
            classification_loss = F.cross_entropy(class_logits, labels_all)

            sampled_pos_inds = torch.where(labels_all > 0)[0]
            num_total = labels_all.numel()
            N, num_classes = class_logits.shape
            box_regression_per_class = box_regression.reshape(N, num_classes, 4)
            labels_pos = labels_all[sampled_pos_inds]
            box_loss = F.smooth_l1_loss(
                box_regression_per_class[sampled_pos_inds, labels_pos],
                regression_targets_all[sampled_pos_inds],
                beta=1.0 / 9.0,
                reduction="sum",
            ) / max(1, num_total)

            losses["loss_classifier"] = classification_loss
            losses["loss_box_reg"] = box_loss

            # Angle regression loss on foreground proposals only.
            if sampled_pos_inds.numel() > 0:
                offset = 0
                pos_angle_preds = []
                pos_angle_targets = []
                for img_idx, (props, matched_idx, lbl, tgt) in enumerate(zip(proposals, matched_idxs, labels, targets)):
                    img_foreground = torch.where(lbl > 0)[0]
                    if img_foreground.numel() == 0:
                        offset += len(props)
                        continue
                    gt_indices = matched_idx[img_foreground]
                    gt_angles = tgt["angles"][gt_indices]
                    # Avoid in-place modification; keep canonical [0, pi)
                    gt_angles = gt_angles % math.pi
                    pos_angle_targets.append(torch.stack([torch.sin(gt_angles), torch.cos(gt_angles)], dim=1))
                    pos_angle_preds.append(angle_pred[offset + img_foreground])
                    offset += len(props)

                if pos_angle_targets:
                    angle_pred_pos = torch.cat(pos_angle_preds, dim=0)
                    angle_target_pos = torch.cat(pos_angle_targets, dim=0)
                    # Pi-periodic angle loss: 1 - (v·t)^2  ==  sin^2(delta).
                    # Invariant to v -> -v, which is the correct 180-degree symmetry for a grasp.
                    dot = (angle_pred_pos * angle_target_pos).sum(dim=1).clamp(-1.0, 1.0)
                    losses["loss_angle"] = (1.0 - dot * dot).mean() * self.angle_loss_weight
                else:
                    losses["loss_angle"] = angle_pred.sum() * 0.0
            else:
                losses["loss_angle"] = angle_pred.sum() * 0.0

            return result, losses

        # Evaluation path
        boxes, scores, labels_out, angles_out = self.postprocess_detections(
            class_logits, box_regression, angle_pred, proposals, image_shapes
        )
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels_out[i],
                    "scores": scores[i],
                    "angles": angles_out[i],
                }
            )
        return result, losses

    def postprocess_detections(
        self,
        class_logits: torch.Tensor,
        box_regression: torch.Tensor,
        angle_pred: torch.Tensor,
        proposals: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Decode detections and add the predicted angle for each kept box (foreground only)."""
        device = class_logits.device
        N, num_classes = class_logits.shape
        num_fg = num_classes - 1
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]

        pred_scores = F.softmax(class_logits, -1)[:, 1:]  # drop background
        pred_boxes = self.box_coder.decode(box_regression, proposals)  # [N, num_classes*4]
        pred_boxes = pred_boxes.reshape(N, num_classes, 4)[:, 1:].reshape(N, num_fg * 4)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_labels_list = [torch.arange(1, num_classes, device=device).reshape(1, -1).expand(s.shape[0], -1) for s in pred_scores_list]
        angle_pred_list = angle_pred.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_angles = []

        for boxes, scores, labels, angles, image_shape in zip(
            pred_boxes_list, pred_scores_list, pred_labels_list, angle_pred_list, image_shapes
        ):
            n = boxes.shape[0]
            boxes = boxes.reshape(n, num_fg, 4).reshape(-1, 4)
            # Create one row per (box, foreground class) for multi-class NMS filtering
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            angles = angles.unsqueeze(1).expand(-1, num_fg, -1).reshape(-1, 2)

            # Remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels, angles = boxes[inds], scores[inds], labels[inds], angles[inds]

            # Remove empty boxes
            keep = torchvision.ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, angles = boxes[keep], scores[keep], labels[keep], angles[keep]

            # Non-maximum suppression (per class)
            keep = torchvision.ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, angles = boxes[keep], scores[keep], labels[keep], angles[keep]

            # Decode angle from (sin, cos) representation
            decoded_angles = torch.atan2(angles[:, 0], angles[:, 1])
            decoded_angles = decoded_angles % math.pi

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_angles.append(decoded_angles)

        return all_boxes, all_scores, all_labels, all_angles


def get_model(num_classes, freeze_backbone=False, angle_loss_weight=1.0, nms_thresh=None, score_thresh=None):
    """Create Faster R-CNN model with ResNet50 FPN backbone and oriented ROI head.

    Args:
        num_classes: Number of classes (including background)
        freeze_backbone: If True, freeze the backbone feature extractor
    """
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    old_roi = model.roi_heads

    # Replace with the oriented detection head, preserving the original head's hyperparameters.
    new_roi = OrientedRoIHeads(
        old_roi.box_roi_pool,
        old_roi.box_head,
        OrientedFastRCNNPredictor(in_features, num_classes),
        old_roi.proposal_matcher.high_threshold,
        old_roi.proposal_matcher.low_threshold,
        old_roi.fg_bg_sampler.batch_size_per_image,
        old_roi.fg_bg_sampler.positive_fraction,
        old_roi.box_coder.weights,
        score_thresh if score_thresh is not None else old_roi.score_thresh,
        nms_thresh if nms_thresh is not None else old_roi.nms_thresh,
        old_roi.detections_per_img,
    )
    new_roi.angle_loss_weight = angle_loss_weight
    model.roi_heads = new_roi

    # Freeze backbone if requested
    if freeze_backbone:
        freeze_backbone_parameters(model)

    return model


def freeze_backbone_parameters(model):
    """Freeze the backbone feature extractor parameters
    
    This keeps the ResNet50 and FPN parameters fixed while allowing
    training of the RPN and ROI head components.
    """
    # Freeze backbone (ResNet50) parameters
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    logger.info("🔒 Backbone feature extractor parameters frozen")
    logger.info("✅ RPN and ROI head parameters remain trainable")
    
    # Print parameter status
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    rpn_params = sum(p.numel() for p in model.rpn.parameters())
    roi_params = sum(p.numel() for p in model.roi_heads.parameters())
    
    logger.info(f"📊 Parameter breakdown:")
    logger.info(f"   Backbone (frozen): {backbone_params:,} parameters")
    logger.info(f"   RPN (trainable): {rpn_params:,} parameters") 
    logger.info(f"   ROI Head (trainable): {roi_params:,} parameters")
    logger.info(f"   Total trainable: {rpn_params + roi_params:,} parameters")


def print_model_info(model):
    """Print model architecture and parameter information"""
    logger.info("="*70)
    logger.info("MODEL ARCHITECTURE AND PARAMETERS")
    logger.info("="*70)
    
    # Print model structure
    logger.info("Model Architecture:")
    logger.info(str(model))
    
    # Count parameters
    total_params = 0
    trainable_params = 0
    
    logger.info("\n" + "="*70)
    logger.info("PARAMETER BREAKDOWN BY MODULE:")
    logger.info("="*70)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
            status = "TRAINABLE"
        else:
            status = "FROZEN"
        
        logger.info(f"{name:<50} | {param_count:>10,} | {status}")
    
    logger.info("="*70)
    logger.info(f"TOTAL PARAMETERS: {total_params:,}")
    logger.info(f"TRAINABLE PARAMETERS: {trainable_params:,}")
    logger.info(f"FROZEN PARAMETERS: {total_params - trainable_params:,}")
    logger.info(f"TRAINABLE RATIO: {trainable_params/total_params*100:.2f}%")
    
    # Calculate memory usage (approximate)
    model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32 parameter
    logger.info(f"APPROXIMATE MODEL SIZE: {model_size_mb:.2f} MB")
    logger.info("="*70)


def print_model_summary(model):
    """Print a concise model summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: Faster R-CNN with ResNet50 FPN backbone")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: ~{total_params * 4 / (1024 * 1024):.2f} MB")


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images, targets = zip(*batch)
    images = list(images)
    targets = list(targets)
    return images, targets


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50, grad_clip=0.0, flops_log_path=None):
    """Train for one epoch"""
    model.train()
    metric_logger = defaultdict(list)
    flops = None
    flops_per_image = None
    first_batch_size = 0

    for i, (images, targets) in enumerate(tqdm(data_loader, desc=f"Training Epoch {epoch}")):
        # Move to device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass with a one-time FLOP count for the first batch
        if i == 0:
            with FlopCounterMode(display=False) as fc:
                loss_dict = model(images, targets)
            flops = int(fc.get_total_flops())
            first_batch_size = len(images)
            flops_per_image = flops / first_batch_size if first_batch_size > 0 else 0.0
            logger.info(f"FLOPs first batch (bs={first_batch_size}): {flops:,} total, {flops_per_image:.2e}/image")
        else:
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Log losses
        for k, v in loss_dict.items():
            metric_logger[k].append(v.item())
        metric_logger['total_loss'].append(losses.item())

        if i % print_freq == 0:
            avg_loss = np.mean(metric_logger['total_loss'][-print_freq:])
            logger.info(f"Epoch {epoch}, Batch {i}/{len(data_loader)}, Avg Loss: {avg_loss:.4f}")

    # Calculate epoch averages
    epoch_metrics = {k: np.mean(v) for k, v in metric_logger.items()}
    if flops is not None:
        epoch_metrics['flops_per_image'] = flops_per_image
        num_images = len(data_loader.dataset)
        total_flops_estimate = flops_per_image * num_images
        epoch_metrics['total_flops_forward_estimate'] = total_flops_estimate
        # Save FLOP count record if a path is provided
        if flops_log_path is not None:
            flops_log_path = Path(flops_log_path)
            flops_log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "epoch": int(epoch),
                "num_batches": len(data_loader),
                "num_images": num_images,
                "first_batch_flops_total": flops,
                "first_batch_size": first_batch_size,
                "flops_per_image": flops_per_image,
                "total_flops_forward_estimate": total_flops_estimate,
            }
            with open(flops_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
    return epoch_metrics


def evaluate(model, data_loader, device, confidence_threshold: float = 0.5, iou_threshold: float = 0.25, angle_threshold: float = 30.0):
    """
    Evaluate the model using oriented-grasp success rate.
    For each image, the best predicted grasp must satisfy IoU > iou_threshold
    AND |angle difference| <= angle_threshold (degrees) with a GT grasp to count.
    """
    model.eval()  # Set to eval mode for predictions

    total_loss = 0
    num_batches = 0
    correct_predictions = 0
    total_images = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            # Move to device
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get loss (temporarily set to train mode)
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            num_batches += 1

            # Get predictions (set back to eval mode)
            model.eval()
            predictions = model(images)

            # Process each image in the batch
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                total_images += 1

                # Filter predictions by confidence
                keep_indices = pred['scores'] >= confidence_threshold
                pred_boxes = pred['boxes'][keep_indices].cpu().numpy()
                pred_angles = pred['angles'][keep_indices].cpu().numpy()
                pred_scores = pred['scores'][keep_indices].cpu().numpy()

                # Skip if no confident predictions
                if len(pred_boxes) == 0:
                    continue

                # Select best prediction (highest confidence)
                best_pred_idx = int(np.argmax(pred_scores))
                best_pred_box = pred_boxes[best_pred_idx]
                best_pred_angle = float(pred_angles[best_pred_idx])

                # Convert to oriented GraspRectangle
                best_pred_grasp = GraspRectangle.from_bbox(
                    best_pred_box[0], best_pred_box[1],
                    best_pred_box[2], best_pred_box[3], best_pred_angle
                )

                # Get ground truth grasps
                gt_boxes = target['boxes'].cpu().numpy()
                gt_angles = target['angles'].cpu().numpy()

                # Check if best prediction has sufficient AABB IoU and angle with any GT grasp
                image_success = False
                aabb_pred = GraspRectangle.from_bbox(
                    best_pred_box[0], best_pred_box[1],
                    best_pred_box[2], best_pred_box[3], 0.0
                )
                for gt_box, gt_angle in zip(gt_boxes, gt_angles):
                    aabb_gt = GraspRectangle.from_bbox(
                        gt_box[0], gt_box[1], gt_box[2], gt_box[3], 0.0
                    )
                    gt_grasp = GraspRectangle.from_bbox(
                        gt_box[0], gt_box[1], gt_box[2], gt_box[3], float(gt_angle)
                    )
                    iou = aabb_pred.intersection_over_union(aabb_gt)
                    angle_error = best_pred_grasp.angle_error(gt_grasp)
                    if iou > iou_threshold and angle_error <= angle_threshold:
                        image_success = True
                        break

                if image_success:
                    correct_predictions += 1

    # Calculate metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = correct_predictions / total_images if total_images > 0 else 0.0

    return {
        'avg_loss': avg_loss,
        'accuracy': accuracy,
        'correct_predictions': correct_predictions,
        'total_images': total_images
    }


def main():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN for Grasp Detection')
    parser.add_argument('--augmented_cornell_path', type=str, default='./augmented_dataset',
                       help='Path to augmented Cornell dataset')
    parser.add_argument('--cornell_path', type=str, default='./cornell_dataset',
                       help='Path to original Cornell dataset')
    parser.add_argument('--jacquard_path', type=str, default='./jacquard_dataset',
                       help='Path to original Jacquard dataset')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                       help='Weight decay')
    
    # Learning rate scheduler options
    parser.add_argument('--scheduler', type=str, default='step', 
                       choices=['step', 'cosine', 'plateau', 'exponential'],
                       help='Learning rate scheduler type')
    parser.add_argument('--step_size', type=int, default=20,
                       help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma for StepLR and ExponentialLR schedulers')
    
    # Optimizer options
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam', 'adamw'],
                       help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD optimizer')
    parser.add_argument('--angle_loss_weight', type=float, default=1.0,
                       help='Weight for the angle regression MSE loss')
    parser.add_argument('--nms_thresh', type=float, default=None,
                       help='Override NMS threshold for oriented ROI head (default: keep pretrained value)')
    parser.add_argument('--score_thresh', type=float, default=None,
                       help='Override score threshold for oriented ROI head (default: keep pretrained value)')
    parser.add_argument('--eval_confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for counting predictions during validation')
    
    # Training improvements
    parser.add_argument('--warmup_epochs', type=int, default=0,
                       help='Number of warmup epochs')
    parser.add_argument('--grad_clip', type=float, default=0.0,
                       help='Gradient clipping threshold (0 = no clipping)')
    parser.add_argument('--finetune_angle_head', action='store_true',
                       help='Freeze all parameters except the angle head and train only it')
    parser.add_argument('--unfreeze_roi_head', action='store_true',
                       help='Freeze backbone/RPN and fine-tune the ROI box head and all predictor heads')
    parser.add_argument('--unfreeze_all', action='store_true',
                       help='Unfreeze all parameters and reset optimizer for end-to-end fine-tuning')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Directory to save trained models')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                       help='Ratio of data to use for training (rest for validation)')
    parser.add_argument('--show_model_details', action='store_true',
                       help='Show detailed model architecture and parameters')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze the backbone feature extractor (only train RPN and ROI head)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flops_log_path = output_dir / "flops.jsonl"

    # Define data sources
    data_sources = []
    
    if os.path.exists(args.augmented_cornell_path):
        data_sources.append({
            'type': 'augmented_cornell',
            'path': args.augmented_cornell_path
        })
    
    if os.path.exists(args.cornell_path):
        data_sources.append({
            'type': 'original_cornell',
            'path': args.cornell_path
        })
    
    if os.path.exists(args.jacquard_path):
        data_sources.append({
            'type': 'original_jacquard',
            'path': args.jacquard_path
        })
    
    if not data_sources:
        logger.error("No valid data sources found!")
        return
    
    logger.info(f"Using data sources: {[s['type'] for s in data_sources]}")
    
    # Create dataset
    dataset = GraspDataset(data_sources, target_size=(640, 480))
    
    # Split dataset into train and validation
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Create model
    model = get_model(num_classes=2, freeze_backbone=args.freeze_backbone,
                      angle_loss_weight=args.angle_loss_weight,
                      nms_thresh=args.nms_thresh, score_thresh=args.score_thresh)  # Background + grasp
    model.to(device)
    
    # Optional: fine-tune only the angle head or the full ROI head
    if args.finetune_angle_head:
        logger.info("Freezing all parameters except the angle head")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.roi_heads.box_predictor.angle_pred.parameters():
            p.requires_grad = True
    elif args.unfreeze_roi_head:
        logger.info("Freezing backbone/RPN, unfreezing ROI box head and all predictor heads")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.roi_heads.box_head.parameters():
            p.requires_grad = True
        for p in model.roi_heads.box_predictor.parameters():
            p.requires_grad = True
    elif args.unfreeze_all:
        logger.info("Unfreezing all parameters for end-to-end fine-tuning")
        for p in model.parameters():
            p.requires_grad = True
    
    # Print model information
    if args.show_model_details:
        print_model_info(model)
    else:
        print_model_summary(model)
    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Learning rate scheduler - Add options for different schedulers
    if args.scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr*0.01)
    elif args.scheduler == 'plateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    elif args.scheduler == 'exponential':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Training loop
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        # strict=False allows resuming into the new (wider) angle-pred MLP.
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if not (args.finetune_angle_head or args.unfreeze_roi_head or args.unfreeze_all):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_accuracy = checkpoint.get('val_accuracy', 0.0)
            best_val_loss = checkpoint.get('val_loss', float('inf'))
        else:
            logger.info("Fine-tuning: optimizer and best metrics reset")
            start_epoch = 0
            best_val_accuracy = 0.0
            best_val_loss = float('inf')
        logger.info(f"Resumed from epoch {start_epoch} (best val acc {best_val_accuracy:.4f}, best val loss {best_val_loss:.4f})")
    
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch + 1, grad_clip=args.grad_clip, flops_log_path=flops_log_path)
        train_loss = train_metrics['total_loss']
        train_losses.append(train_loss)
        
        # Validate with IoU-based accuracy
        val_metrics = evaluate(model, val_loader, device, confidence_threshold=args.eval_confidence_threshold)
        val_loss = val_metrics['avg_loss']
        val_accuracy = val_metrics['accuracy']
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate
        if args.scheduler == 'plateau':
            lr_scheduler.step(val_loss)  # ReduceLROnPlateau needs the metric
        else:
            lr_scheduler.step()  # Other schedulers don't need arguments
        
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Val Accuracy: {val_accuracy:.4f} ({val_metrics['correct_predictions']}/{val_metrics['total_images']})")
        
        # Save best model based on accuracy (or loss if you prefer)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, output_dir / 'best_model.pth')
            logger.info(f"New best model saved with validation accuracy: {val_accuracy:.4f}")
        
        # Also save if loss improved but accuracy didn't (fallback)
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, output_dir / 'best_loss_model.pth')
            logger.info(f"Best loss model saved with validation loss: {val_loss:.4f}")
        
        # Save latest model
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_dir / f'model_epoch_{epoch + 1}.pth')
    
    # Save final model
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
    }, output_dir / 'final_model.pth')
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU-based Accuracy')
    ax2.set_title('Validation IoU-based Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png')
    plt.close()
    
    logger.info(f"Training completed!")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
