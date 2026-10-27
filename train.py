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
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
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


from copy import copy
from torch.utils.data import Subset
from grasp_geometry import GraspRectangle, letterbox, transform_grasps
from grasp_data import (discover_samples, load_grasps, create_split, validate_split,
                        read_split, select_indices, split_digest)
from grasp_metrics import score_image, summarize
from oriented_model import (GRASP_FORMAT, OrientedFastRCNNPredictor, OrientedRoIHeads,
                            install_grasp_transform, load_training_weights)


class GraspDataset(Dataset):
    """Positive oriented grasps; no image stretching or negative-as-positive targets."""
    def __init__(self, data_sources, transform=None, target_size=(640, 480), max_grasps_per_image=300):
        self.data_sources = data_sources
        self.transform = transform
        self.target_size = target_size
        self.max_grasps_per_image = max_grasps_per_image
        self.samples = discover_samples(data_sources)
        logger.info('Loaded %d images', len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample['rgb_path'])
        if image is None:
            raise ValueError(f"Cannot read RGB image: {sample['rgb_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, scale, offset = letterbox(image, self.target_size)
        grasps = transform_grasps(load_grasps(sample), scale, offset)
        if self.max_grasps_per_image and len(grasps) > self.max_grasps_per_image:
            keep = np.random.default_rng(idx).choice(len(grasps), self.max_grasps_per_image, replace=False)
            grasps = grasps[keep]
        boxes = np.asarray([GraspRectangle.from_array(g).to_bbox() for g in grasps], dtype=np.float32).reshape(-1, 4)
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self.target_size[0])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self.target_size[1])
            valid = (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
            boxes, grasps = boxes[valid], grasps[valid]
        target = dict(boxes=torch.from_numpy(boxes), grasps=torch.from_numpy(grasps),
                      angles=torch.from_numpy(grasps[:, 4].copy()),
                      labels=torch.ones(len(grasps), dtype=torch.int64))
        image = self.transform(image) if self.transform else torch.from_numpy(image).permute(2, 0, 1).float() / 255.
        return image, target


def build_training_split(dataset, train_ratio, output_dir, manifest_path=None, resume=None):
    """Split original identities first, then attach only training-derived augmentations."""
    output_path = Path(output_dir) / 'dataset_split.json'
    checkpoint_split = None
    if resume:
        checkpoint_split = torch.load(resume, map_location='cpu', weights_only=False).get('split_manifest')
    if manifest_path:
        manifest = read_split(manifest_path)
    elif checkpoint_split:
        manifest = checkpoint_split
    elif output_path.exists():
        manifest = read_split(output_path)
    else:
        manifest = create_split(dataset.samples, train_ratio)
    validate_split(manifest, dataset.samples)
    if checkpoint_split and split_digest(checkpoint_split) != split_digest(manifest):
        raise ValueError('Resume checkpoint and requested split differ')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + '\n')
    train_indices = select_indices(dataset.samples, manifest, 'train')
    val_indices = select_indices(dataset.samples, manifest, 'val')
    validation_dataset = copy(dataset)
    validation_dataset.max_grasps_per_image = None
    logger.info('Saved split to %s; validation contains original images only', output_path)
    return Subset(dataset, train_indices), Subset(validation_dataset, val_indices), manifest


def checkpoint_info(manifest, parent_checkpoint=None):
    provenance = 'grouped_original_split'
    if parent_checkpoint is not None:
        provenance = parent_checkpoint.get('holdout_provenance',
                     'grouped_original_split' if parent_checkpoint.get('grasp_format') == GRASP_FORMAT
                     else 'unverified_legacy_warm_start')
    return dict(grasp_format=GRASP_FORMAT, split_manifest=manifest,
                split_digest=split_digest(manifest), holdout_provenance=provenance)


def get_model(num_classes, freeze_backbone=False, angle_loss_weight=1.0, nms_thresh=None, score_thresh=None, pretrained=True):
    """Create Faster R-CNN model with ResNet50 FPN backbone and oriented ROI head.

    Args:
        num_classes: Number of classes (including background)
        freeze_backbone: If True, freeze the backbone feature extractor
    """
    if pretrained:
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT", weights_backbone=None)
    else:
        # The torchvision convenience factory uses trainable BatchNorm when no
        # weights are requested. Build the SAME FrozenBatchNorm backbone as the
        # pretrained detector, without downloading weights during checkpoint load.
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        from torchvision.ops.misc import FrozenBatchNorm2d
        backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=None, trainable_layers=3)
        model = FasterRCNN(backbone, num_classes=num_classes)
        for module in model.modules():
            if isinstance(module, FrozenBatchNorm2d):
                module.eps = 0.0  # matches torchvision's pretrained COCO_V1 detector

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
    install_grasp_transform(model)

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


def evaluate(model, data_loader, device, confidence_threshold=0.5, iou_threshold=0.25,
             angle_threshold=30.0, top_k=1):
    """True rotated geometry and any-valid-GT matching, shared with standalone eval."""
    records, losses = [], []
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='Evaluating'):
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            model.train()
            # Freeze batch statistics during validation-loss computation.
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
            losses.append(float(sum(model(images, targets).values()).item()))
            model.eval()
            predictions = model(images)
            for prediction, target in zip(predictions, targets):
                prediction = {key: value.cpu().numpy() for key, value in prediction.items()}
                target = {key: value.cpu().numpy() for key, value in target.items()}
                records.append(score_image(prediction, target, confidence_threshold,
                                           iou_threshold, angle_threshold, top_k, compute_per_grasp=False))
    metrics = summarize(records, confidence_threshold, iou_threshold, angle_threshold, top_k)
    metrics['avg_loss'] = float(np.mean(losses)) if losses else 0.
    return metrics


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
    parser.add_argument('--scheduler', type=str, default=None,
                       choices=['step', 'cosine', 'plateau', 'exponential'],
                       help='Learning rate scheduler type')
    parser.add_argument('--step_size', type=int, default=20,
                       help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma for StepLR and ExponentialLR schedulers')
    
    # Optimizer options
    parser.add_argument('--optimizer', type=str, default=None,
                       choices=['sgd', 'adam', 'adamw'],
                       help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD optimizer')
    parser.add_argument('--angle_loss_weight', type=float, default=1.0,
                       help='Weight for the pi-periodic angle loss')
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
                       help='Train only the new angle and grasp-dimension heads')
    parser.add_argument('--unfreeze_roi_head', action='store_true',
                       help='Freeze backbone/RPN and fine-tune the ROI box head and all predictor heads')
    parser.add_argument('--unfreeze_all', action='store_true',
                       help='Unfreeze all parameters and reset optimizer for end-to-end fine-tuning')
    parser.add_argument('--split_manifest', type=str, default=None, help='Saved grouped train/validation split')
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
    if (args.finetune_angle_head or args.unfreeze_roi_head) and not args.resume:
        parser.error('Fine-tuning requires --resume with a trained detector checkpoint')
    if args.num_epochs < 1 or args.batch_size < 1:
        parser.error('num_epochs and batch_size must be positive')
    
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
    
    train_dataset, val_dataset, manifest = build_training_split(
        dataset, args.train_ratio, output_dir, args.split_manifest, args.resume)
    parent_checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    args.optimizer = args.optimizer or (parent_checkpoint or {}).get('optimizer_name', 'sgd')
    args.scheduler = args.scheduler or (parent_checkpoint or {}).get('scheduler_name', 'step')
    continuing = args.resume and not (args.finetune_angle_head or args.unfreeze_roi_head or args.unfreeze_all)
    if continuing:
        for name in ('optimizer', 'scheduler'):
            saved = parent_checkpoint.get(name + '_name')
            if saved and saved != getattr(args, name):
                raise ValueError(f'Resume requires the saved {name} {saved}; use a fine-tuning flag to change it')
    checkpoint_metadata = checkpoint_info(manifest, parent_checkpoint)
    checkpoint_metadata.update(optimizer_name=args.optimizer, scheduler_name=args.scheduler)

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
                      nms_thresh=args.nms_thresh, score_thresh=args.score_thresh,
                      pretrained=not bool(args.resume))  # Background + grasp
    model.to(device)
    
    # Optional: fine-tune only the angle head or the full ROI head
    if args.finetune_angle_head:
        logger.info("Training angle and oriented-dimension heads; detector weights frozen")
        for p in model.parameters():
            p.requires_grad = False
        for head in (model.roi_heads.box_predictor.angle_pred, model.roi_heads.box_predictor.grasp_pred):
            for p in head.parameters():
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
    if continuing and 'trainable_parameter_names' in parent_checkpoint:
        saved_names = set(parent_checkpoint['trainable_parameter_names'])
        if not saved_names <= set(dict(model.named_parameters())):
            raise ValueError('Saved trainable parameter names differ from the model architecture')
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name in saved_names
    checkpoint_metadata['trainable_parameter_names'] = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad]
    
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
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.num_epochs - args.warmup_epochs), eta_min=args.lr*0.01)
    elif args.scheduler == 'plateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
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
        checkpoint = parent_checkpoint
        load_training_weights(model, checkpoint, allow_legacy=(args.finetune_angle_head or args.unfreeze_roi_head or args.unfreeze_all))
        if not (args.finetune_angle_head or args.unfreeze_roi_head or args.unfreeze_all):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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
        if epoch < args.warmup_epochs:
            for group in optimizer.param_groups:
                group['lr'] = args.lr * (epoch + 1) / args.warmup_epochs
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
        if epoch < args.warmup_epochs:
            pass
        elif args.scheduler == 'plateau':
            lr_scheduler.step(val_loss)  # ReduceLROnPlateau needs the metric
        else:
            lr_scheduler.step()  # Other schedulers don't need arguments
        
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Val Accuracy: {val_accuracy:.4f} ({val_metrics['correct_predictions']}/{val_metrics['total_images']})")
        logger.info('AABB success: %.4f; angle success given AABB: %.4f',
                    val_metrics['aabb_accuracy'], val_metrics['angle_success_given_aabb'])
        
        # Save best model based on accuracy (or loss if you prefer)
        if val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and val_loss < best_val_loss):
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            torch.save({
                **checkpoint_metadata,
                "scheduler_state_dict": lr_scheduler.state_dict(),
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
                **checkpoint_metadata,
                "scheduler_state_dict": lr_scheduler.state_dict(),
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
                **checkpoint_metadata,
                "scheduler_state_dict": lr_scheduler.state_dict(),
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_dir / f'model_epoch_{epoch + 1}.pth')
    
    # Save final model
    torch.save({
        **checkpoint_metadata,
        "scheduler_state_dict": lr_scheduler.state_dict(),
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
    ax2.set_ylabel('Rotated Grasp Success')
    ax2.set_title('Validation Rotated Grasp Success')
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
