"""
Comprehensive Faster R-CNN Pipeline Visualization
Shows all intermediate stages: FPN features, RPN proposals, anchors, ROI pooling, and predictions
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import os
from pathlib import Path
import cv2
from torchvision import transforms
from torch.nn import functional as F

# Import dataset
from train import GraspDataset

class PipelineVisualizer:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # Load the trained model
        print(f"Loading model from {model_path}...")
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Hook storage
        self.features = {}
        self.register_hooks()
        
    def load_model(self, model_path):
        """Load the trained Faster R-CNN model"""
        # Create model with same architecture
        model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def register_hooks(self):
        """Register forward hooks to capture intermediate outputs"""
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, dict):
                    self.features[name] = output
                elif isinstance(output, list):
                    self.features[name] = output
                elif isinstance(output, tuple):
                    self.features[name] = output
                else:
                    self.features[name] = output.detach()
            return hook
        
        # Hook FPN
        self.model.backbone.fpn.register_forward_hook(get_activation('fpn'))
        
        # Hook RPN
        self.model.rpn.register_forward_hook(get_activation('rpn'))
        
        # Hook ROI heads
        self.model.roi_heads.register_forward_hook(get_activation('roi_heads'))
    
    def load_sample(self, cornell_path, sample_idx=0):
        """Load a sample from Cornell dataset"""
        # Create data sources list
        data_sources = [
            {
                'type': 'original_cornell',
                'path': cornell_path
            }
        ]
        
        dataset = GraspDataset(
            data_sources=data_sources,
            transform=None
        )
        
        print(f"Dataset size: {len(dataset)} samples")
        print(f"Loading sample index: {sample_idx}")
        
        image, target = dataset[sample_idx]
        
        # Get image path for reference
        sample_info = dataset.samples[sample_idx]
        print(f"Sample info: {sample_info}")
        
        return image, target, sample_info
    
    def visualize_fpn_features(self, image_tensor, output_dir):
        """Visualize Feature Pyramid Network outputs at each level"""
        print("\n" + "="*60)
        print("STAGE 1: FEATURE PYRAMID NETWORK (FPN)")
        print("="*60)
        
        # Get FPN features
        with torch.no_grad():
            # Pass single tensor, not list
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            # Get backbone features
            backbone_features = self.model.backbone(image_batch)
        
        fpn_levels = list(backbone_features.keys())
        print(f"FPN levels: {fpn_levels}")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Feature Pyramid Network (FPN) - Multi-Scale Features', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for idx, level in enumerate(fpn_levels):
            if idx >= len(axes):
                break
                
            features = backbone_features[level][0]  # First image in batch
            print(f"\n{level}: Shape {features.shape}")
            
            # Average across channels for visualization
            feature_map = features.mean(dim=0).cpu().numpy()
            
            ax = axes[idx]
            im = ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'{level}\nShape: {features.shape[1:]}', fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            print(f"  - Channels: {features.shape[0]}")
            print(f"  - Spatial size: {features.shape[1]}x{features.shape[2]}")
            print(f"  - Value range: [{features.min():.4f}, {features.max():.4f}]")
        
        # Hide unused subplots
        for idx in range(len(fpn_levels), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, '1_fpn_features.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ FPN features saved to: {output_path}")
        plt.close()
        
        return backbone_features
    
    def visualize_anchors(self, image_tensor, backbone_features, output_dir):
        """Visualize anchor boxes at different scales"""
        print("\n" + "="*60)
        print("STAGE 2: ANCHOR GENERATION")
        print("="*60)
        
        # Get image size
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        h, w = image_np.shape[:2]
        
        # Get anchors from RPN
        anchor_generator = self.model.rpn.anchor_generator
        
        # Get feature map sizes - convert dict values to list
        feature_maps = list(backbone_features.values())
        
        # Create proper ImageList object
        image_batch = image_tensor.unsqueeze(0).to(self.device)
        image_list = torchvision.models.detection.image_list.ImageList(
            image_batch,
            [(h, w)]
        )
        
        # Generate anchors
        anchors = anchor_generator(image_list, feature_maps)
        
        print(f"Total number of anchors: {len(anchors[0])}")
        print(f"Anchor sizes: {anchor_generator.sizes}")
        print(f"Anchor aspect ratios: {anchor_generator.aspect_ratios}")
        
        # Visualize sample anchors from different scales
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle('Anchor Boxes at Different Scales', fontsize=16, fontweight='bold')
        
        # Sample anchors from different regions
        num_anchors = len(anchors[0])
        sample_indices = [
            num_anchors // 4,      # First quarter
            num_anchors // 2,      # Middle
            3 * num_anchors // 4,  # Third quarter
            num_anchors - 1000     # Near end
        ]
        
        for idx, sample_idx in enumerate(sample_indices):
            ax = axes[idx // 2, idx % 2]
            ax.imshow(image_np)
            
            # Show 20 consecutive anchors from this region
            start_idx = max(0, sample_idx)
            end_idx = min(num_anchors, start_idx + 20)
            
            for anchor_idx in range(start_idx, end_idx):
                anchor = anchors[0][anchor_idx].cpu().numpy()
                x1, y1, x2, y2 = anchor
                width = x2 - x1
                height = y2 - y1
                
                rect = patches.Rectangle((x1, y1), width, height,
                                        linewidth=1.5, edgecolor='red',
                                        facecolor='none', alpha=0.7)
                ax.add_patch(rect)
            
            ax.set_title(f'Anchors {start_idx}-{end_idx} (Sample Region {idx+1})', 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, '2_anchors.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Anchor visualization saved to: {output_path}")
        plt.close()
        
        return anchors
    
    def visualize_rpn_proposals(self, image_tensor, target, output_dir):
        """Visualize RPN proposals"""
        print("\n" + "="*60)
        print("STAGE 3: REGION PROPOSAL NETWORK (RPN)")
        print("="*60)
        
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            targets = [{'boxes': target['boxes'].to(self.device),
                       'labels': target['labels'].to(self.device)}]
            
            # Get image features
            features = self.model.backbone(image_batch)
            
            # Get RPN proposals
            image_shapes = [image_batch.shape[-2:]]
            image_list = torchvision.models.detection.image_list.ImageList(image_batch, image_shapes)
            
            proposals, proposal_losses = self.model.rpn(image_list, features, targets)
        
        proposals = proposals[0].cpu()
        print(f"Number of proposals: {len(proposals)}")
        print(f"Proposal shape: {proposals.shape}")
        
        # Create visualization with different confidence levels
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle('RPN Proposals at Different Stages', fontsize=16, fontweight='bold')
        
        # Top proposals at different quantities
        num_proposals_list = [10, 50, 100, len(proposals)]
        
        for idx, num_props in enumerate(num_proposals_list):
            ax = axes[idx // 2, idx % 2]
            ax.imshow(image_np)
            
            # Draw proposals
            for i in range(min(num_props, len(proposals))):
                box = proposals[i].numpy()
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                rect = patches.Rectangle((x1, y1), width, height,
                                        linewidth=2, edgecolor='yellow',
                                        facecolor='none', alpha=0.5)
                ax.add_patch(rect)
            
            # Draw ground truth
            for box in target['boxes']:
                box = box.numpy()
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                rect = patches.Rectangle((x1, y1), width, height,
                                        linewidth=3, edgecolor='lime',
                                        facecolor='none', alpha=0.8)
                ax.add_patch(rect)
            
            ax.set_title(f'Top {min(num_props, len(proposals))} Proposals\n(Yellow: Proposals, Green: Ground Truth)', 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, '3_rpn_proposals.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ RPN proposals saved to: {output_path}")
        plt.close()
        
        return proposals
    
    def visualize_roi_pooling(self, image_tensor, output_dir):
        """Visualize ROI pooling features"""
        print("\n" + "="*60)
        print("STAGE 4: ROI POOLING")
        print("="*60)
        
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            
            # Get features and proposals
            features = self.model.backbone(image_batch)
            image_shapes = [image_batch.shape[-2:]]
            image_list = torchvision.models.detection.image_list.ImageList(image_batch, image_shapes)
            
            proposals, _ = self.model.rpn(image_list, features, None)
            
            # Get ROI features
            box_features = self.model.roi_heads.box_roi_pool(
                features, proposals, image_shapes
            )
            
            print(f"ROI pooled features shape: {box_features.shape}")
            print(f"Number of ROIs: {box_features.shape[0]}")
            print(f"Feature channels: {box_features.shape[1]}")
            print(f"Pooled size: {box_features.shape[2]}x{box_features.shape[3]}")
        
        # Visualize features from top ROIs
        num_rois_to_show = min(12, box_features.shape[0])
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle('ROI Pooling Features (Top Proposals)', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for i in range(num_rois_to_show):
            roi_features = box_features[i]  # Shape: [channels, 7, 7]
            
            # Average across channels
            feature_map = roi_features.mean(dim=0).cpu().numpy()
            
            ax = axes[i]
            im = ax.imshow(feature_map, cmap='hot')
            ax.set_title(f'ROI {i+1}', fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(num_rois_to_show, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, '4_roi_pooling.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ ROI pooling features saved to: {output_path}")
        plt.close()
        
        return box_features
    
    def visualize_final_predictions(self, image_tensor, target, output_dir, confidence_threshold=0.5):
        """Visualize final predictions with all stages"""
        print("\n" + "="*60)
        print("STAGE 5: FINAL PREDICTIONS")
        print("="*60)
        
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            predictions = self.model(image_batch)
        
        pred = predictions[0]
        
        # Filter by confidence
        keep = pred['scores'] > confidence_threshold
        boxes = pred['boxes'][keep].cpu()
        scores = pred['scores'][keep].cpu()
        labels = pred['labels'][keep].cpu()
        
        print(f"Predictions above {confidence_threshold} confidence: {len(boxes)}")
        print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle('Final Predictions Analysis', fontsize=16, fontweight='bold')
        
        # 1. All predictions with scores
        ax = axes[0, 0]
        ax.imshow(image_np)
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box.numpy()
            width = x2 - x1
            height = y2 - y1
            
            color = 'red' if score > 0.8 else 'orange' if score > 0.6 else 'yellow'
            rect = patches.Rectangle((x1, y1), width, height,
                                    linewidth=2, edgecolor=color,
                                    facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{score:.2f}', color=color, fontsize=10,
                   fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        ax.set_title(f'All Predictions (n={len(boxes)})', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # 2. High confidence predictions only
        ax = axes[0, 1]
        ax.imshow(image_np)
        high_conf = scores > 0.7
        for box, score in zip(boxes[high_conf], scores[high_conf]):
            x1, y1, x2, y2 = box.numpy()
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle((x1, y1), width, height,
                                    linewidth=3, edgecolor='red',
                                    facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=10,
                   fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        ax.set_title(f'High Confidence (>0.7, n={high_conf.sum()})', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # 3. Ground truth
        ax = axes[1, 0]
        ax.imshow(image_np)
        for box in target['boxes']:
            box = box.numpy()
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle((x1, y1), width, height,
                                    linewidth=3, edgecolor='lime',
                                    facecolor='none', alpha=0.8)
            ax.add_patch(rect)
        ax.set_title(f'Ground Truth (n={len(target["boxes"])})', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # 4. Comparison: predictions vs ground truth
        ax = axes[1, 1]
        ax.imshow(image_np)
        
        # Draw ground truth in green
        for box in target['boxes']:
            box = box.numpy()
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle((x1, y1), width, height,
                                    linewidth=3, edgecolor='lime',
                                    facecolor='none', alpha=0.6)
            ax.add_patch(rect)
        
        # Draw predictions in red
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.numpy()
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle((x1, y1), width, height,
                                    linewidth=2, edgecolor='red',
                                    facecolor='none', alpha=0.6)
            ax.add_patch(rect)
        
        ax.set_title('Overlay: Predictions (Red) vs Ground Truth (Green)', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, '5_final_predictions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Final predictions saved to: {output_path}")
        plt.close()
        
        return predictions
    
    def create_summary_visualization(self, image_tensor, output_dir):
        """Create a summary showing all pipeline stages"""
        print("\n" + "="*60)
        print("CREATING PIPELINE SUMMARY")
        print("="*60)
        
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Complete Faster R-CNN Pipeline - End-to-End Visualization', 
                    fontsize=18, fontweight='bold')
        
        # Original image
        ax = fig.add_subplot(gs[0, :2])
        ax.imshow(image_np)
        ax.set_title('Input Image', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add text description of pipeline stages
        ax_text = fig.add_subplot(gs[0, 2:])
        ax_text.axis('off')
        pipeline_text = """
        FASTER R-CNN PIPELINE STAGES:
        
        1. BACKBONE (ResNet50):
           • Extract hierarchical features
           • Multiple convolutional layers
        
        2. FEATURE PYRAMID NETWORK (FPN):
           • Multi-scale feature maps
           • Levels: P2, P3, P4, P5, P6
        
        3. REGION PROPOSAL NETWORK (RPN):
           • Generate object proposals
           • Anchor-based detection
        
        4. ROI POOLING:
           • Fixed-size feature extraction
           • 7x7 pooled features per ROI
        
        5. DETECTION HEAD:
           • Classification (grasp/background)
           • Bounding box regression
        """
        ax_text.text(0.1, 0.9, pipeline_text, fontsize=11, verticalalignment='top',
                    fontfamily='monospace', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, '0_pipeline_summary.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Pipeline summary saved to: {output_path}")
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize complete Faster R-CNN pipeline')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--cornell_path', type=str, required=True,
                       help='Path to Cornell dataset')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Index of sample to visualize')
    parser.add_argument('--output_dir', type=str, default='./pipeline_visualization',
                       help='Directory to save visualizations')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("FASTER R-CNN COMPLETE PIPELINE VISUALIZATION")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.cornell_path}")
    print(f"Sample index: {args.sample_idx}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    # Initialize visualizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    visualizer = PipelineVisualizer(args.model_path, device)
    
    # Load sample
    image, target, sample_path = visualizer.load_sample(args.cornell_path, args.sample_idx)
    print(f"\nImage shape: {image.shape}")
    print(f"Number of ground truth grasps: {len(target['boxes'])}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 0. Pipeline summary
    visualizer.create_summary_visualization(image, args.output_dir)
    
    # 1. FPN features
    backbone_features = visualizer.visualize_fpn_features(image, args.output_dir)
    
    # 2. Anchors
    anchors = visualizer.visualize_anchors(image, backbone_features, args.output_dir)
    
    # 3. RPN proposals
    proposals = visualizer.visualize_rpn_proposals(image, target, args.output_dir)
    
    # 4. ROI pooling
    roi_features = visualizer.visualize_roi_pooling(image, args.output_dir)
    
    # 5. Final predictions
    predictions = visualizer.visualize_final_predictions(
        image, target, args.output_dir, args.confidence_threshold
    )
    
    print("\n" + "="*60)
    print("✅ PIPELINE VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\n📁 All visualizations saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  0. 0_pipeline_summary.png - Overview of the pipeline")
    print("  1. 1_fpn_features.png - Feature Pyramid Network features")
    print("  2. 2_anchors.png - Anchor boxes at different scales")
    print("  3. 3_rpn_proposals.png - Region Proposal Network outputs")
    print("  4. 4_roi_pooling.png - ROI pooling features")
    print("  5. 5_final_predictions.png - Final detection results")
    print("="*60)


if __name__ == '__main__':
    main()
