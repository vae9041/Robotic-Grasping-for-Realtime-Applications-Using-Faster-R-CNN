"""
Faster R-CNN Model Architecture Visualization
Creates detailed visualizations of the Faster R-CNN model using torchviz and custom diagrams
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchviz import make_dot
import graphviz
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os
from pathlib import Path


def get_model():
    """Load the Faster R-CNN model"""
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


def create_torchviz_visualization():
    """Create a computational graph visualization using torchviz"""
    print("Creating torchviz computational graph visualization...")
    
    # Load model
    model = get_model()
    model.eval()
    
    # Create dummy input
    batch_size = 1
    channels = 3
    height = 480
    width = 640
    
    # Dummy input tensor
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    # Forward pass to get output
    with torch.no_grad():
        # For visualization, we need inference mode
        model.eval()
        output = model([dummy_input])
    
    # Get the first output for visualization
    sample_output = output[0]
    
    # Create visualization for the boxes (most informative output)
    if 'boxes' in sample_output and len(sample_output['boxes']) > 0:
        # Take first prediction for graph generation
        first_box = sample_output['boxes'][0]
        
        # Create graph
        graph = make_dot(first_box.mean(), params=dict(model.named_parameters()))
        graph.attr(rankdir='TB')  # Top to bottom layout
        graph.attr('node', shape='box')
        graph.attr('graph', size='20,30')
        
        # Save the graph
        output_dir = Path('./model_visualizations')
        output_dir.mkdir(exist_ok=True)
        
        graph_path = output_dir / 'fasterrcnn_computational_graph'
        graph.render(graph_path, format='png', cleanup=True)
        print(f"✅ Computational graph saved to: {graph_path}.png")
        
        return str(graph_path) + '.png'
    
    return None


def create_architecture_diagram():
    """Create a custom architecture diagram using matplotlib"""
    print("Creating custom architecture diagram...")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#E8F4FD',
        'backbone': '#B3E5FC',
        'fpn': '#81C784',
        'rpn': '#FFB74D',
        'roi': '#F48FB1',
        'output': '#FFCDD2'
    }
    
    # Helper function to create boxes
    def create_box(x, y, width, height, text, color, fontsize=10):
        box = FancyBboxPatch((x, y), width, height,
                           boxstyle="round,pad=0.1",
                           facecolor=color,
                           edgecolor='black',
                           linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text, 
               ha='center', va='center', fontsize=fontsize, weight='bold')
    
    # Helper function to create arrows
    def create_arrow(x1, y1, x2, y2, text=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        if text:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, text, fontsize=8, ha='center')
    
    # Input
    create_box(1, 10, 3, 1.5, 'Input Image\n640×480×3', colors['input'], 11)
    
    # Backbone (ResNet50)
    create_box(6, 9.5, 4, 2.5, 'ResNet50 Backbone\n23.5M params\n\nconv1 → layer1 → layer2\n→ layer3 → layer4', 
               colors['backbone'], 10)
    
    # Feature Pyramid Network
    create_box(12, 9.5, 3, 2.5, 'Feature Pyramid\nNetwork (FPN)\n3.3M params', 
               colors['fpn'], 10)
    
    # Feature maps from FPN
    feature_levels = ['P2', 'P3', 'P4', 'P5', 'P6']
    for i, level in enumerate(feature_levels):
        y_pos = 7.5 - i * 0.8
        create_box(12.5, y_pos, 2, 0.6, f'{level}\n256 channels', colors['fpn'], 8)
    
    # RPN
    create_box(2, 6.5, 4, 2, 'Region Proposal Network\n(RPN)\n0.6M params\n\nAnchor Generation\n↓\nClassification + Regression', 
               colors['rpn'], 9)
    
    # ROI Pooling and Head
    create_box(8, 6.5, 4, 2, 'ROI Head\n13.9M params\n\nROI Align\n↓\nFC Layers\n↓\nClassification + Regression', 
               colors['roi'], 9)
    
    # Outputs
    create_box(1, 2.5, 3, 1.5, 'Object Proposals\n~2000 boxes', colors['output'], 9)
    create_box(6, 2.5, 3, 1.5, 'Classification\n2 classes\n(background + grasp)', colors['output'], 9)
    create_box(11, 2.5, 3, 1.5, 'Bounding Box\nRegression\n4 coordinates', colors['output'], 9)
    
    # Add arrows
    create_arrow(4, 10.75, 6, 10.75)  # Input to backbone
    create_arrow(10, 10.75, 12, 10.75)  # Backbone to FPN
    create_arrow(13.5, 9.5, 13.5, 8.3)  # FPN to feature maps
    
    # FPN to RPN
    create_arrow(12.5, 8.5, 6, 7.5)  # FPN to RPN
    create_arrow(12.5, 7.5, 12, 7.5)  # FPN to ROI Head
    
    # RPN to ROI Head
    create_arrow(6, 7.5, 8, 7.5)  # RPN proposals to ROI Head
    
    # ROI Head to outputs
    create_arrow(8.5, 6.5, 7.5, 4)  # To classification
    create_arrow(10.5, 6.5, 12.5, 4)  # To bbox regression
    
    # RPN to proposals
    create_arrow(3, 6.5, 2.5, 4)  # RPN to proposals
    
    # Title and annotations
    ax.text(8, 11.5, 'Faster R-CNN Architecture for Grasp Detection', 
           fontsize=16, weight='bold', ha='center')
    
    # Parameter summary
    param_text = ('Total Parameters: 41.3M\n'
                 'Trainable: 41.1M (99.46%)\n'
                 'Model Size: ~157 MB')
    ax.text(0.5, 1, param_text, fontsize=10, va='bottom',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.7))
    
    # Performance summary  
    perf_text = ('Performance:\n'
                'Cornell: 97.06% success\n'
                'Jacquard: 86.68% success')
    ax.text(13.5, 1, perf_text, fontsize=10, va='bottom',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the diagram
    output_dir = Path('./model_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    diagram_path = output_dir / 'fasterrcnn_architecture_diagram.png'
    plt.savefig(diagram_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Architecture diagram saved to: {diagram_path}")
    return str(diagram_path)


def create_detailed_component_diagram():
    """Create detailed component-wise diagram"""
    print("Creating detailed component diagram...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # ResNet50 Backbone Details
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('ResNet50 Backbone Components', fontsize=14, weight='bold', pad=20)
    
    backbone_layers = [
        ('Conv1', '64 filters, 7×7', 9),
        ('Layer1', '3 blocks, 256 channels', 7.5),
        ('Layer2', '4 blocks, 512 channels', 6),
        ('Layer3', '6 blocks, 1024 channels', 4.5),
        ('Layer4', '3 blocks, 2048 channels', 3)
    ]
    
    for i, (name, desc, y_pos) in enumerate(backbone_layers):
        ax1.add_patch(FancyBboxPatch((1, y_pos-0.4), 8, 0.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#B3E5FC',
                                   edgecolor='black'))
        ax1.text(5, y_pos, f'{name}: {desc}', ha='center', va='center', fontsize=10)
        
        if i < len(backbone_layers) - 1:
            ax1.annotate('', xy=(5, y_pos-0.6), xytext=(5, y_pos-0.4),
                        arrowprops=dict(arrowstyle='->', lw=1.5))
    
    # FPN Details
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Feature Pyramid Network', fontsize=14, weight='bold', pad=20)
    
    fpn_features = ['P2 (1/4)', 'P3 (1/8)', 'P4 (1/16)', 'P5 (1/32)', 'P6 (1/64)']
    for i, feature in enumerate(fpn_features):
        y_pos = 8 - i * 1.5
        ax2.add_patch(FancyBboxPatch((2, y_pos-0.3), 6, 0.6,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#81C784',
                                   edgecolor='black'))
        ax2.text(5, y_pos, f'{feature}\n256 channels', ha='center', va='center', fontsize=9)
    
    # RPN Details
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('Region Proposal Network', fontsize=14, weight='bold', pad=20)
    
    rpn_components = [
        ('Anchor Generation', '3 scales × 3 ratios', 8),
        ('3×3 Conv', '256 channels', 6.5),
        ('Classification Head', '2 classes (obj/bg)', 5),
        ('Regression Head', '4 coordinates (x,y,w,h)', 3.5),
        ('NMS', 'Top proposals selection', 2)
    ]
    
    for name, desc, y_pos in rpn_components:
        ax3.add_patch(FancyBboxPatch((1, y_pos-0.4), 8, 0.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#FFB74D',
                                   edgecolor='black'))
        ax3.text(5, y_pos, f'{name}: {desc}', ha='center', va='center', fontsize=9)
    
    # ROI Head Details
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    ax4.set_title('ROI Head Components', fontsize=14, weight='bold', pad=20)
    
    roi_components = [
        ('ROI Align', '7×7 pooling', 8),
        ('FC6', '12.8M params → 1024', 6.5),
        ('FC7', '1M params → 1024', 5.5),
        ('Classification', '2 classes', 4),
        ('Box Regression', '8 coordinates', 3)
    ]
    
    for name, desc, y_pos in roi_components:
        ax4.add_patch(FancyBboxPatch((1, y_pos-0.4), 8, 0.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#F48FB1',
                                   edgecolor='black'))
        ax4.text(5, y_pos, f'{name}: {desc}', ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save the detailed diagram
    output_dir = Path('./model_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    detailed_path = output_dir / 'fasterrcnn_detailed_components.png'
    plt.savefig(detailed_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Detailed components diagram saved to: {detailed_path}")
    return str(detailed_path)


def create_parameter_breakdown_chart():
    """Create a parameter breakdown visualization"""
    print("Creating parameter breakdown chart...")
    
    # Parameter data
    components = ['ResNet50\nBackbone', 'Feature\nPyramid\nNetwork', 'Region\nProposal\nNetwork', 'ROI\nHead']
    params = [23.5, 3.3, 0.6, 13.9]  # in millions
    colors = ['#B3E5FC', '#81C784', '#FFB74D', '#F48FB1']
    
    # Create pie chart and bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Pie chart
    wedges, texts, autotexts = ax1.pie(params, labels=components, colors=colors, 
                                      autopct='%1.1f%%', startangle=90,
                                      textprops={'fontsize': 11})
    ax1.set_title('Parameter Distribution\nTotal: 41.3M Parameters', 
                  fontsize=14, weight='bold', pad=20)
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    # Bar chart
    bars = ax2.bar(range(len(components)), params, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Components', fontsize=12, weight='bold')
    ax2.set_ylabel('Parameters (Millions)', fontsize=12, weight='bold')
    ax2.set_title('Parameter Count by Component', fontsize=14, weight='bold', pad=20)
    ax2.set_xticks(range(len(components)))
    ax2.set_xticklabels([comp.replace('\n', ' ') for comp in components], fontsize=10)
    
    # Add value labels on bars
    for i, (bar, param) in enumerate(zip(bars, params)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{param}M', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax2.set_ylim(0, max(params) * 1.2)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the chart
    output_dir = Path('./model_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    chart_path = output_dir / 'fasterrcnn_parameter_breakdown.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Parameter breakdown chart saved to: {chart_path}")
    return str(chart_path)


def main():
    """Main function to create all visualizations"""
    print("🎨 Creating Faster R-CNN Model Visualizations...")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path('./model_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    created_files = []
    
    try:
        # 1. Computational graph (torchviz)
        graph_path = create_torchviz_visualization()
        if graph_path:
            created_files.append(graph_path)
    except Exception as e:
        print(f"❌ Error creating computational graph: {e}")
    
    try:
        # 2. Architecture diagram
        arch_path = create_architecture_diagram()
        created_files.append(arch_path)
    except Exception as e:
        print(f"❌ Error creating architecture diagram: {e}")
    
    try:
        # 3. Detailed component diagram
        detail_path = create_detailed_component_diagram()
        created_files.append(detail_path)
    except Exception as e:
        print(f"❌ Error creating detailed diagram: {e}")
    
    try:
        # 4. Parameter breakdown chart
        param_path = create_parameter_breakdown_chart()
        created_files.append(param_path)
    except Exception as e:
        print(f"❌ Error creating parameter chart: {e}")
    
    print("=" * 60)
    print("✅ Model visualization complete!")
    print(f"📁 All files saved to: {output_dir.absolute()}")
    print("\n📋 Created visualizations:")
    for i, file_path in enumerate(created_files, 1):
        print(f"   {i}. {Path(file_path).name}")
    
    print("\n🎯 Visualization Summary:")
    print("   • Computational graph: Shows the forward pass flow")
    print("   • Architecture diagram: High-level model structure")  
    print("   • Detailed components: Component-wise breakdown")
    print("   • Parameter breakdown: Parameter distribution analysis")


if __name__ == "__main__":
    main()