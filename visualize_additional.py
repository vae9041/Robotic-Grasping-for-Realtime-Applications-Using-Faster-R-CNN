"""
Fixed Computational Graph Visualization
Creates a simplified computational graph for Faster R-CNN
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchviz import make_dot
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def create_simple_computational_graph():
    """Create a simplified computational graph visualization"""
    print("Creating simplified computational graph...")
    
    # Load model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.eval()
    
    # Create proper input format for Faster R-CNN (list of tensors)
    dummy_input = [torch.randn(3, 480, 640)]  # List of [C, H, W] tensors
    
    # Forward pass
    with torch.no_grad():
        try:
            output = model(dummy_input)
            
            # Get a tensor from the output for graph generation
            if output and len(output) > 0:
                sample_output = output[0]
                if 'boxes' in sample_output and len(sample_output['boxes']) > 0:
                    target_tensor = sample_output['boxes'].sum()
                elif 'scores' in sample_output and len(sample_output['scores']) > 0:
                    target_tensor = sample_output['scores'].sum()
                else:
                    # If no detections, create a dummy tensor connected to model
                    target_tensor = sum(p.sum() for p in model.parameters() if p.requires_grad)
                
                # Create the graph
                graph = make_dot(target_tensor, params=dict(model.named_parameters()))
                
                # Customize graph appearance
                graph.attr(rankdir='TB')  # Top to bottom
                graph.attr('node', shape='box', style='filled', fillcolor='lightblue')
                graph.attr('graph', size='12,16', dpi='300')
                
                # Save the graph
                output_dir = Path('./model_visualizations')
                output_dir.mkdir(exist_ok=True)
                
                graph_path = output_dir / 'fasterrcnn_computational_graph'
                graph.render(graph_path, format='png', cleanup=True)
                print(f"✅ Computational graph saved to: {graph_path}.png")
                
                return str(graph_path) + '.png'
                
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return None
    
    return None


def create_flow_diagram():
    """Create a data flow diagram showing tensor shapes through the network"""
    print("Creating data flow diagram...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Define the flow stages with tensor shapes
    stages = [
        {
            'name': 'Input Image',
            'shape': '[3, 480, 640]',
            'pos': (6, 15),
            'color': '#E3F2FD'
        },
        {
            'name': 'ResNet50 Conv1',
            'shape': '[64, 120, 160]',
            'pos': (6, 13.5),
            'color': '#BBDEFB'
        },
        {
            'name': 'ResNet50 Layer1',
            'shape': '[256, 120, 160]',
            'pos': (6, 12),
            'color': '#90CAF9'
        },
        {
            'name': 'ResNet50 Layer2',
            'shape': '[512, 60, 80]',
            'pos': (6, 10.5),
            'color': '#64B5F6'
        },
        {
            'name': 'ResNet50 Layer3',
            'shape': '[1024, 30, 40]',
            'pos': (6, 9),
            'color': '#42A5F5'
        },
        {
            'name': 'ResNet50 Layer4',
            'shape': '[2048, 15, 20]',
            'pos': (6, 7.5),
            'color': '#2196F3'
        },
        {
            'name': 'FPN P2',
            'shape': '[256, 120, 160]',
            'pos': (1.5, 6),
            'color': '#C8E6C9'
        },
        {
            'name': 'FPN P3',
            'shape': '[256, 60, 80]',
            'pos': (3.5, 6),
            'color': '#A5D6A7'
        },
        {
            'name': 'FPN P4',
            'shape': '[256, 30, 40]',
            'pos': (5.5, 6),
            'color': '#81C784'
        },
        {
            'name': 'FPN P5',
            'shape': '[256, 15, 20]',
            'pos': (7.5, 6),
            'color': '#66BB6A'
        },
        {
            'name': 'FPN P6',
            'shape': '[256, 8, 10]',
            'pos': (9.5, 6),
            'color': '#4CAF50'
        },
        {
            'name': 'RPN Anchors',
            'shape': '[~260K anchors]',
            'pos': (2.5, 4),
            'color': '#FFE0B2'
        },
        {
            'name': 'RPN Proposals',
            'shape': '[2000 boxes]',
            'pos': (2.5, 2.5),
            'color': '#FFCC80'
        },
        {
            'name': 'ROI Features',
            'shape': '[N, 256, 7, 7]',
            'pos': (8.5, 4),
            'color': '#F8BBD9'
        },
        {
            'name': 'FC Features',
            'shape': '[N, 1024]',
            'pos': (8.5, 2.5),
            'color': '#F48FB1'
        },
        {
            'name': 'Classifications',
            'shape': '[N, 2]',
            'pos': (6, 1),
            'color': '#FFCDD2'
        },
        {
            'name': 'Box Regressions',
            'shape': '[N, 8]',
            'pos': (10, 1),
            'color': '#FFCDD2'
        }
    ]
    
    # Draw stages
    for stage in stages:
        x, y = stage['pos']
        
        # Create box
        box = patches.FancyBboxPatch(
            (x-1, y-0.4), 2, 0.8,
            boxstyle="round,pad=0.1",
            facecolor=stage['color'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(x, y+0.1, stage['name'], ha='center', va='center', 
               fontsize=9, weight='bold')
        ax.text(x, y-0.2, stage['shape'], ha='center', va='center', 
               fontsize=8, style='italic')
    
    # Add arrows to show data flow
    arrows = [
        # Backbone flow
        ((6, 14.6), (6, 14.1)),  # Input -> Conv1
        ((6, 13.1), (6, 12.6)),  # Conv1 -> Layer1
        ((6, 11.6), (6, 11.1)),  # Layer1 -> Layer2
        ((6, 10.1), (6, 9.6)),   # Layer2 -> Layer3
        ((6, 8.6), (6, 8.1)),    # Layer3 -> Layer4
        
        # Layer outputs to FPN
        ((5.5, 12), (2.5, 6.4)),   # Layer1 -> P2
        ((5.5, 10.5), (4.5, 6.4)), # Layer2 -> P3
        ((5.5, 9), (6.5, 6.4)),    # Layer3 -> P4
        ((6.5, 7.5), (8.5, 6.4)),  # Layer4 -> P5
        ((8.5, 6), (10.5, 6.4)),   # P5 -> P6
        
        # FPN to RPN
        ((4.5, 5.6), (3.5, 4.4)),  # FPN -> RPN anchors
        ((2.5, 3.6), (2.5, 2.9)),  # Anchors -> Proposals
        
        # FPN to ROI
        ((7.5, 5.6), (8.5, 4.4)),  # FPN -> ROI features
        ((8.5, 3.6), (8.5, 2.9)),  # ROI features -> FC
        
        # Final outputs
        ((8, 2.1), (6.5, 1.4)),    # FC -> Classifications
        ((9, 2.1), (9.5, 1.4)),    # FC -> Box regressions
    ]
    
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='#333333'))
    
    # Title
    ax.text(7, 15.8, 'Faster R-CNN Data Flow & Tensor Shapes', 
           fontsize=16, weight='bold', ha='center')
    
    # Add legend
    legend_elements = [
        ('Backbone', '#2196F3'),
        ('FPN', '#4CAF50'),
        ('RPN', '#FF9800'),
        ('ROI Head', '#E91E63'),
        ('Output', '#FFCDD2')
    ]
    
    for i, (label, color) in enumerate(legend_elements):
        ax.add_patch(patches.Rectangle((11, 14-i*0.5), 0.5, 0.3, 
                                     facecolor=color, edgecolor='black'))
        ax.text(11.7, 14.15-i*0.5, label, va='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save the diagram
    output_dir = Path('./model_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    flow_path = output_dir / 'fasterrcnn_data_flow.png'
    plt.savefig(flow_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Data flow diagram saved to: {flow_path}")
    return str(flow_path)


def main():
    """Create additional visualizations"""
    print("🎨 Creating additional Faster R-CNN visualizations...")
    print("=" * 50)
    
    created_files = []
    
    try:
        # Try to create computational graph
        graph_path = create_simple_computational_graph()
        if graph_path:
            created_files.append(graph_path)
    except Exception as e:
        print(f"❌ Error creating computational graph: {e}")
    
    try:
        # Create data flow diagram
        flow_path = create_flow_diagram()
        created_files.append(flow_path)
    except Exception as e:
        print(f"❌ Error creating flow diagram: {e}")
    
    print("=" * 50)
    print("✅ Additional visualizations complete!")
    if created_files:
        print("\n📋 Created files:")
        for i, file_path in enumerate(created_files, 1):
            print(f"   {i}. {Path(file_path).name}")


if __name__ == "__main__":
    main()