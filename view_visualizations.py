"""
Model Visualization Summary
Displays information about all created Faster R-CNN visualizations
"""

from pathlib import Path
import os


def display_visualization_summary():
    """Display summary of all created visualizations"""
    
    print("🎨 FASTER R-CNN MODEL VISUALIZATIONS SUMMARY")
    print("=" * 70)
    
    output_dir = Path('./model_visualizations')
    
    visualizations = {
        'fasterrcnn_architecture_diagram.png': {
            'title': '🏗️ Architecture Diagram',
            'description': 'High-level overview of Faster R-CNN architecture showing main components, data flow, and parameter counts. Perfect for presentations and understanding the overall structure.'
        },
        'fasterrcnn_computational_graph.png': {
            'title': '📊 Computational Graph',
            'description': 'Detailed computational graph generated with torchviz showing the forward pass, tensor operations, and gradient flow through the network.'
        },
        'fasterrcnn_data_flow.png': {
            'title': '🔄 Data Flow Diagram',
            'description': 'Tensor shapes and transformations through each layer of the network, from input image to final predictions.'
        },
        'fasterrcnn_detailed_components.png': {
            'title': '🔧 Detailed Components',
            'description': 'Component-wise breakdown showing internal structure of ResNet50, FPN, RPN, and ROI Head with layer specifications.'
        },
        'fasterrcnn_parameter_breakdown.png': {
            'title': '📈 Parameter Breakdown',
            'description': 'Visual analysis of parameter distribution across components using pie chart and bar chart representations.'
        }
    }
    
    # Check which files exist and display info
    created_count = 0
    for filename, info in visualizations.items():
        filepath = output_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"\n{info['title']}")
            print(f"📁 File: {filename}")
            print(f"📏 Size: {size_mb:.2f} MB")
            print(f"📝 Description: {info['description']}")
            print(f"🔗 Path: {filepath.absolute()}")
            created_count += 1
        else:
            print(f"\n❌ {info['title']} - File not found: {filename}")
    
    print("\n" + "=" * 70)
    print(f"✅ Successfully created {created_count}/{len(visualizations)} visualizations")
    print(f"📁 All files saved to: {output_dir.absolute()}")
    
    # Model specifications
    print("\n🤖 MODEL SPECIFICATIONS:")
    print("   • Architecture: Faster R-CNN with ResNet50 FPN backbone")
    print("   • Total Parameters: 41,299,161 (41.3M)")
    print("   • Trainable Parameters: 41,076,761 (99.46%)")
    print("   • Model Size: ~157 MB")
    print("   • Input Size: Variable (typical: 640×480)")
    print("   • Output Classes: 2 (background + grasp)")
    
    # Performance metrics
    print("\n📊 PERFORMANCE METRICS:")
    print("   • Cornell Dataset: 97.06% success rate")
    print("   • Jacquard Dataset: 86.68% success rate") 
    print("   • Frozen Backbone Cornell: 87.57% success rate")
    print("   • Frozen Backbone Jacquard: 86.75% success rate")
    
    # Usage instructions
    print("\n🎯 USAGE INSTRUCTIONS:")
    print("   1. Open any PNG file with your preferred image viewer")
    print("   2. Use for presentations, documentation, or analysis")
    print("   3. High-resolution files suitable for printing/publishing")
    print("   4. Combine multiple diagrams for comprehensive understanding")
    
    print("\n💡 TIPS:")
    print("   • Architecture diagram: Best for high-level overview")
    print("   • Computational graph: Detailed for debugging/analysis") 
    print("   • Data flow: Understanding tensor transformations")
    print("   • Components: Deep dive into each module")
    print("   • Parameter breakdown: Resource allocation analysis")


if __name__ == "__main__":
    display_visualization_summary()