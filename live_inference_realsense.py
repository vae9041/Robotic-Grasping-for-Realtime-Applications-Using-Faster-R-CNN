"""
Real-time Grasp Detection using Intel RealSense Camera
Live inference script for trained Faster R-CNN grasp detection model
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# RealSense imports
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not installed. Install with: pip install pyrealsense2")

# Import model components
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.ops import nms

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraspRectangle:
    """Represents a grasp rectangle for visualization and processing."""
    
    def __init__(self, center_x: float, center_y: float, width: float, height: float, 
                 angle: float = 0.0, confidence: float = 1.0):
        """Initialize a grasp rectangle."""
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.angle = angle  # In radians
        self.confidence = confidence
    
    @classmethod
    def from_bbox(cls, x_min: float, y_min: float, x_max: float, y_max: float, confidence: float = 1.0):
        """Create GraspRectangle from bounding box coordinates."""
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        return cls(center_x, center_y, width, height, angle=0.0, confidence=confidence)
    
    def to_corners(self) -> List[Tuple[float, float]]:
        """Convert to 4 corner points for visualization."""
        import math
        
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

class RealSenseCamera:
    """Wrapper for Intel RealSense camera operations."""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """Initialize RealSense camera."""
        if not REALSENSE_AVAILABLE:
            raise ImportError("pyrealsense2 not available. Install with: pip install pyrealsense2")
        
        self.width = width
        self.height = height
        self.fps = fps
        
        # Configure streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # Start pipeline
        self.pipeline.start(self.config)
        
        # Allow camera to warm up
        logger.info("Warming up camera...")
        for i in range(10):
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                if frames.get_color_frame():
                    logger.info(f"Camera warm-up successful after {i+1} attempts")
                    break
            except Exception as e:
                logger.warning(f"Warm-up attempt {i+1} failed: {e}")
                if i == 9:
                    raise RuntimeError("Camera failed to initialize after 10 attempts")
        
        # Get camera intrinsics
        profile = self.pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        logger.info(f"RealSense camera initialized: {width}x{height} @ {fps}fps")
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get color and depth frames."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)  # Increased timeout
            
            # Get color frame
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data()) if color_frame else None
            
            # Get depth frame
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
            
            return color_image, depth_image
            
        except RuntimeError as e:
            logger.warning(f"Failed to get frames: {e}")
            return None, None
    
    def stop(self):
        """Stop the camera pipeline."""
        self.pipeline.stop()
        logger.info("RealSense camera stopped")

class LiveGraspDetector:
    """Real-time grasp detection using trained Faster R-CNN model."""
    
    def __init__(self, model_path: str, device: str = 'cuda', confidence_threshold: float = 0.5):
        """Initialize the live grasp detector."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Performance tracking
        self.inference_times = []
        self.frame_count = 0
        
        logger.info(f"Live grasp detector initialized on {self.device}")
    
    def _load_model(self, model_path: str):
        """Load the trained model."""
        # Create model architecture
        model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model inference."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        image_tensor = F.to_tensor(image_rgb)
        
        return image_tensor.unsqueeze(0).to(self.device)
    
    def predict(self, image: np.ndarray) -> List[GraspRectangle]:
        """Predict grasps for a single image."""
        start_time = time.time()
        
        # Preprocess
        image_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Post-process predictions
        grasps = self._post_process_predictions(predictions[0])
        
        # Track performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        return grasps
    
    def _post_process_predictions(self, prediction: Dict) -> List[GraspRectangle]:
        """Post-process model predictions to grasp rectangles."""
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        
        # Filter by confidence threshold
        valid_indices = scores >= self.confidence_threshold
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        
        # Apply NMS
        if len(boxes) > 0:
            keep_indices = nms(
                torch.tensor(boxes), 
                torch.tensor(scores), 
                iou_threshold=0.3
            ).numpy()
            
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
        
        # Convert to grasp rectangles
        grasps = []
        for box, score in zip(boxes, scores):
            grasp = GraspRectangle.from_bbox(
                box[0], box[1], box[2], box[3], confidence=score
            )
            grasps.append(grasp)
        
        return grasps
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.inference_times:
            return {}
        
        avg_inference_time = np.mean(self.inference_times[-100:])  # Last 100 frames
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            'avg_inference_time': avg_inference_time,
            'fps': fps,
            'frame_count': self.frame_count
        }

class GraspVisualizer:
    """Visualize grasp predictions on images."""
    
    def __init__(self):
        """Initialize visualizer."""
        self.colors = [
            (0, 255, 0),    # Green for high confidence
            (255, 255, 0),  # Yellow for medium confidence  
            (255, 0, 0),    # Red for low confidence
        ]
    
    def draw_grasps(self, image: np.ndarray, grasps: List[GraspRectangle], 
                   depth_image: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw grasp predictions on image."""
        result_image = image.copy()
        
        for i, grasp in enumerate(grasps):
            # Choose color based on confidence
            if grasp.confidence > 0.8:
                color = self.colors[0]  # Green
            elif grasp.confidence > 0.6:
                color = self.colors[1]  # Yellow
            else:
                color = self.colors[2]  # Red
            
            # Draw grasp rectangle
            self._draw_grasp_rectangle(result_image, grasp, color)
            
            # Add confidence text
            conf_text = f"{grasp.confidence:.2f}"
            cv2.putText(result_image, conf_text, 
                       (int(grasp.center_x), int(grasp.center_y - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result_image
    
    def _draw_grasp_rectangle(self, image: np.ndarray, grasp: GraspRectangle, color: Tuple[int, int, int]):
        """Draw a single grasp rectangle."""
        corners = grasp.to_corners()
        
        # Convert to integer points
        points = np.array([(int(x), int(y)) for x, y in corners], dtype=np.int32)
        
        # Draw filled rectangle with transparency
        overlay = image.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Draw rectangle outline
        cv2.polylines(image, [points], True, color, 2)
        
        # Draw center point
        cv2.circle(image, (int(grasp.center_x), int(grasp.center_y)), 3, color, -1)
    
    def add_info_overlay(self, image: np.ndarray, grasps: List[GraspRectangle], 
                        stats: Dict[str, float]) -> np.ndarray:
        """Add information overlay to image."""
        result_image = image.copy()
        
        # Add performance stats
        y_offset = 30
        if 'fps' in stats:
            fps_text = f"FPS: {stats['fps']:.1f}"
            cv2.putText(result_image, fps_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        if 'avg_inference_time' in stats:
            time_text = f"Inference: {stats['avg_inference_time']*1000:.1f}ms"
            cv2.putText(result_image, time_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        # Add grasp count
        grasp_text = f"Grasps: {len(grasps)}"
        cv2.putText(result_image, grasp_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_image

def main():
    """Main function for live inference."""
    parser = argparse.ArgumentParser(description="Live Grasp Detection with Intel RealSense")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    parser.add_argument('--camera_width', type=int, default=640,
                       help='Camera width resolution')
    parser.add_argument('--camera_height', type=int, default=480,
                       help='Camera height resolution')
    parser.add_argument('--camera_fps', type=int, default=30,
                       help='Camera FPS')
    parser.add_argument('--save_images', action='store_true',
                       help='Save annotated images')
    parser.add_argument('--output_dir', type=str, default='./live_inference_output',
                       help='Output directory for saved images')
    
    args = parser.parse_args()
    
    # Check if RealSense is available
    if not REALSENSE_AVAILABLE:
        logger.error("RealSense library not available. Install with: pip install pyrealsense2")
        return
    
    # Create output directory if saving images
    if args.save_images:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Saving images to: {output_dir}")
    
    try:
        # Initialize components
        camera = RealSenseCamera(args.camera_width, args.camera_height, args.camera_fps)
        detector = LiveGraspDetector(args.model_path, confidence_threshold=args.confidence_threshold)
        visualizer = GraspVisualizer()
        
        logger.info("Starting live inference... Press 'q' to quit, 's' to save current frame")
        
        frame_id = 0
        while True:
            # Get camera frames
            color_image, depth_image = camera.get_frames()
            
            if color_image is None:
                logger.warning("No color frame received")
                continue
            
            # Predict grasps
            grasps = detector.predict(color_image)
            
            # Visualize results
            result_image = visualizer.draw_grasps(color_image, grasps, depth_image)
            
            # Add info overlay
            stats = detector.get_performance_stats()
            result_image = visualizer.add_info_overlay(result_image, grasps, stats)
            
            # Display
            cv2.imshow('Live Grasp Detection', result_image)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and args.save_images:
                # Save current frame
                save_path = output_dir / f"grasp_detection_{frame_id:06d}.jpg"
                cv2.imwrite(str(save_path), result_image)
                logger.info(f"Saved frame to: {save_path}")
            
            frame_id += 1
        
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()
        
        # Print final stats
        final_stats = detector.get_performance_stats()
        logger.info(f"Session complete. Final stats: {final_stats}")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during live inference: {e}")
        raise
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()