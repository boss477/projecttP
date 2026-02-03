"""
Tool Detection Model: YOLOv5-based detector for surgical instruments.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
import cv2


class ToolDetectionModel:
    """
    Surgical tool detection using YOLOv5 architecture.
    
    Uses the ultralytics YOLOv5 implementation for object detection.
    Detects 7 surgical instrument classes.
    """
    
    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = None,
        img_size: int = 640
    ):
        """
        Initialize tool detection model.
        
        Args:
            model_path: Path to trained weights (None = use pretrained)
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run on ('cuda', 'cpu', or None for auto)
            img_size: Input image size
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        
        # Tool class names
        self.class_names = [
            'Grasper',
            'Bipolar',
            'Hook',
            'Scissors',
            'Clipper',
            'Irrigator',
            'SpecimenBag'
        ]
        
        # Try to load YOLOv5
        try:
            # Import YOLOv5 from ultralytics
            from ultralytics import YOLO
            
            if model_path and os.path.exists(model_path):
                # Load custom trained model
                self.model = YOLO(model_path)
            else:
                # Create new model with custom classes
                self.model = YOLO('yolov5s.pt')  # Start with pretrained weights
                
        except ImportError:
            print("Warning: ultralytics not installed. Using placeholder model.")
            self.model = None
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        project: str = 'runs/train',
        name: str = 'tool_detection'
    ):
        """
        Train the model on custom dataset.
        
        Args:
            data_yaml: Path to data configuration YAML
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Image size
            project: Project directory
            name: Experiment name
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=project,
            name=name,
            device=self.device
        )
        
        return results
    
    def predict(
        self,
        image: np.ndarray,
        return_boxes: bool = True
    ) -> List[Dict]:
        """
        Detect tools in image.
        
        Args:
            image: Input image (BGR format)
            return_boxes: Return bounding box coordinates
            
        Returns:
            List of detections, each containing:
                - class_id: Tool class ID
                - class_name: Tool class name
                - confidence: Detection confidence
                - bbox: [x1, y1, x2, y2] if return_boxes=True
        """
        if self.model is None:
            # Return empty detections if model not loaded
            return []
        
        # Run inference
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False
        )
        
        # Parse results
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get box coordinates
                box = boxes.xyxy[i].cpu().numpy()
                
                # Get class and confidence
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                
                detection = {
                    'class_id': class_id,
                    'class_name': self.class_names[class_id] if class_id < len(self.class_names) else 'Unknown',
                    'confidence': confidence
                }
                
                if return_boxes:
                    detection['bbox'] = box.tolist()
                
                detections.append(detection)
        
        return detections
    
    def save(self, save_path: str):
        """
        Save model weights.
        
        Args:
            save_path: Path to save weights
        """
        if self.model is not None:
            self.model.save(save_path)


class SimplifiedToolDetector(nn.Module):
    """
    Simplified tool detector for demonstration purposes.
    Uses a basic CNN architecture when full YOLO is not available.
    """
    
    def __init__(self, num_classes: int = 7):
        """
        Initialize simplified detector.
        
        Args:
            num_classes: Number of tool classes
        """
        super(SimplifiedToolDetector, self).__init__()
        
        self.num_classes = num_classes
        
        # Simple CNN backbone
        self.backbone = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Detection head (simplified)
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        output = self.detection_head(features)
        return output


def create_yolo_data_yaml(
    train_path: str,
    val_path: str,
    output_path: str,
    class_names: List[str] = None
):
    """
    Create YAML configuration for YOLO training.
    
    Args:
        train_path: Path to training images
        val_path: Path to validation images
        output_path: Path to save YAML file
        class_names: List of class names
    """
    if class_names is None:
        class_names = [
            'Grasper', 'Bipolar', 'Hook', 'Scissors',
            'Clipper', 'Irrigator', 'SpecimenBag'
        ]
    
    yaml_content = f"""# Tool Detection Dataset Configuration

# Paths
train: {train_path}
val: {val_path}

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}  # class names
"""
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)


def load_tool_model(
    model_path: str,
    confidence_threshold: float = 0.5,
    device: str = None
) -> ToolDetectionModel:
    """
    Load trained tool detection model.
    
    Args:
        model_path: Path to model weights
        confidence_threshold: Detection confidence threshold
        device: Device to run on
        
    Returns:
        Loaded model
    """
    model = ToolDetectionModel(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device
    )
    
    return model


def postprocess_detections(
    detections: List[Dict],
    image_shape: Tuple[int, int],
    min_confidence: float = 0.3,
    nms_iou: float = 0.5
) -> List[Dict]:
    """
    Post-process detections with additional filtering.
    
    Args:
        detections: Raw detections from model
        image_shape: (height, width) of image
        min_confidence: Minimum confidence threshold
        nms_iou: IoU threshold for additional NMS
        
    Returns:
        Filtered detections
    """
    if not detections:
        return []
    
    # Filter by confidence
    filtered = [d for d in detections if d['confidence'] >= min_confidence]
    
    # Additional NMS by class
    final_detections = []
    
    for class_name in set(d['class_name'] for d in filtered):
        class_dets = [d for d in filtered if d['class_name'] == class_name]
        
        if not class_dets:
            continue
        
        # Sort by confidence
        class_dets.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Simple NMS
        keep = []
        while class_dets:
            best = class_dets.pop(0)
            keep.append(best)
            
            # Remove overlapping boxes
            class_dets = [
                d for d in class_dets
                if compute_iou(best['bbox'], d['bbox']) < nms_iou
            ]
        
        final_detections.extend(keep)
    
    return final_detections


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


# Add os import for file existence check
import os