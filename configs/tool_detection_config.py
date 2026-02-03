"""
Configuration for tool detection model.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ToolDetectionConfig:
    """Configuration for tool detection training and inference."""
    
    # Model architecture
    model_type: str = 'yolov5s'  # yolov5s, yolov5m, yolov5l, yolov5x
    num_classes: int = 7
    
    # Input dimensions
    img_size: int = 640
    
    # Detection thresholds
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    
    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    num_epochs: int = 100
    
    # Warmup
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # Data augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    
    # Dataset
    train_split: float = 0.8
    cache_images: bool = False
    
    # Tool classes
    class_names: List[str] = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = [
                'Grasper',
                'Bipolar',
                'Hook',
                'Scissors',
                'Clipper',
                'Irrigator',
                'SpecimenBag'
            ]


# Default configuration
DEFAULT_TOOL_CONFIG = ToolDetectionConfig()


# Configuration for high precision
HIGH_PRECISION_TOOL_CONFIG = ToolDetectionConfig(
    model_type='yolov5l',
    img_size=1280,
    batch_size=8,
    num_epochs=150,
    confidence_threshold=0.6,
    iou_threshold=0.5
)


# Configuration for fast inference
FAST_TOOL_CONFIG = ToolDetectionConfig(
    model_type='yolov5s',
    img_size=416,
    batch_size=32,
    confidence_threshold=0.4,
    iou_threshold=0.4
)


# Configuration for debugging
DEBUG_TOOL_CONFIG = ToolDetectionConfig(
    batch_size=8,
    num_epochs=10,
    img_size=416
)