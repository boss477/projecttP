"""
Configuration for step recognition model.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class StepRecognitionConfig:
    """Configuration for step recognition training and inference."""
    
    # Model architecture
    num_classes: int = 7
    lstm_hidden: int = 256
    lstm_layers: int = 2
    dropout: float = 0.5
    pretrained: bool = True
    
    # Input dimensions
    input_height: int = 224
    input_width: int = 224
    input_channels: int = 3
    
    # Temporal settings
    sequence_length: int = 5
    frame_skip: int = 1
    
    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    num_epochs: int = 50
    
    # Scheduler
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001
    
    # Data augmentation
    horizontal_flip: float = 0.5
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    blur_limit: int = 3
    
    # Dataset
    train_split: float = 0.8
    num_workers: int = 4
    
    # Surgical steps (Cholecystectomy)
    step_names: List[str] = None
    
    def __post_init__(self):
        if self.step_names is None:
            self.step_names = [
                "Preparation",
                "Calot Triangle Dissection",
                "Clipping and Cutting",
                "Gallbladder Dissection",
                "Gallbladder Packaging",
                "Cleaning and Coagulation",
                "Gallbladder Retraction"
            ]


# Default configuration
DEFAULT_STEP_CONFIG = StepRecognitionConfig()


# Configuration for fine-tuning
FINETUNE_STEP_CONFIG = StepRecognitionConfig(
    lstm_hidden=128,
    lstm_layers=1,
    learning_rate=0.0001,
    num_epochs=30,
    pretrained=True
)


# Configuration for fast training (debugging)
DEBUG_STEP_CONFIG = StepRecognitionConfig(
    batch_size=4,
    num_epochs=5,
    sequence_length=3,
    lstm_hidden=128,
    lstm_layers=1
)