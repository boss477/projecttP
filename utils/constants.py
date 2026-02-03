"""
Constants and configuration for surgical training system.
Includes step definitions, instructions, tool classes, and visual styling.
"""

# Surgical Step Definitions for Cholecystectomy
SURGICAL_STEPS = {
    0: "Preparation",
    1: "Calot Triangle Dissection",
    2: "Clipping and Cutting",
    3: "Gallbladder Dissection",
    4: "Gallbladder Packaging",
    5: "Cleaning and Coagulation",
    6: "Gallbladder Retraction"
}

# Detailed Instructions for Each Step
STEP_INSTRUCTIONS = {
    0: [
        "Position patient in reverse Trendelenburg",
        "Insert trocars: umbilical, epigastric, and lateral ports",
        "Establish pneumoperitoneum (12-15 mmHg)",
        "Insert laparoscope and inspect abdominal cavity"
    ],
    1: [
        "Grasp gallbladder fundus and retract superiorly",
        "Grasp Hartmann's pouch and retract laterally",
        "Dissect peritoneum over Calot's triangle",
        "Identify cystic duct and cystic artery",
        "Clear all tissue to achieve critical view of safety"
    ],
    2: [
        "Verify critical view of safety achieved",
        "Apply clips to cystic duct (2 proximal, 1 distal)",
        "Cut cystic duct between clips",
        "Apply clips to cystic artery (2 proximal, 1 distal)",
        "Cut cystic artery between clips"
    ],
    3: [
        "Dissect gallbladder from liver bed using hook",
        "Use cautery to control small bleeding vessels",
        "Continue dissection toward fundus",
        "Avoid injury to liver parenchyma",
        "Complete separation of gallbladder"
    ],
    4: [
        "Open specimen bag in abdominal cavity",
        "Place gallbladder into specimen bag",
        "Close specimen bag securely",
        "Position bag near extraction port"
    ],
    5: [
        "Irrigate liver bed with saline",
        "Inspect for bleeding or bile leakage",
        "Apply cautery or clips if needed",
        "Aspirate all fluid from abdominal cavity",
        "Perform final inspection of surgical field"
    ],
    6: [
        "Extract specimen bag through umbilical port",
        "Remove all instruments under vision",
        "Deflate pneumoperitoneum",
        "Remove trocars and close port sites"
    ]
}

# Tool Classes for Detection
TOOL_CLASSES = {
    0: "Grasper",
    1: "Bipolar",
    2: "Hook",
    3: "Scissors",
    4: "Clipper",
    5: "Irrigator",
    6: "SpecimenBag"
}

# Tool Usage Recommendations per Step
TOOL_RECOMMENDATIONS = {
    0: ["Grasper"],
    1: ["Grasper", "Hook", "Scissors"],
    2: ["Grasper", "Clipper", "Scissors"],
    3: ["Grasper", "Hook", "Bipolar"],
    4: ["Grasper", "SpecimenBag"],
    5: ["Irrigator", "Grasper", "Bipolar"],
    6: ["Grasper"]
}

# Visual Styling Constants
COLORS = {
    # Step overlay
    'step_bg': (200, 100, 50),  # Blue-ish (BGR format)
    'step_text': (255, 255, 255),  # White
    
    # Instruction box
    'instruction_bg': (50, 50, 50),  # Dark gray
    'instruction_text': (0, 255, 255),  # Yellow
    
    # Tool bounding boxes (different colors for different tools)
    'tool_colors': {
        'Grasper': (0, 255, 0),      # Green
        'Bipolar': (255, 0, 0),      # Blue
        'Hook': (0, 165, 255),       # Orange
        'Scissors': (255, 0, 255),   # Magenta
        'Clipper': (0, 255, 255),    # Yellow
        'Irrigator': (255, 255, 0),  # Cyan
        'SpecimenBag': (128, 0, 128) # Purple
    },
    
    # Visual guidance
    'arrow': (0, 255, 0),       # Green
    'highlight': (0, 255, 255), # Yellow
    'warning': (0, 0, 255)      # Red
}

# Font Settings
FONT_SETTINGS = {
    'font': 0,  # cv2.FONT_HERSHEY_SIMPLEX
    'step_scale': 1.2,
    'instruction_scale': 0.7,
    'tool_label_scale': 0.6,
    'thickness': 2,
    'line_type': 2  # cv2.LINE_AA
}

# Overlay Dimensions (relative to frame size)
OVERLAY_CONFIG = {
    'step_bar_height': 60,
    'instruction_box_height': 150,
    'instruction_box_width': 500,
    'padding': 20,
    'alpha_step': 0.9,
    'alpha_instruction': 0.7,
    'alpha_tool_box': 0.3
}

# Model Configuration
MODEL_CONFIG = {
    'step_recognition': {
        'input_size': (224, 224),
        'lstm_hidden': 256,
        'lstm_layers': 2,
        'num_classes': 7,
        'dropout': 0.5
    },
    'tool_detection': {
        'input_size': (640, 640),
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45,
        'num_classes': 7
    }
}

# Training Configuration
TRAINING_CONFIG = {
    'step_recognition': {
        'batch_size': 8,
        'learning_rate': 0.001,
        'epochs': 50,
        'weight_decay': 0.0001,
        'scheduler_step': 10,
        'scheduler_gamma': 0.5,
        'early_stopping_patience': 10
    },
    'tool_detection': {
        'batch_size': 16,
        'learning_rate': 0.01,
        'epochs': 100,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'momentum': 0.937
    }
}

# Video Processing
VIDEO_CONFIG = {
    'default_fps': 30,
    'codec': 'mp4v',
    'quality': 90,
    'frame_skip': 1,  # Process every N frames (1 = all frames)
    'max_frames': None  # None = process entire video
}

# Data Augmentation
AUGMENTATION_CONFIG = {
    'horizontal_flip': 0.5,
    'vertical_flip': 0.0,
    'rotation_limit': 10,
    'brightness_limit': 0.2,
    'contrast_limit': 0.2,
    'blur_limit': 3
}