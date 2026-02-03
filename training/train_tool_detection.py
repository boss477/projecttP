"""
Training script for surgical tool detection model.

Usage:
    python training/train_tool_detection.py --data_dir data/endovis --epochs 100
"""

import os
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.tool_detection import ToolDetectionModel, create_yolo_data_yaml


def prepare_yolo_dataset(data_dir: str, output_dir: str, train_split: float = 0.8):
    """
    Prepare EndoVis dataset for YOLO training.
    
    Args:
        data_dir: Root directory of EndoVis dataset
        output_dir: Output directory for YOLO dataset
        train_split: Fraction of data for training
    """
    from pathlib import Path
    import shutil
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Create directories
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(list((data_path / 'images').glob('*.png')))
    
    # Split
    split_idx = int(len(image_files) * train_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Preparing YOLO dataset...")
    print(f"Train images: {len(train_files)}")
    print(f"Val images: {len(val_files)}")
    
    # Copy files
    for img_file in train_files:
        # Copy image
        shutil.copy(img_file, output_path / 'images' / 'train' / img_file.name)
        
        # Copy annotation
        ann_file = data_path / 'annotations' / (img_file.stem + '.txt')
        if ann_file.exists():
            shutil.copy(ann_file, output_path / 'labels' / 'train' / ann_file.name)
    
    for img_file in val_files:
        # Copy image
        shutil.copy(img_file, output_path / 'images' / 'val' / img_file.name)
        
        # Copy annotation
        ann_file = data_path / 'annotations' / (img_file.stem + '.txt')
        if ann_file.exists():
            shutil.copy(ann_file, output_path / 'labels' / 'val' / ann_file.name)
    
    # Create data.yaml
    yaml_path = output_path / 'data.yaml'
    create_yolo_data_yaml(
        train_path=str((output_path / 'images' / 'train').resolve()),
        val_path=str((output_path / 'images' / 'val').resolve()),
        output_path=str(yaml_path)
    )
    
    print(f"Dataset prepared at: {output_path}")
    print(f"YAML config created at: {yaml_path}")
    
    return str(yaml_path)


def train_tool_detection(
    data_yaml: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = None
):
    """
    Train tool detection model.
    
    Args:
        data_yaml: Path to YOLO dataset configuration
        output_dir: Directory to save model
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        device: Device to train on
    """
    print("\nInitializing tool detection model...")
    
    # Create model
    model = ToolDetectionModel(
        confidence_threshold=0.5,
        device=device,
        img_size=img_size
    )
    
    if model.model is None:
        print("\nERROR: YOLOv5 model could not be initialized.")
        print("Please install ultralytics: pip install ultralytics")
        return
    
    print(f"Training on device: {model.device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print("=" * 60)
    
    # Train
    try:
        results = model.train(
            data_yaml=data_yaml,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            project=output_dir,
            name='tool_detection'
        )
        
        print("\nTraining completed successfully!")
        
        # Save final model
        final_model_path = Path(output_dir) / 'tool_model.pth'
        model.save(str(final_model_path))
        print(f"Final model saved to: {final_model_path}")
        
        return results
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Make sure ultralytics is properly installed and CUDA is available.")
        raise


def main():
    parser = argparse.ArgumentParser(description='Train tool detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to EndoVis dataset')
    parser.add_argument('--output_dir', type=str, default='models/pretrained',
                       help='Directory to save model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--img_size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to train on (cuda/cpu)')
    parser.add_argument('--prepare_only', action='store_true',
                       help='Only prepare dataset without training')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Prepare dataset
    print("\nPreparing YOLO dataset...")
    yolo_data_dir = Path(args.data_dir).parent / 'endovis_yolo'
    data_yaml = prepare_yolo_dataset(
        args.data_dir,
        str(yolo_data_dir),
        train_split=0.8
    )
    
    if args.prepare_only:
        print("\nDataset preparation complete. Exiting.")
        return
    
    # Train model
    print("\nStarting training...")
    train_tool_detection(
        data_yaml=data_yaml,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=device
    )


if __name__ == '__main__':
    main()