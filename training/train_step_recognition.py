"""
Training script for surgical step recognition model.

Usage:
    python training/train_step_recognition.py --data_dir data/cholec80 --epochs 50
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.step_recognition import StepRecognitionModel
from data.preprocessing import create_data_loaders
from utils.constants import TRAINING_CONFIG


class StepRecognitionTrainer:
    """
    Trainer class for step recognition model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str,
        output_dir: str,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ):
        """
        Initialize trainer.
        
        Args:
            model: Step recognition model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            output_dir: Directory to save outputs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.5
        )
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience = 10
        self.patience_counter = 0
    
    def train_epoch(self, epoch: int) -> dict:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (frames, labels) in enumerate(pbar):
            # Move to device
            frames = frames.to(self.device)  # (batch, seq_len, C, H, W)
            labels = labels.to(self.device)  # (batch, seq_len)
            
            # Forward pass
            outputs, _ = self.model(frames)  # (batch, seq_len, num_classes)
            
            # Reshape for loss computation
            batch_size, seq_len, num_classes = outputs.shape
            outputs = outputs.view(batch_size * seq_len, num_classes)
            labels = labels.view(batch_size * seq_len)
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(self, epoch: int) -> dict:
        """
        Validate model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class accuracy
        class_correct = [0] * 7
        class_total = [0] * 7
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            
            for frames, labels in pbar:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(frames)
                
                # Reshape
                batch_size, seq_len, num_classes = outputs.shape
                outputs = outputs.view(batch_size * seq_len, num_classes)
                labels = labels.view(batch_size * seq_len)
                
                # Loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Per-class accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Per-class accuracy
        class_accuracies = {}
        for i in range(7):
            if class_total[i] > 0:
                class_accuracies[f'class_{i}'] = 100. * class_correct[i] / class_total[i]
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'class_accuracies': class_accuracies
        }
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_acc': self.best_val_acc
        }
        
        # Save latest
        checkpoint_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if is_best:
            best_path = self.output_dir / 'step_model.pth'
            torch.save(checkpoint, best_path)
            print(f'Saved best model with accuracy: {metrics["accuracy"]:.2f}%')
    
    def train(self, num_epochs: int):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Learning rate step
            self.scheduler.step()
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check if best model
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
                break
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train step recognition model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to Cholec80 dataset')
    parser.add_argument('--output_dir', type=str, default='models/pretrained',
                       help='Directory to save model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to train on (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nLoading dataset...")
    train_loader, val_loader = create_data_loaders(
        args.data_dir,
        dataset_type='cholec80',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = StepRecognitionModel(
        num_classes=7,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.5,
        pretrained=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = StepRecognitionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate
    )
    
    # Train
    trainer.train(args.epochs)


if __name__ == '__main__':
    main()