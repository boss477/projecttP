"""
Training utilities and helper functions.
"""

import torch
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from pathlib import Path


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def save_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    output_path: str
):
    """
    Plot and save training curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accs: Training accuracies per epoch
        val_accs: Validation accuracies per epoch
        output_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted class labels
        targets: True class labels
        num_classes: Number of classes
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for pred, target in zip(predictions, targets):
        conf_matrix[target, pred] += 1
    
    return conf_matrix


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: List[str],
    output_path: str,
    normalize: bool = False
):
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix
        class_names: List of class names
        output_path: Path to save plot
        normalize: Normalize by row (true class)
    """
    if normalize:
        conf_matrix = conf_matrix.astype('float')
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix = np.divide(conf_matrix, row_sums, 
                               where=row_sums != 0,
                               out=np.zeros_like(conf_matrix))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(conf_matrix.shape[1]),
           yticks=np.arange(conf_matrix.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], fmt),
                   ha="center", va="center",
                   color="white" if conf_matrix[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted class labels
        targets: True class labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with metrics
    """
    # Overall accuracy
    accuracy = (predictions == targets).mean() * 100
    
    # Per-class metrics
    class_metrics = {}
    
    for i in range(num_classes):
        mask = targets == i
        if mask.sum() > 0:
            class_acc = (predictions[mask] == i).mean() * 100
            class_metrics[f'class_{i}_acc'] = class_acc
    
    # Macro average
    macro_acc = np.mean([v for v in class_metrics.values()])
    
    return {
        'accuracy': accuracy,
        'macro_accuracy': macro_acc,
        **class_metrics
    }


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:  # max
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Make CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(
    state: dict,
    filepath: str,
    is_best: bool = False
):
    """
    Save training checkpoint.
    
    Args:
        state: State dictionary to save
        filepath: Path to save checkpoint
        is_best: Whether this is the best model
    """
    torch.save(state, filepath)
    
    if is_best:
        best_path = Path(filepath).parent / 'model_best.pth'
        torch.save(state, best_path)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer=None,
    device: str = 'cpu'
) -> dict:
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        device: Device to load checkpoint on
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint