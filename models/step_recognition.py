"""
Step Recognition Model: ResNet-50 + LSTM for surgical step classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class StepRecognitionModel(nn.Module):
    """
    Surgical step recognition using CNN + LSTM architecture.
    
    Architecture:
        - ResNet-50 backbone for feature extraction
        - LSTM for temporal modeling
        - Fully connected layers for classification
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        """
        Initialize step recognition model.
        
        Args:
            num_classes: Number of surgical steps
            lstm_hidden: Hidden size for LSTM
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
            pretrained: Use pretrained ResNet weights
        """
        super(StepRecognitionModel, self).__init__()
        
        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        
        # ResNet-50 backbone (remove final FC layer)
        resnet = models.resnet50(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension from ResNet-50
        self.feature_dim = 2048
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        lstm_output_dim = lstm_hidden * 2  # Bidirectional
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for FC layers."""
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, C, H, W)
            hidden: Optional hidden state for LSTM
            
        Returns:
            Tuple of (output, hidden_state)
            - output: Predictions of shape (batch, sequence_length, num_classes)
            - hidden_state: LSTM hidden state
        """
        batch_size, seq_length, C, H, W = x.size()
        
        # Reshape for CNN: (batch * seq_length, C, H, W)
        x = x.view(batch_size * seq_length, C, H, W)
        
        # Extract features with ResNet
        features = self.feature_extractor(x)  # (batch * seq_length, 2048, 1, 1)
        features = features.view(batch_size * seq_length, self.feature_dim)
        
        # Reshape for LSTM: (batch, seq_length, feature_dim)
        features = features.view(batch_size, seq_length, self.feature_dim)
        
        # LSTM forward
        if hidden is None:
            lstm_out, hidden = self.lstm(features)
        else:
            lstm_out, hidden = self.lstm(features, hidden)
        
        # Classification
        # Reshape: (batch * seq_length, lstm_hidden * 2)
        lstm_out = lstm_out.contiguous().view(batch_size * seq_length, -1)
        output = self.fc(lstm_out)
        
        # Reshape back: (batch, seq_length, num_classes)
        output = output.view(batch_size, seq_length, self.num_classes)
        
        return output, hidden
    
    def predict_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Predict step for a single frame (inference mode).
        
        Args:
            frame: Input frame tensor of shape (C, H, W) or (1, C, H, W)
            
        Returns:
            Predicted class probabilities
        """
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)  # Add batch dimension
        
        # Add sequence dimension
        frame = frame.unsqueeze(1)  # (1, 1, C, H, W)
        
        with torch.no_grad():
            output, _ = self.forward(frame)
            output = output.squeeze(0).squeeze(0)  # Remove batch and seq dims
            
        return torch.softmax(output, dim=0)
    
    def freeze_backbone(self):
        """Freeze ResNet backbone for fine-tuning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze ResNet backbone."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True


class StepRecognitionInference:
    """
    Wrapper class for inference with step recognition model.
    """
    
    def __init__(
        self,
        model: StepRecognitionModel,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        sequence_length: int = 5
    ):
        """
        Initialize inference wrapper.
        
        Args:
            model: Trained step recognition model
            device: Device to run inference on
            sequence_length: Number of frames to use for temporal context
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.sequence_length = sequence_length
        self.frame_buffer = []
        self.hidden = None
    
    def preprocess_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Preprocess frame for model input.
        
        Args:
            frame: Input frame tensor
            
        Returns:
            Preprocessed frame
        """
        # Normalize using ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        frame = frame.float() / 255.0
        frame = (frame - mean) / std
        
        return frame
    
    def predict(self, frame: torch.Tensor, use_temporal: bool = True) -> int:
        """
        Predict surgical step for a frame.
        
        Args:
            frame: Input frame tensor of shape (C, H, W)
            use_temporal: Use temporal context from previous frames
            
        Returns:
            Predicted step class (0-indexed)
        """
        # Preprocess
        frame = self.preprocess_frame(frame).to(self.device)
        
        if use_temporal:
            # Add to buffer
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) > self.sequence_length:
                self.frame_buffer.pop(0)
            
            # Create sequence
            if len(self.frame_buffer) < self.sequence_length:
                # Pad with first frame
                sequence = [self.frame_buffer[0]] * (self.sequence_length - len(self.frame_buffer))
                sequence.extend(self.frame_buffer)
            else:
                sequence = self.frame_buffer
            
            # Stack and add batch dimension
            sequence = torch.stack(sequence).unsqueeze(0)  # (1, seq_len, C, H, W)
            
            # Predict with temporal context
            with torch.no_grad():
                output, self.hidden = self.model(sequence, self.hidden)
                output = output[:, -1, :]  # Take last frame prediction
        else:
            # Single frame prediction
            output = self.model.predict_single_frame(frame)
            output = output.unsqueeze(0)
        
        # Get predicted class
        pred_class = torch.argmax(output, dim=1).item()
        
        return pred_class
    
    def reset(self):
        """Reset temporal buffer and hidden state."""
        self.frame_buffer = []
        self.hidden = None


def load_step_model(model_path: str, device: str = None) -> StepRecognitionInference:
    """
    Load trained step recognition model for inference.
    
    Args:
        model_path: Path to saved model weights
        device: Device to load model on
        
    Returns:
        Inference wrapper
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = StepRecognitionModel()
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create inference wrapper
    inference = StepRecognitionInference(model, device)
    
    return inference