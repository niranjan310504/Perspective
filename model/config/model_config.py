"""
Model Configuration for Perspective
=====================================

Configuration dataclass for BERT multi-label classification model.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for the bias detection model."""
    
    # Model architecture
    model_name: str = "bert-base-uncased"
    num_labels: int = 7
    hidden_dropout_prob: float = 0.1
    classifier_dropout: float = 0.3
    
    # Label information
    label_names: tuple = (
        "political",
        "gender",
        "entity",
        "racial",
        "religious",
        "regional",
        "sensationalism"
    )
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 4
    batch_size: int = 16
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Tokenization
    max_length: int = 512
    
    # Thresholds for prediction
    default_threshold: float = 0.5
    label_thresholds: Optional[dict] = None  # Per-label thresholds
    
    # Paths
    checkpoint_dir: str = "model/checkpoints"
    logs_dir: str = "model/logs"
    
    def get_threshold(self, label: str) -> float:
        """Get classification threshold for a specific label."""
        if self.label_thresholds and label in self.label_thresholds:
            return self.label_thresholds[label]
        return self.default_threshold
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "classifier_dropout": self.classifier_dropout,
            "label_names": list(self.label_names),
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "max_length": self.max_length,
            "default_threshold": self.default_threshold,
            "label_thresholds": self.label_thresholds
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


# Default configuration
DEFAULT_CONFIG = ModelConfig()
