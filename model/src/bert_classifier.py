"""
BERT Multi-Label Classification Model for Perspective
=======================================================

This module implements a BERT-based classifier for detecting
7 types of media bias in Indian news articles.

Architecture:
- Base: bert-base-uncased (110M parameters)
- Head: Linear classifier with dropout
- Output: 7 sigmoid activations (one per bias type)
- Loss: Binary Cross-Entropy with Logits
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
from typing import Optional, Tuple


class BertForMultiLabelBiasClassification(BertPreTrainedModel):
    """
    BERT model for multi-label bias classification.
    
    This model adds a classification head on top of BERT for
    detecting multiple bias types simultaneously.
    
    The key differences from standard classification:
    1. Uses BCEWithLogitsLoss instead of CrossEntropyLoss
    2. Each output is independent (not mutually exclusive)
    3. Sigmoid activation instead of softmax
    """
    
    def __init__(
        self,
        config: BertConfig,
        num_labels: int = 7,
        classifier_dropout: float = 0.3
    ):
        """
        Initialize the model.
        
        Args:
            config: BERT configuration
            num_labels: Number of bias types to detect
            classifier_dropout: Dropout probability for classifier
        """
        super().__init__(config)
        self.num_labels = num_labels
        
        # BERT encoder
        self.bert = BertModel(config)
        
        # Classification head
        # We use the [CLS] token representation
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Loss function: BCEWithLogitsLoss
        # Combines sigmoid + BCE for numerical stability
        # pos_weight can be used for class imbalance
        self.loss_fct = nn.BCEWithLogitsLoss()
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> dict:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            token_type_ids: Segment IDs [batch_size, seq_length]
            labels: Ground truth labels [batch_size, num_labels]
            return_dict: Whether to return a dict
            
        Returns:
            Dictionary with loss, logits, and probabilities
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Extract [CLS] token representation
        # pooler_output has shape [batch_size, hidden_size]
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Ensure labels are float for BCE loss
            labels = labels.float()
            loss = self.loss_fct(logits, labels)
        
        # Convert logits to probabilities using sigmoid
        probabilities = torch.sigmoid(logits)
        
        return {
            "loss": loss,
            "logits": logits,
            "probabilities": probabilities,
            "hidden_states": outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with thresholding.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            threshold: Classification threshold
            
        Returns:
            Tuple of (binary_predictions, probabilities)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            probabilities = outputs["probabilities"]
            predictions = (probabilities >= threshold).int()
            
        return predictions, probabilities


class BiasClassificationHead(nn.Module):
    """
    Custom classification head with optional hidden layers.
    
    This can be used instead of a simple linear layer for
    more complex classification.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 7,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        use_hidden_layer: bool = True
    ):
        """
        Initialize classification head.
        
        Args:
            hidden_size: BERT hidden dimension
            num_labels: Number of output labels
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
            use_hidden_layer: Whether to use intermediate layer
        """
        super().__init__()
        self.use_hidden_layer = use_hidden_layer
        
        if use_hidden_layer:
            self.dense = nn.Linear(hidden_size, hidden_dim)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim, num_labels)
        else:
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head."""
        if self.use_hidden_layer:
            x = self.dense(x)
            x = self.activation(x)
            x = self.dropout(x)
        else:
            x = self.dropout(x)
        
        logits = self.classifier(x)
        return logits


def create_model(
    model_name: str = "bert-base-uncased",
    num_labels: int = 7,
    classifier_dropout: float = 0.3,
    device: Optional[str] = None
) -> BertForMultiLabelBiasClassification:
    """
    Factory function to create the bias classification model.
    
    Args:
        model_name: HuggingFace model name
        num_labels: Number of bias types
        classifier_dropout: Dropout for classifier head
        device: Target device ('cuda', 'cpu', or None for auto)
        
    Returns:
        Initialized model
    """
    # Load BERT config
    config = BertConfig.from_pretrained(model_name)
    
    # Create model
    model = BertForMultiLabelBiasClassification(
        config=config,
        num_labels=num_labels,
        classifier_dropout=classifier_dropout
    )
    
    # Load pre-trained BERT weights
    bert_pretrained = BertModel.from_pretrained(model_name)
    model.bert.load_state_dict(bert_pretrained.state_dict())
    
    # Move to device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"Model created: {model_name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable:  {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Device:     {device}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    model = create_model()
    
    # Test forward pass
    batch_size = 2
    seq_length = 128
    
    # Create dummy inputs
    input_ids = torch.randint(0, 30000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, 2, (batch_size, 7)).float()
    
    # Forward pass
    outputs = model(
        input_ids=input_ids.to(model.bert.device),
        attention_mask=attention_mask.to(model.bert.device),
        labels=labels.to(model.bert.device)
    )
    
    print(f"\nTest forward pass:")
    print(f"  Loss:          {outputs['loss'].item():.4f}")
    print(f"  Logits shape:  {outputs['logits'].shape}")
    print(f"  Probs shape:   {outputs['probabilities'].shape}")
    print(f"  Sample probs:  {outputs['probabilities'][0].tolist()}")
