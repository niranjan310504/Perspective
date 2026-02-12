"""
Training Script for Perspective Bias Detection Model
=====================================================

This script handles:
1. Model training with early stopping
2. Validation and metrics computation
3. Checkpoint saving
4. Training visualization
"""

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, multilabel_confusion_matrix
)

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.src.bert_classifier import create_model
from model.src.dataset import create_dataloaders
from model.config.model_config import ModelConfig, DEFAULT_CONFIG
from data.schema import BIAS_LABELS


class Trainer:
    """
    Trainer class for BERT multi-label bias classifier.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        train_path: str,
        val_path: str,
        output_dir: str
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Model configuration
            train_path: Path to training data
            val_path: Path to validation data
            output_dir: Directory to save checkpoints and logs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create dataloaders
        self.train_loader, self.val_loader, _, self.tokenizer = create_dataloaders(
            train_path=train_path,
            val_path=val_path,
            tokenizer_name=config.model_name,
            max_length=config.max_length,
            batch_size=config.batch_size
        )
        
        # Create model
        self.model = create_model(
            model_name=config.model_name,
            num_labels=config.num_labels,
            classifier_dropout=config.classifier_dropout,
            device=str(self.device)
        )
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        total_steps = len(self.train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        self.scheduler = self._create_scheduler(total_steps, warmup_steps)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_f1 = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'learning_rates': []
        }
        
    def _create_scheduler(
        self,
        total_steps: int,
        warmup_steps: int
    ):
        """Create learning rate scheduler with warmup."""
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        decay_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps - warmup_steps
        )
        
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps]
        )
    
    def train(self) -> Dict:
        """
        Main training loop.
        
        Returns:
            Training history dictionary
        """
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("=" * 60 + "\n")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate()
            
            # Log progress
            self._log_epoch(train_loss, val_metrics)
            
            # Save checkpoint if best
            if val_metrics['macro_f1'] > self.best_f1:
                self.best_f1 = val_metrics['macro_f1']
                self._save_checkpoint(is_best=True)
                print(f"  ★ New best model! F1: {self.best_f1:.4f}")
            
            # Save periodic checkpoint
            self._save_checkpoint(is_best=False)
        
        # Save final model
        self._save_final_model()
        
        # Save training history
        self._save_history()
        
        print("\n" + "=" * 60)
        print(f"TRAINING COMPLETE - Best F1: {self.best_f1:.4f}")
        print("=" * 60)
        
        return self.training_history
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
            leave=True
        )
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        avg_loss = total_loss / num_batches
        self.training_history['train_loss'].append(avg_loss)
        self.training_history['learning_rates'].append(current_lr)
        
        return avg_loss
    
    def _validate(self) -> Dict:
        """
        Validate the model.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs['loss'].item()
                
                # Get predictions
                probs = outputs['probabilities']
                preds = (probs >= self.config.default_threshold).int()
                
                all_predictions.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probabilities.append(probs.cpu().numpy())
        
        # Concatenate all batches
        predictions = np.vstack(all_predictions)
        labels = np.vstack(all_labels)
        probabilities = np.vstack(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, labels)
        metrics['avg_loss'] = total_loss / len(self.val_loader)
        
        self.training_history['val_loss'].append(metrics['avg_loss'])
        self.training_history['val_f1'].append(metrics['macro_f1'])
        
        return metrics
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Binary predictions [N, num_labels]
            labels: Ground truth [N, num_labels]
            
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        # Macro-averaged metrics
        metrics['macro_f1'] = f1_score(labels, predictions, average='macro', zero_division=0)
        metrics['macro_precision'] = precision_score(labels, predictions, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(labels, predictions, average='macro', zero_division=0)
        
        # Micro-averaged metrics
        metrics['micro_f1'] = f1_score(labels, predictions, average='micro', zero_division=0)
        
        # Per-label metrics
        metrics['per_label'] = {}
        for i, label_name in enumerate(BIAS_LABELS):
            metrics['per_label'][label_name] = {
                'f1': f1_score(labels[:, i], predictions[:, i], zero_division=0),
                'precision': precision_score(labels[:, i], predictions[:, i], zero_division=0),
                'recall': recall_score(labels[:, i], predictions[:, i], zero_division=0)
            }
        
        return metrics
    
    def _log_epoch(self, train_loss: float, val_metrics: Dict):
        """Log epoch results."""
        print(f"\nEpoch {self.current_epoch + 1}/{self.config.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_metrics['avg_loss']:.4f}")
        print(f"  Macro F1:   {val_metrics['macro_f1']:.4f}")
        print(f"  Macro P:    {val_metrics['macro_precision']:.4f}")
        print(f"  Macro R:    {val_metrics['macro_recall']:.4f}")
        
        print("\n  Per-label F1:")
        for label, scores in val_metrics['per_label'].items():
            print(f"    {label:15s}: {scores['f1']:.4f}")
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_f1': self.best_f1,
            'config': self.config.to_dict()
        }
        
        if is_best:
            path = self.output_dir / 'best_model.pt'
        else:
            path = self.output_dir / f'checkpoint_epoch_{self.current_epoch + 1}.pt'
        
        torch.save(checkpoint, path)
    
    def _save_final_model(self):
        """Save the final model for inference."""
        # Save model weights
        model_path = self.output_dir / 'model.pt'
        torch.save(self.model.state_dict(), model_path)
        
        # Save tokenizer
        tokenizer_path = self.output_dir / 'tokenizer'
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save config
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        print(f"\nFinal model saved to {self.output_dir}/")
    
    def _save_history(self):
        """Save training history."""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train bias detection model")
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/splits/train.csv",
        help="Path to training data"
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default="data/splits/val.csv",
        help="Path to validation data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model/checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = ModelConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Create trainer
    trainer = Trainer(
        config=config,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
