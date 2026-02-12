"""
Dataset and DataLoader for Perspective
========================================

PyTorch Dataset class for loading and preprocessing
news articles for bias classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from transformers import BertTokenizer

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.schema import BIAS_LABELS, LABEL_COLUMNS


class BiasDataset(Dataset):
    """
    PyTorch Dataset for bias classification.
    
    Handles:
    1. Loading articles from CSV
    2. Tokenization using BERT tokenizer
    3. Label encoding for multi-label classification
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: BertTokenizer,
        max_length: int = 512,
        include_headline: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to CSV file with articles and labels
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
            include_headline: Whether to prepend headline to content
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_headline = include_headline
        
        # Load data
        self.df = pd.read_csv(data_path)
        print(f"Loaded {len(self.df)} samples from {data_path}")
        
        # Validate columns
        self._validate_columns()
        
    def _validate_columns(self):
        """Validate that required columns exist."""
        required = ['content'] + LABEL_COLUMNS
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        row = self.df.iloc[idx]
        
        # Prepare text: optionally combine headline and content
        if self.include_headline and 'headline' in self.df.columns:
            text = f"{row['headline']} [SEP] {row['content']}"
        else:
            text = row['content']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract labels
        labels = torch.tensor(
            [row[col] for col in LABEL_COLUMNS],
            dtype=torch.float
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }
    
    def get_label_weights(self) -> torch.Tensor:
        """
        Calculate positive class weights for handling imbalance.
        
        For BCEWithLogitsLoss, pos_weight should be:
        pos_weight = num_negative / num_positive
        
        Returns:
            Tensor of weights for each label
        """
        weights = []
        for col in LABEL_COLUMNS:
            positive = self.df[col].sum()
            negative = len(self.df) - positive
            
            if positive > 0:
                weight = negative / positive
            else:
                weight = 1.0
            
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float)


def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None,
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 512,
    batch_size: int = 16,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], BertTokenizer]:
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV (optional)
        tokenizer_name: HuggingFace tokenizer name
        max_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, tokenizer)
    """
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    
    # Create datasets
    train_dataset = BiasDataset(train_path, tokenizer, max_length)
    val_dataset = BiasDataset(val_path, tokenizer, max_length)
    
    test_dataset = None
    if test_path:
        test_dataset = BiasDataset(test_path, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    if test_loader:
        print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, tokenizer


class InferenceDataset(Dataset):
    """
    Lightweight dataset for inference (no labels required).
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: BertTokenizer,
        max_length: int = 512
    ):
        """
        Initialize inference dataset.
        
        Args:
            texts: List of article texts
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


if __name__ == "__main__":
    # Test dataset creation
    print("Testing dataset module...")
    
    # Create a sample CSV for testing
    sample_data = {
        'article_id': ['test_001', 'test_002'],
        'headline': ['Test Headline 1', 'Test Headline 2'],
        'content': [
            'This is a test article about politics and government policies.',
            'Another article discussing regional issues and development.'
        ],
        'label_political': [1, 0],
        'label_gender': [0, 0],
        'label_entity': [0, 1],
        'label_racial': [0, 0],
        'label_religious': [0, 0],
        'label_regional': [0, 1],
        'label_sensationalism': [0, 0]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_path = Path("model/src/test_data.csv")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(test_path, index=False)
    
    # Test loading
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = BiasDataset(str(test_path), tokenizer)
    
    print(f"\nDataset length: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample attention_mask shape: {sample['attention_mask'].shape}")
    print(f"Sample labels: {sample['labels']}")
    
    # Get label weights
    weights = dataset.get_label_weights()
    print(f"\nLabel weights: {weights}")
    
    # Clean up
    test_path.unlink()
    print("\nTest completed successfully!")
