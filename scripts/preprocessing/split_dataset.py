"""
Dataset Splitting Script for Perspective
==========================================

Splits the labeled dataset into train/validation/test sets
with stratified sampling to maintain label distribution.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from collections import Counter
import argparse

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.schema import BIAS_LABELS, LABEL_COLUMNS


class DatasetSplitter:
    """
    Splits dataset with stratified sampling for multi-label classification.
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        min_confidence: float = 0.7
    ):
        """
        Initialize the splitter.
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_seed: Random seed for reproducibility
            min_confidence: Minimum LLM confidence to include in training
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
            "Ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.min_confidence = min_confidence
        
    def create_stratification_key(self, df: pd.DataFrame) -> pd.Series:
        """
        Create a stratification key for multi-label data.
        
        For multi-label classification, we can't directly stratify.
        Instead, we create a composite key based on label patterns.
        
        Args:
            df: DataFrame with label columns
            
        Returns:
            Series with stratification keys
        """
        # Create a string representation of the label combination
        # Example: "1_0_1_0_0_1_0" for a specific label pattern
        label_patterns = df[LABEL_COLUMNS].apply(
            lambda row: '_'.join(row.astype(str)), 
            axis=1
        )
        return label_patterns
    
    def split_dataset(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/val/test with stratification.
        
        Args:
            df: Full labeled dataset
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Filter by confidence if column exists
        if 'confidence_score' in df.columns:
            high_confidence = df[df['confidence_score'] >= self.min_confidence]
            low_confidence = df[df['confidence_score'] < self.min_confidence]
            
            print(f"High confidence samples (>= {self.min_confidence}): {len(high_confidence)}")
            print(f"Low confidence samples (excluded): {len(low_confidence)}")
            
            df = high_confidence.copy()
        
        # Create stratification key
        strat_key = self.create_stratification_key(df)
        
        # Handle rare patterns by grouping them
        pattern_counts = strat_key.value_counts()
        rare_patterns = pattern_counts[pattern_counts < 5].index
        strat_key = strat_key.apply(
            lambda x: 'rare' if x in rare_patterns else x
        )
        
        # First split: train + val/test
        train_df, temp_df, train_strat, temp_strat = train_test_split(
            df, strat_key,
            train_size=self.train_ratio,
            random_state=self.random_seed,
            stratify=strat_key
        )
        
        # Second split: val/test
        relative_test_ratio = self.test_ratio / (self.val_ratio + self.test_ratio)
        
        # Re-create stratification key for temp_df
        temp_strat_key = self.create_stratification_key(temp_df)
        pattern_counts = temp_strat_key.value_counts()
        rare_patterns = pattern_counts[pattern_counts < 2].index
        temp_strat_key = temp_strat_key.apply(
            lambda x: 'rare' if x in rare_patterns else x
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_ratio,
            random_state=self.random_seed,
            stratify=temp_strat_key
        )
        
        return train_df, val_df, test_df
    
    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path
    ):
        """Save the dataset splits to CSV files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "val.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)
        
        print(f"\nSaved splits to {output_dir}/")
        print(f"  train.csv: {len(train_df)} samples")
        print(f"  val.csv:   {len(val_df)} samples")
        print(f"  test.csv:  {len(test_df)} samples")
    
    def print_split_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):
        """Print label distribution statistics for each split."""
        splits = {
            'Train': train_df,
            'Val': val_df,
            'Test': test_df
        }
        
        print("\n" + "=" * 70)
        print("LABEL DISTRIBUTION ACROSS SPLITS")
        print("=" * 70)
        
        # Header
        print(f"\n{'Bias Type':<18} | {'Train %':>8} | {'Val %':>8} | {'Test %':>8}")
        print("-" * 52)
        
        for bias in BIAS_LABELS:
            col = f'label_{bias}'
            row = f"{bias:<18}"
            for name, df in splits.items():
                pct = (df[col].sum() / len(df)) * 100
                row += f" | {pct:>7.1f}%"
            print(row)
        
        print("-" * 52)
        
        # Total samples
        print(f"\n{'Total Samples':<18}", end="")
        for name, df in splits.items():
            print(f" | {len(df):>8}", end="")
        print()
        print("=" * 70)


def analyze_label_distribution(df: pd.DataFrame):
    """Analyze and print label distribution in dataset."""
    print("\n" + "=" * 50)
    print("DATASET LABEL ANALYSIS")
    print("=" * 50)
    
    print("\nIndividual Label Distribution:")
    for bias in BIAS_LABELS:
        col = f'label_{bias}'
        positive = df[col].sum()
        negative = len(df) - positive
        pct = (positive / len(df)) * 100
        print(f"  {bias:15s}: {positive:5d} positive ({pct:5.1f}%) | {negative:5d} negative")
    
    # Multi-label statistics
    label_counts = df[LABEL_COLUMNS].sum(axis=1)
    print(f"\nArticles with no bias: {(label_counts == 0).sum()}")
    print(f"Articles with 1 bias:  {(label_counts == 1).sum()}")
    print(f"Articles with 2+ bias: {(label_counts >= 2).sum()}")
    print(f"Average biases per article: {label_counts.mean():.2f}")
    
    # Label co-occurrence
    print("\nLabel Co-occurrence Matrix:")
    cooc = df[LABEL_COLUMNS].T.dot(df[LABEL_COLUMNS])
    print(cooc)


def main():
    parser = argparse.ArgumentParser(description="Split labeled dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/articles_labeled.csv",
        help="Path to labeled CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits",
        help="Directory to save splits"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence score to include"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} articles")
    
    # Analyze distribution
    analyze_label_distribution(df)
    
    # Split dataset
    splitter = DatasetSplitter(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        min_confidence=args.min_confidence
    )
    
    train_df, val_df, test_df = splitter.split_dataset(df)
    
    # Print statistics
    splitter.print_split_statistics(train_df, val_df, test_df)
    
    # Save splits
    splitter.save_splits(
        train_df, val_df, test_df,
        Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
