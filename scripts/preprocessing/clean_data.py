"""
Data Cleaning Pipeline for Perspective
========================================

This script handles:
1. Text normalization and cleaning
2. Duplicate detection and removal
3. Quality filtering
4. Preparing articles for LLM labeling
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import hashlib

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.schema import MIN_ARTICLE_LENGTH, MAX_ARTICLE_LENGTH, validate_article


class ArticleCleaner:
    """Handles text cleaning and normalization for news articles."""
    
    def __init__(self):
        # Patterns to remove
        self.patterns_to_remove = [
            r'(?i)also read:.*?(?=\n|$)',           # "Also read" links
            r'(?i)related:.*?(?=\n|$)',              # Related articles
            r'(?i)follow us on.*?(?=\n|$)',          # Social media links
            r'(?i)subscribe to.*?(?=\n|$)',          # Subscription prompts
            r'(?i)click here.*?(?=\n|$)',            # Click prompts
            r'(?i)advertisement\s*',                  # Ad markers
            r'(?i)\(with inputs from.*?\)',          # Agency credits
            r'(?i)pti|ani|ians',                     # News agency tags
            r'http[s]?://\S+',                       # URLs
            r'\s+',                                   # Multiple whitespace
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(p) for p in self.patterns_to_remove]
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize article text.
        
        Args:
            text: Raw article text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Apply all cleaning patterns
        cleaned = text
        for pattern in self.compiled_patterns:
            cleaned = pattern.sub(' ', cleaned)
        
        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def clean_headline(self, headline: str) -> str:
        """
        Clean article headline.
        
        Args:
            headline: Raw headline
            
        Returns:
            Cleaned headline
        """
        if not headline or not isinstance(headline, str):
            return ""
        
        # Remove extra whitespace
        cleaned = ' '.join(headline.split())
        
        # Remove trailing punctuation if excessive
        cleaned = re.sub(r'[!]{2,}', '!', cleaned)
        cleaned = re.sub(r'[?]{2,}', '?', cleaned)
        
        return cleaned.strip()
    
    def get_content_hash(self, text: str) -> str:
        """
        Generate hash for duplicate detection.
        
        Args:
            text: Article text
            
        Returns:
            MD5 hash of normalized text
        """
        # Normalize for hashing (lowercase, no whitespace)
        normalized = re.sub(r'\s+', '', text.lower())
        return hashlib.md5(normalized.encode()).hexdigest()


class DataCleaner:
    """Main class for cleaning the entire dataset."""
    
    def __init__(self, input_path: str, output_path: str):
        """
        Initialize the data cleaner.
        
        Args:
            input_path: Path to raw CSV file
            output_path: Path to save cleaned CSV
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.article_cleaner = ArticleCleaner()
        
        # Statistics
        self.stats = {
            "total_input": 0,
            "duplicates_removed": 0,
            "too_short": 0,
            "too_long": 0,
            "invalid": 0,
            "total_output": 0
        }
        
    def clean_dataset(self) -> pd.DataFrame:
        """
        Clean the entire dataset.
        
        Returns:
            Cleaned DataFrame
        """
        print(f"Loading data from {self.input_path}...")
        df = pd.read_csv(self.input_path)
        self.stats["total_input"] = len(df)
        print(f"Loaded {len(df)} articles")
        
        # Step 1: Clean text content
        print("\n[1/4] Cleaning text content...")
        df = self._clean_text_columns(df)
        
        # Step 2: Remove duplicates
        print("\n[2/4] Removing duplicates...")
        df = self._remove_duplicates(df)
        
        # Step 3: Filter by length
        print("\n[3/4] Filtering by article length...")
        df = self._filter_by_length(df)
        
        # Step 4: Validate and finalize
        print("\n[4/4] Validating articles...")
        df = self._validate_articles(df)
        
        # Add metadata
        df['word_count'] = df['content'].apply(lambda x: len(x.split()))
        df['cleaned'] = True
        
        self.stats["total_output"] = len(df)
        
        # Save cleaned data
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"\nSaved cleaned data to {self.output_path}")
        
        self._print_stats()
        
        return df
    
    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean headline and content columns."""
        tqdm.pandas(desc="Cleaning headlines")
        df['headline'] = df['headline'].progress_apply(
            self.article_cleaner.clean_headline
        )
        
        tqdm.pandas(desc="Cleaning content")
        df['content'] = df['content'].progress_apply(
            self.article_cleaner.clean_text
        )
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate articles based on content hash."""
        initial_count = len(df)
        
        # Generate content hashes
        df['content_hash'] = df['content'].apply(
            self.article_cleaner.get_content_hash
        )
        
        # Keep first occurrence of each hash
        df = df.drop_duplicates(subset=['content_hash'], keep='first')
        df = df.drop(columns=['content_hash'])
        
        self.stats["duplicates_removed"] = initial_count - len(df)
        print(f"  Removed {self.stats['duplicates_removed']} duplicates")
        
        return df
    
    def _filter_by_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter articles by word count."""
        initial_count = len(df)
        
        # Calculate word counts
        df['_word_count'] = df['content'].apply(lambda x: len(x.split()))
        
        # Track reasons for removal
        too_short = df['_word_count'] < MIN_ARTICLE_LENGTH
        too_long = df['_word_count'] > MAX_ARTICLE_LENGTH
        
        self.stats["too_short"] = too_short.sum()
        self.stats["too_long"] = too_long.sum()
        
        # Filter
        df = df[~too_short & ~too_long]
        df = df.drop(columns=['_word_count'])
        
        print(f"  Removed {self.stats['too_short']} articles (too short)")
        print(f"  Removed {self.stats['too_long']} articles (too long)")
        
        return df
    
    def _validate_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate articles against schema."""
        valid_mask = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
            is_valid, errors = validate_article(row.to_dict())
            valid_mask.append(is_valid)
        
        invalid_count = len(valid_mask) - sum(valid_mask)
        self.stats["invalid"] = invalid_count
        
        df = df[valid_mask]
        print(f"  Removed {invalid_count} invalid articles")
        
        return df
    
    def _print_stats(self):
        """Print cleaning statistics."""
        print("\n" + "=" * 50)
        print("CLEANING STATISTICS")
        print("=" * 50)
        print(f"Total input articles:   {self.stats['total_input']:,}")
        print(f"Duplicates removed:     {self.stats['duplicates_removed']:,}")
        print(f"Too short (< {MIN_ARTICLE_LENGTH} words): {self.stats['too_short']:,}")
        print(f"Too long (> {MAX_ARTICLE_LENGTH} words):  {self.stats['too_long']:,}")
        print(f"Invalid articles:       {self.stats['invalid']:,}")
        print("-" * 50)
        print(f"Total output articles:  {self.stats['total_output']:,}")
        retention_rate = (self.stats['total_output'] / self.stats['total_input']) * 100
        print(f"Retention rate:         {retention_rate:.1f}%")
        print("=" * 50)


def main():
    """Main entry point for data cleaning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean raw news article data")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/raw/articles_raw.csv",
        help="Path to raw CSV file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/processed/articles_cleaned.csv",
        help="Path to save cleaned CSV"
    )
    
    args = parser.parse_args()
    
    cleaner = DataCleaner(args.input, args.output)
    cleaner.clean_dataset()


if __name__ == "__main__":
    main()
