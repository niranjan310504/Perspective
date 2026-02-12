"""
LLM-Assisted Labeling Pipeline using Google Gemini
===================================================

Uses Gemini Pro for programmatic labeling with safeguards.
Gemini Pro is FREE with generous rate limits (60 RPM).

Get your API key at: https://makersuite.google.com/app/apikey

Key Features:
1. Few-shot prompting with diverse examples
2. Multiple LLM calls for consensus voting
3. Confidence scoring
4. Human verification flagging
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import time
import random
import re

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.schema import BIAS_LABELS, BIAS_DESCRIPTIONS, LABEL_COLUMNS

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed.")
    print("Install with: pip install google-generativeai")


@dataclass
class LabelingResult:
    """Container for labeling results."""
    labels: Dict[str, int]           # Binary labels (0/1)
    confidence_scores: Dict[str, float]  # Confidence per label
    overall_confidence: float         # Average confidence
    reasoning: str                    # LLM's explanation
    needs_review: bool               # Flag for human review


class BiasLabelingPrompts:
    """
    Prompt templates for LLM-assisted labeling.
    """
    
    SYSTEM_PROMPT = """You are an expert media analyst specializing in detecting bias in Indian news articles. 

Your task is to analyze articles and identify 7 types of bias:
1. Political - Favoring/opposing political parties, leaders, or ideologies
2. Gender - Stereotyping or discrimination based on gender
3. Entity - Undue favor/criticism toward organizations or public figures
4. Racial - Discrimination based on race, caste, or ethnicity
5. Religious - Favoring or targeting religious groups
6. Regional - Bias toward/against states or regions
7. Sensationalism - Clickbait, exaggeration, fear-mongering

RULES:
- Be OBJECTIVE - Don't let your own biases affect analysis
- Be PRECISE - Only flag bias when there's clear evidence
- Consider INDIAN CONTEXT - Understand local politics and culture
- Be CONSERVATIVE - When uncertain, lean towards "no bias"
"""

    FEW_SHOT_EXAMPLES = """
## EXAMPLES:

### Example 1 - Political Bias:
Text: "The government's visionary policies have transformed the economy, while opposition continues baseless criticism."
Labels: {"political": 1, "sensationalism": 1}
Reasoning: Uses "visionary" for government, "baseless" for opposition - clear political bias.

### Example 2 - No Bias:
Text: "Parliament passed the budget with 312 votes in favor and 245 against. Both parties presented arguments."
Labels: {"political": 0, "gender": 0, "entity": 0, "racial": 0, "religious": 0, "regional": 0, "sensationalism": 0}
Reasoning: Factual reporting without loaded language.

### Example 3 - Religious Bias:
Text: "Members of a particular community gathered to protest, raising concerns about communal harmony."
Labels: {"religious": 1}
Reasoning: Uses dog-whistle "particular community" and implies threat.

### Example 4 - Gender Bias:
Text: "The lady doctor showed great courage. Her emotional nature helped connect with patients."
Labels: {"gender": 1}
Reasoning: Unnecessary "lady" prefix and "emotional" stereotype.

### Example 5 - Sensationalism:
Text: "SHOCKING! You won't BELIEVE what this celebrity did! EXPLOSIVE truth revealed!!!"
Labels: {"sensationalism": 1}
Reasoning: All-caps, excessive punctuation, clickbait language.
"""

    @classmethod
    def get_labeling_prompt(cls, article_text: str, headline: str) -> str:
        """Generate the full prompt for labeling an article."""
        
        # Truncate article if too long (Gemini has good context but let's be safe)
        max_chars = 4000
        if len(article_text) > max_chars:
            article_text = article_text[:max_chars] + "..."
        
        prompt = f"""{cls.SYSTEM_PROMPT}

{cls.FEW_SHOT_EXAMPLES}

---

## ARTICLE TO ANALYZE:

**Headline:** {headline}

**Content:** {article_text}

---

## YOUR RESPONSE:

Respond with ONLY valid JSON in this exact format:
{{
    "labels": {{
        "political": 0 or 1,
        "gender": 0 or 1,
        "entity": 0 or 1,
        "racial": 0 or 1,
        "religious": 0 or 1,
        "regional": 0 or 1,
        "sensationalism": 0 or 1
    }},
    "confidence": {{
        "political": 0.0 to 1.0,
        "gender": 0.0 to 1.0,
        "entity": 0.0 to 1.0,
        "racial": 0.0 to 1.0,
        "religious": 0.0 to 1.0,
        "regional": 0.0 to 1.0,
        "sensationalism": 0.0 to 1.0
    }},
    "reasoning": "Brief explanation"
}}
"""
        return prompt


class GeminiLabeler:
    """
    LLM labeler using Google Gemini Pro.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",  # Fast and free
        num_votes: int = 1,  # Single call is usually enough for Gemini
        confidence_threshold: float = 0.7,
        review_threshold: float = 0.5
    ):
        """
        Initialize the Gemini labeler.
        
        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            model_name: Gemini model to use
            num_votes: Number of LLM calls per article for consensus
            confidence_threshold: Minimum confidence to accept
            review_threshold: Flag for review if below this
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.num_votes = num_votes
        self.confidence_threshold = confidence_threshold
        self.review_threshold = review_threshold
        
        if GEMINI_AVAILABLE and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
            print(f"✓ Gemini initialized: {model_name}")
        else:
            self.model = None
            if not self.api_key:
                print("Warning: No API key. Set GOOGLE_API_KEY env var.")
                print("Get key at: https://makersuite.google.com/app/apikey")
    
    def label_article(
        self,
        headline: str,
        content: str
    ) -> LabelingResult:
        """
        Label a single article using Gemini.
        
        Args:
            headline: Article headline
            content: Article content
            
        Returns:
            LabelingResult with labels and metadata
        """
        if not self.model:
            return self._get_default_result()
        
        # Collect votes (usually 1 is enough for Gemini)
        votes = []
        for _ in range(self.num_votes):
            result = self._single_gemini_call(headline, content)
            if result:
                votes.append(result)
            time.sleep(0.5)  # Rate limiting
        
        if not votes:
            return self._get_default_result()
        
        return self._aggregate_votes(votes)
    
    def _single_gemini_call(
        self,
        headline: str,
        content: str
    ) -> Optional[Dict]:
        """Make a single Gemini API call."""
        prompt = BiasLabelingPrompts.get_labeling_prompt(content, headline)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,  # Low for consistency
                    max_output_tokens=1000,
                )
            )
            
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Try to find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())
            else:
                print(f"No JSON found in response: {response_text[:200]}")
                return None
                
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return None
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None
    
    def _aggregate_votes(self, votes: List[Dict]) -> LabelingResult:
        """Aggregate votes using majority voting."""
        label_counts = {bias: 0 for bias in BIAS_LABELS}
        confidence_sums = {bias: 0.0 for bias in BIAS_LABELS}
        
        for vote in votes:
            for bias in BIAS_LABELS:
                label_counts[bias] += vote.get("labels", {}).get(bias, 0)
                confidence_sums[bias] += vote.get("confidence", {}).get(bias, 0.5)
        
        num_votes = len(votes)
        final_labels = {}
        final_confidences = {}
        
        for bias in BIAS_LABELS:
            final_labels[bias] = 1 if label_counts[bias] > num_votes / 2 else 0
            final_confidences[bias] = confidence_sums[bias] / num_votes
        
        overall_confidence = sum(final_confidences.values()) / len(BIAS_LABELS)
        
        needs_review = any(
            conf < self.review_threshold 
            for conf in final_confidences.values()
        )
        
        reasonings = [v.get("reasoning", "") for v in votes if v.get("reasoning")]
        combined_reasoning = " | ".join(reasonings[:2])
        
        return LabelingResult(
            labels=final_labels,
            confidence_scores=final_confidences,
            overall_confidence=overall_confidence,
            reasoning=combined_reasoning,
            needs_review=needs_review
        )
    
    def _get_default_result(self) -> LabelingResult:
        """Return default result when labeling fails."""
        return LabelingResult(
            labels={bias: 0 for bias in BIAS_LABELS},
            confidence_scores={bias: 0.0 for bias in BIAS_LABELS},
            overall_confidence=0.0,
            reasoning="Labeling failed - using defaults",
            needs_review=True
        )


class DatasetLabeler:
    """
    Process an entire dataset with Gemini labeling.
    """
    
    def __init__(
        self,
        input_path: str,
        output_path: str,
        api_key: Optional[str] = None,
        batch_size: int = 50
    ):
        """
        Initialize the dataset labeler.
        
        Args:
            input_path: Path to cleaned CSV
            output_path: Path to save labeled CSV
            api_key: Google API key
            batch_size: Number of articles to process before saving checkpoint
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.labeler = GeminiLabeler(api_key=api_key)
        
    def label_dataset(
        self,
        max_articles: Optional[int] = None,
        resume: bool = True
    ) -> pd.DataFrame:
        """
        Label the entire dataset.
        
        Args:
            max_articles: Maximum articles to label (None = all)
            resume: Whether to resume from checkpoint
            
        Returns:
            Labeled DataFrame
        """
        print(f"Loading data from {self.input_path}...")
        df = pd.read_csv(self.input_path)
        
        if max_articles:
            df = df.head(max_articles)
        
        print(f"Processing {len(df)} articles")
        
        # Initialize or load progress
        checkpoint_path = self.output_path.with_suffix('.checkpoint.csv')
        start_idx = 0
        
        if resume and checkpoint_path.exists():
            checkpoint_df = pd.read_csv(checkpoint_path)
            start_idx = len(checkpoint_df)
            print(f"Resuming from checkpoint: {start_idx} already done")
            
            # Merge checkpoint with remaining
            if start_idx < len(df):
                df = pd.concat([
                    checkpoint_df,
                    df.iloc[start_idx:]
                ]).reset_index(drop=True)
        
        # Initialize label columns if not present
        for col in LABEL_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        
        if 'confidence_score' not in df.columns:
            df['confidence_score'] = 0.0
        if 'labeling_method' not in df.columns:
            df['labeling_method'] = ''
        if 'needs_review' not in df.columns:
            df['needs_review'] = False
        if 'reasoning' not in df.columns:
            df['reasoning'] = ''
        
        # Process articles
        print("\nLabeling articles with Gemini...")
        print("Rate: ~60 per minute (free tier)")
        print("-" * 50)
        
        for idx in tqdm(range(start_idx, len(df))):
            row = df.iloc[idx]
            
            # Skip if already labeled
            if row.get('labeling_method') == 'gemini':
                continue
            
            result = self.labeler.label_article(
                headline=str(row.get('headline', '')),
                content=str(row.get('content', ''))
            )
            
            # Update DataFrame
            for bias in BIAS_LABELS:
                df.at[idx, f'label_{bias}'] = result.labels[bias]
            
            df.at[idx, 'confidence_score'] = result.overall_confidence
            df.at[idx, 'labeling_method'] = 'gemini'
            df.at[idx, 'needs_review'] = result.needs_review
            df.at[idx, 'reasoning'] = result.reasoning[:500]  # Truncate
            
            # Save checkpoint
            if (idx + 1) % self.batch_size == 0:
                df.iloc[:idx+1].to_csv(checkpoint_path, index=False)
                print(f"\n  ✓ Checkpoint: {idx + 1}/{len(df)} articles")
            
            # Rate limiting (60 RPM = 1 per second)
            time.sleep(1.0)
        
        # Final save
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"\n✓ Saved labeled data to {self.output_path}")
        
        # Clean up checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        # Print statistics
        self._print_stats(df)
        
        return df
    
    def _print_stats(self, df: pd.DataFrame):
        """Print labeling statistics."""
        print("\n" + "=" * 50)
        print("LABELING STATISTICS")
        print("=" * 50)
        
        print("\nLabel Distribution:")
        for bias in BIAS_LABELS:
            col = f'label_{bias}'
            positive = df[col].sum()
            pct = (positive / len(df)) * 100 if len(df) > 0 else 0
            print(f"  {bias:15s}: {positive:5.0f} ({pct:5.1f}%)")
        
        print(f"\nArticles needing review: {df['needs_review'].sum()}")
        print(f"Average confidence: {df['confidence_score'].mean():.3f}")
        print("=" * 50)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Label articles using Gemini")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/processed/articles_cleaned.csv",
        help="Path to cleaned CSV file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/processed/articles_labeled.csv",
        help="Path to save labeled CSV"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Google API key (or set GOOGLE_API_KEY env var)"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum articles to label (for testing)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, don't resume from checkpoint"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("PERSPECTIVE - Gemini Labeling")
    print("=" * 50)
    print("\nUsing Google Gemini Pro for bias labeling")
    print("Get API key at: https://makersuite.google.com/app/apikey\n")
    
    labeler = DatasetLabeler(
        input_path=args.input,
        output_path=args.output,
        api_key=args.api_key
    )
    
    labeler.label_dataset(
        max_articles=args.max_articles,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
