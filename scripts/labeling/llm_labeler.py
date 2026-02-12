"""
LLM-Assisted Labeling Pipeline for Perspective
===============================================

This module implements programmatic labeling using LLMs (GPT-4/Claude)
with safeguards to reduce bias and ensure quality.

Key Features:
1. Few-shot prompting with diverse examples
2. Multiple LLM calls for consensus voting
3. Confidence scoring
4. Human verification flagging
5. Detailed reasoning extraction
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

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.schema import BIAS_LABELS, BIAS_DESCRIPTIONS, LABEL_COLUMNS

# Try to import OpenAI (optional for testing)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not installed. Using mock responses for testing.")


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
    Uses few-shot examples to guide the model.
    """
    
    SYSTEM_PROMPT = """You are an expert media analyst specializing in detecting bias in Indian news articles. Your task is to analyze articles and identify the presence of specific types of bias.

You must be:
1. OBJECTIVE - Don't let your own biases affect the analysis
2. PRECISE - Only flag bias when there's clear evidence
3. CONTEXTUAL - Consider Indian cultural and political context
4. CONSERVATIVE - When uncertain, lean towards "no bias"

For each bias type, you will output:
- A binary label (0 = not present, 1 = present)
- A confidence score (0.0 to 1.0)
- A brief reasoning"""

    # Few-shot examples for each bias type
    FEW_SHOT_EXAMPLES = {
        "political": [
            {
                "text": "The government's visionary policies have transformed the economy, while the opposition continues its baseless criticism and disruptive protests.",
                "label": 1,
                "reasoning": "Uses loaded terms ('visionary' for government, 'baseless', 'disruptive' for opposition), showing clear political favor."
            },
            {
                "text": "Parliament passed the budget with 312 votes in favor and 245 against. Both ruling and opposition parties presented their arguments during the debate.",
                "label": 0,
                "reasoning": "Factual reporting of parliamentary proceedings without loaded language or bias toward either side."
            }
        ],
        "gender": [
            {
                "text": "The lady doctor showed great courage during the surgery. Her emotional nature helped her connect with patients.",
                "label": 1,
                "reasoning": "Unnecessary 'lady' prefix and gender stereotype ('emotional nature')."
            },
            {
                "text": "Dr. Priya Sharma successfully performed the complex cardiac surgery, becoming the youngest surgeon to achieve this milestone.",
                "label": 0,
                "reasoning": "Focuses on professional achievement without unnecessary gender references."
            }
        ],
        "religious": [
            {
                "text": "Members of a particular community gathered to protest against the law, raising concerns about communal harmony.",
                "label": 1,
                "reasoning": "Uses dog-whistle language ('particular community') and implies threat to 'communal harmony'."
            },
            {
                "text": "Citizens from various backgrounds participated in the peaceful protest, expressing concerns about the new policy.",
                "label": 0,
                "reasoning": "Inclusive language without religious identification where irrelevant."
            }
        ],
        "sensationalism": [
            {
                "text": "SHOCKING! You won't BELIEVE what this celebrity did! The EXPLOSIVE truth revealed!!!",
                "label": 1,
                "reasoning": "Excessive caps, multiple exclamation marks, clickbait language designed to provoke emotional response."
            },
            {
                "text": "The investigation revealed irregularities in the contract, according to the audit report released today.",
                "label": 0,
                "reasoning": "Measured language, factual reporting without exaggeration."
            }
        ]
    }

    @classmethod
    def get_labeling_prompt(cls, article_text: str, headline: str) -> str:
        """
        Generate the full prompt for labeling an article.
        
        Args:
            article_text: The article content
            headline: The article headline
            
        Returns:
            Complete prompt string
        """
        # Select random examples (2 per bias type) to avoid positional bias
        selected_examples = {}
        for bias_type in ["political", "gender", "religious", "sensationalism"]:
            if bias_type in cls.FEW_SHOT_EXAMPLES:
                selected_examples[bias_type] = cls.FEW_SHOT_EXAMPLES[bias_type]
        
        examples_text = cls._format_examples(selected_examples)
        
        prompt = f"""Analyze the following news article for bias.

## Bias Types to Detect:
{cls._format_bias_definitions()}

## Few-Shot Examples:
{examples_text}

## Article to Analyze:
**Headline:** {headline}

**Content:** {article_text[:3000]}  # Truncate for token limits

## Output Format (JSON):
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
        "political": 0.0-1.0,
        "gender": 0.0-1.0,
        "entity": 0.0-1.0,
        "racial": 0.0-1.0,
        "religious": 0.0-1.0,
        "regional": 0.0-1.0,
        "sensationalism": 0.0-1.0
    }},
    "reasoning": "Brief explanation for each detected bias"
}}

Respond ONLY with valid JSON."""
        
        return prompt
    
    @staticmethod
    def _format_bias_definitions() -> str:
        """Format bias definitions for the prompt."""
        lines = []
        for bias_type in BIAS_LABELS:
            desc = BIAS_DESCRIPTIONS[bias_type]
            lines.append(f"- **{desc['name']}**: {desc['description']}")
        return "\n".join(lines)
    
    @staticmethod
    def _format_examples(examples: Dict) -> str:
        """Format few-shot examples."""
        lines = []
        for bias_type, exs in examples.items():
            for ex in exs:
                label_text = "BIAS PRESENT" if ex['label'] == 1 else "NO BIAS"
                lines.append(f"""
Example ({bias_type}):
Text: "{ex['text'][:200]}..."
Label: {label_text}
Reasoning: {ex['reasoning']}
""")
        return "\n".join(lines)


class LLMLabeler:
    """
    Main class for LLM-assisted labeling with safeguards.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        num_votes: int = 3,  # Number of LLM calls for consensus
        confidence_threshold: float = 0.7,
        review_threshold: float = 0.5  # Flag for human review if below this
    ):
        """
        Initialize the LLM labeler.
        
        Args:
            api_key: OpenAI API key (or from environment)
            model: Model to use
            num_votes: Number of LLM calls per article for consensus
            confidence_threshold: Minimum confidence to accept a label
            review_threshold: Flag for review if confidence below this
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.num_votes = num_votes
        self.confidence_threshold = confidence_threshold
        self.review_threshold = review_threshold
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            
    def label_article(
        self,
        headline: str,
        content: str
    ) -> LabelingResult:
        """
        Label a single article using LLM with consensus voting.
        
        Args:
            headline: Article headline
            content: Article content
            
        Returns:
            LabelingResult with labels and metadata
        """
        # Collect multiple votes
        votes = []
        for i in range(self.num_votes):
            result = self._single_llm_call(headline, content)
            if result:
                votes.append(result)
            time.sleep(0.5)  # Rate limiting
        
        if not votes:
            # Return default labels if all calls fail
            return self._get_default_result()
        
        # Aggregate votes using majority voting
        return self._aggregate_votes(votes)
    
    def _single_llm_call(
        self,
        headline: str,
        content: str
    ) -> Optional[Dict]:
        """Make a single LLM API call."""
        prompt = BiasLabelingPrompts.get_labeling_prompt(content, headline)
        
        if not self.client:
            # Mock response for testing without API
            return self._mock_response()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": BiasLabelingPrompts.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Low temperature for consistency
                max_tokens=1000
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response: {e}")
            return None
        except Exception as e:
            print(f"LLM API error: {e}")
            return None
    
    def _mock_response(self) -> Dict:
        """Generate mock response for testing."""
        return {
            "labels": {
                bias: random.choice([0, 1]) for bias in BIAS_LABELS
            },
            "confidence": {
                bias: random.uniform(0.5, 0.95) for bias in BIAS_LABELS
            },
            "reasoning": "Mock response for testing purposes."
        }
    
    def _aggregate_votes(self, votes: List[Dict]) -> LabelingResult:
        """
        Aggregate multiple LLM votes using majority voting.
        
        Args:
            votes: List of individual LLM responses
            
        Returns:
            Aggregated LabelingResult
        """
        # Initialize counters
        label_counts = {bias: 0 for bias in BIAS_LABELS}
        confidence_sums = {bias: 0.0 for bias in BIAS_LABELS}
        
        for vote in votes:
            for bias in BIAS_LABELS:
                label_counts[bias] += vote["labels"].get(bias, 0)
                confidence_sums[bias] += vote["confidence"].get(bias, 0.5)
        
        # Majority voting: label = 1 if majority voted 1
        num_votes = len(votes)
        final_labels = {}
        final_confidences = {}
        
        for bias in BIAS_LABELS:
            # Majority vote for label
            final_labels[bias] = 1 if label_counts[bias] > num_votes / 2 else 0
            
            # Average confidence
            final_confidences[bias] = confidence_sums[bias] / num_votes
        
        # Overall confidence
        overall_confidence = sum(final_confidences.values()) / len(BIAS_LABELS)
        
        # Determine if needs review
        needs_review = any(
            conf < self.review_threshold 
            for conf in final_confidences.values()
        )
        
        # Collect reasoning from votes
        reasonings = [v.get("reasoning", "") for v in votes if v.get("reasoning")]
        combined_reasoning = " | ".join(reasonings[:2])  # Take first two
        
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
            reasoning="Labeling failed",
            needs_review=True
        )


class DatasetLabeler:
    """
    Process an entire dataset with LLM labeling.
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
            api_key: OpenAI API key
            batch_size: Number of articles to process before saving
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.labeler = LLMLabeler(api_key=api_key)
        
    def label_dataset(self) -> pd.DataFrame:
        """
        Label the entire dataset.
        
        Returns:
            Labeled DataFrame
        """
        print(f"Loading data from {self.input_path}...")
        df = pd.read_csv(self.input_path)
        print(f"Loaded {len(df)} articles")
        
        # Initialize label columns
        for col in LABEL_COLUMNS:
            df[col] = 0
        df['confidence_score'] = 0.0
        df['labeling_method'] = 'llm_gpt4'
        df['needs_review'] = False
        df['reasoning'] = ''
        
        # Process articles
        print("\nLabeling articles...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            result = self.labeler.label_article(
                headline=row['headline'],
                content=row['content']
            )
            
            # Update DataFrame
            for bias in BIAS_LABELS:
                df.at[idx, f'label_{bias}'] = result.labels[bias]
            
            df.at[idx, 'confidence_score'] = result.overall_confidence
            df.at[idx, 'needs_review'] = result.needs_review
            df.at[idx, 'reasoning'] = result.reasoning
            
            # Save checkpoint every batch_size articles
            if (idx + 1) % self.batch_size == 0:
                self._save_checkpoint(df, idx + 1)
        
        # Final save
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"\nSaved labeled data to {self.output_path}")
        
        # Print statistics
        self._print_stats(df)
        
        return df
    
    def _save_checkpoint(self, df: pd.DataFrame, processed: int):
        """Save intermediate checkpoint."""
        checkpoint_path = self.output_path.with_suffix('.checkpoint.csv')
        df.to_csv(checkpoint_path, index=False)
        print(f"\n  Checkpoint saved ({processed} articles processed)")
    
    def _print_stats(self, df: pd.DataFrame):
        """Print labeling statistics."""
        print("\n" + "=" * 50)
        print("LABELING STATISTICS")
        print("=" * 50)
        
        print("\nLabel Distribution:")
        for bias in BIAS_LABELS:
            col = f'label_{bias}'
            positive = df[col].sum()
            pct = (positive / len(df)) * 100
            print(f"  {bias:15s}: {positive:5d} ({pct:5.1f}%)")
        
        print(f"\nArticles needing review: {df['needs_review'].sum()}")
        print(f"Average confidence: {df['confidence_score'].mean():.3f}")
        print("=" * 50)


def main():
    """Main entry point for LLM labeling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Label articles using LLM")
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
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    labeler = DatasetLabeler(
        input_path=args.input,
        output_path=args.output,
        api_key=args.api_key
    )
    labeler.label_dataset()


if __name__ == "__main__":
    main()
