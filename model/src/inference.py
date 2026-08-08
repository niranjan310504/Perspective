"""
Inference Module for Perspective Bias Detection
=================================================

This module provides classes for:
1. Loading trained models
2. Running inference on text/articles
3. Post-processing predictions
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import torch
from transformers import BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification

# Setup logging
logger = logging.getLogger(__name__)

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.schema import BIAS_LABELS, BIAS_DESCRIPTIONS


BIAS_KEYWORDS = {
    'political': ['government', 'opposition', 'party', 'prime minister', 'chief minister', 'election', 'bjp', 'congress'],
    'gender': ['woman', 'women', 'man', 'men', 'lady', 'female', 'male', 'gender'],
    'entity': ['company', 'corporation', 'organization', 'minister', 'leader', 'officials', 'business'],
    'racial': ['race', 'ethnic', 'caste', 'community', 'minority', 'majority'],
    'religious': ['religion', 'religious', 'muslim', 'hindu', 'christian', 'sikh', 'temple', 'mosque', 'church'],
    'regional': ['state', 'region', 'north', 'south', 'east', 'west', 'tamil nadu', 'kerala', 'up', 'bihar', 'maharashtra'],
    'sensationalism': ['shocking', 'breaking', 'horrific', 'explosive', 'massive', 'disaster', 'carnage', 'outrage'],
}


class BiasPredictor:
    """
    Inference class for bias detection.
    
    Handles model loading, preprocessing, inference, and
    post-processing of predictions.
    """
    
    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        threshold: float = 0.5
    ):
        """
        Initialize the predictor.
        
        Args:
            model_dir: Directory containing model files (supports HuggingFace format)
            device: Target device ('cuda' or 'cpu')
            threshold: Classification threshold
        """
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Find the actual model directory (check for perspective_model subfolder)
        self.actual_model_dir = self._find_model_dir()
        
        # Load components
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.num_labels = 7  # 7 bias types
        
        logger.info(f"BiasPredictor initialized on {self.device}")
    
    def _find_model_dir(self) -> Path:
        """Find the actual model directory."""
        # Check if model files exist directly in model_dir
        if (self.model_dir / 'model.safetensors').exists() or (self.model_dir / 'pytorch_model.bin').exists():
            return self.model_dir
        
        # Check for perspective_model subfolder
        perspective_dir = self.model_dir / 'perspective_model'
        if perspective_dir.exists():
            if (perspective_dir / 'model.safetensors').exists() or (perspective_dir / 'pytorch_model.bin').exists():
                return perspective_dir
        
        # Check for model.pt (legacy format)
        if (self.model_dir / 'model.pt').exists():
            return self.model_dir
            
        raise FileNotFoundError(f"No model found in {self.model_dir} or {self.model_dir / 'perspective_model'}")
    
    def _load_tokenizer(self):
        """Load tokenizer from model directory or HuggingFace."""
        try:
            # Try loading from model directory (HuggingFace format)
            return AutoTokenizer.from_pretrained(str(self.actual_model_dir))
        except Exception as e:
            logger.warning(f"Could not load tokenizer from {self.actual_model_dir}: {e}")
            logger.info("Loading tokenizer from HuggingFace (bert-base-uncased)")
            return AutoTokenizer.from_pretrained('bert-base-uncased')
    
    def _load_model(self):
        """Load trained model (supports HuggingFace format with safetensors)."""
        # Check for HuggingFace format (model.safetensors or pytorch_model.bin)
        safetensors_path = self.actual_model_dir / 'model.safetensors'
        pytorch_path = self.actual_model_dir / 'pytorch_model.bin'
        legacy_path = self.actual_model_dir / 'model.pt'
        
        if safetensors_path.exists() or pytorch_path.exists():
            # Load HuggingFace format model
            logger.info(f"Loading HuggingFace model from {self.actual_model_dir}")
            model = AutoModelForSequenceClassification.from_pretrained(
                str(self.actual_model_dir),
                local_files_only=True  # Security: only load local files
            )
            model = model.to(self.device)
            model.eval()
            return model
        elif legacy_path.exists():
            # Load legacy format
            from model.src.bert_classifier import BertForMultiLabelBiasClassification
            from model.config.model_config import ModelConfig
            from transformers import BertConfig
            
            config = ModelConfig()
            bert_config = BertConfig.from_pretrained(config.model_name)
            model = BertForMultiLabelBiasClassification(
                config=bert_config,
                num_labels=config.num_labels,
                classifier_dropout=config.classifier_dropout
            )
            state_dict = torch.load(legacy_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            return model
        else:
            raise FileNotFoundError(f"No model found in {self.actual_model_dir}")

    def _forward_logits(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the underlying model and normalize the logits output."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        if hasattr(outputs, 'logits'):
            return outputs.logits
        if isinstance(outputs, dict) and 'logits' in outputs:
            return outputs['logits']
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def _build_result(self, probabilities) -> Dict[str, Any]:
        """Format probabilities into the project response schema."""
        predictions = (probabilities >= self.threshold).astype(int)

        result: Dict[str, Any] = {
            'biases': {},
            'detected_biases': [],
            'summary': '',
        }

        for i, label_name in enumerate(BIAS_LABELS):
            result['biases'][label_name] = {
                'detected': bool(predictions[i]),
                'score': float(probabilities[i])
            }

            if predictions[i]:
                result['detected_biases'].append(label_name)

        result['summary'] = self._generate_summary(result['detected_biases'])
        return result

    def _heuristic_explanations(self, text: str, detected_biases: List[str], reason: Optional[str] = None) -> Dict[str, Any]:
        """Create deterministic keyword-based explanations as fallback."""
        lowered = text.lower()
        explanations: Dict[str, Any] = {
            'method': 'heuristic_keywords',
            'reason': reason,
            'labels': {},
        }

        labels_to_check = detected_biases or list(BIAS_LABELS)

        for label in labels_to_check:
            matches = [keyword for keyword in BIAS_KEYWORDS.get(label, []) if keyword in lowered]
            if not matches:
                continue

            explanations['labels'][label] = {
                'method': 'heuristic_keywords',
                'summary': f"Keyword cues matched for {label.replace('_', ' ')}.",
                'highlights': [
                    {
                        'text': keyword,
                        'score': 0.72,
                        'reason': 'keyword match',
                    }
                    for keyword in matches[:5]
                ],
            }

        return explanations

    def _merge_spans(self, spans: List[Dict[str, Any]], max_gap: int = 2) -> List[Dict[str, Any]]:
        """Merge nearby token spans into short phrase-level highlights."""
        if not spans:
            return []

        ordered = sorted(spans, key=lambda item: (item['start'], item['end']))
        merged: List[Dict[str, Any]] = [ordered[0].copy()]

        for span in ordered[1:]:
            current = merged[-1]
            if span['start'] <= current['end'] + max_gap:
                current['end'] = max(current['end'], span['end'])
                current['score'] = max(current['score'], span['score'])
            else:
                merged.append(span.copy())

        return merged

    def _build_span_highlights(
        self,
        text: str,
        input_ids: torch.Tensor,
        token_scores: torch.Tensor,
        offsets,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Convert token scores to readable text spans."""
        special_token_ids = {
            token_id for token_id in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]
            if token_id is not None
        }

        candidates: List[Dict[str, Any]] = []
        input_ids_list = input_ids[0].tolist()
        token_scores_list = token_scores.tolist()

        for index, token_id in enumerate(input_ids_list):
            if token_id in special_token_ids:
                continue

            start, end = offsets[index]
            if end <= start:
                continue

            phrase = text[start:end].strip()
            if not phrase:
                continue

            candidates.append({
                'text': phrase,
                'start': int(start),
                'end': int(end),
                'score': float(token_scores_list[index]),
            })

        if not candidates:
            return []

        top_candidates = sorted(candidates, key=lambda item: item['score'], reverse=True)[:max(top_k * 2, top_k)]
        merged = self._merge_spans(top_candidates)

        for span in merged:
            span['text'] = text[span['start']:span['end']].strip()

        return merged[:top_k]

    def _gradient_explanation(
        self,
        text: str,
        label_index: int,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Compute gradient-based token attribution for a single label."""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True,
        )

        offsets = encoding.pop('offset_mapping')[0].tolist()
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        embedding_layer = self.model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids).detach().requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        logits = self._forward_logits(
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        target_logit = logits[0, label_index]
        target_logit.backward()

        token_scores = (inputs_embeds.grad * inputs_embeds).abs().sum(dim=-1).squeeze(0).detach().cpu()
        highlights = self._build_span_highlights(text, input_ids.detach().cpu(), token_scores, offsets, top_k)

        return {
            'method': 'gradient_attribution',
            'label': BIAS_LABELS[label_index],
            'summary': f"Top gradient-based spans supporting {BIAS_LABELS[label_index].replace('_', ' ')}.",
            'highlights': highlights,
        }

    def explain(
        self,
        text: str,
        detected_biases: Optional[List[str]] = None,
        probabilities=None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Generate bias explanations with gradient attribution and heuristic fallback."""
        if not getattr(self.tokenizer, 'is_fast', False):
            return self._heuristic_explanations(text, detected_biases or [], reason='Tokenizer does not support offset mappings')

        labels_to_explain = detected_biases or []
        if not labels_to_explain and probabilities is not None:
            sorted_indices = torch.topk(torch.tensor(probabilities), k=min(3, len(probabilities))).indices.tolist()
            labels_to_explain = [BIAS_LABELS[index] for index in sorted_indices]

        explanations: Dict[str, Any] = {
            'method': 'gradient_attribution',
            'labels': {},
        }

        try:
            if not labels_to_explain:
                labels_to_explain = list(BIAS_LABELS[:3])

            for label_name in labels_to_explain:
                label_index = BIAS_LABELS.index(label_name)
                explanations['labels'][label_name] = self._gradient_explanation(text, label_index, top_k=top_k)

            return explanations
        except Exception as exc:
            logger.debug(f"Gradient explainability failed, using heuristic fallback: {exc}")
            return self._heuristic_explanations(text, labels_to_explain, reason=str(exc))
    
    def predict(
        self,
        text: str,
        return_probabilities: bool = True,
        include_explanations: bool = False,
        explanation_top_k: int = 5,
    ) -> Dict:
        """
        Predict bias labels for a single text.
        
        Args:
            text: Article text to analyze
            return_probabilities: Whether to include probability scores
            
        Returns:
            Dictionary with predictions and metadata
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self._forward_logits(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        
        result = self._build_result(probabilities)
        result['analysis_mode'] = 'trained_model'

        if include_explanations:
            try:
                result['explanations'] = self.explain(
                    text,
                    detected_biases=result['detected_biases'],
                    probabilities=probabilities,
                    top_k=explanation_top_k,
                )
            except Exception as exc:
                logger.debug(f"Explainability generation failed, using heuristic fallback: {exc}")
                result['explanations'] = self._heuristic_explanations(text, result['detected_biases'], reason=str(exc))
        
        return result
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 16
    ) -> List[Dict]:
        """
        Predict bias labels for multiple texts.
        
        Args:
            texts: List of article texts
            batch_size: Batch size for inference
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            token_type_ids = encodings.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = self._forward_logits(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )

            probabilities = torch.sigmoid(logits).cpu().numpy()
            
            # Process each result
            for j, probs in enumerate(probabilities):
                result = self._build_result(probs)
                result['analysis_mode'] = 'trained_model'
                results.append(result)
        
        return results
    
    def _generate_summary(self, detected_biases: List[str]) -> str:
        """
        Generate a human-readable summary of detected biases.
        
        Args:
            detected_biases: List of detected bias types
            
        Returns:
            Summary string
        """
        if not detected_biases:
            return "No significant bias detected in this article."
        
        if len(detected_biases) == 1:
            bias_name = BIAS_DESCRIPTIONS[detected_biases[0]]['name']
            return f"{bias_name} detected in this article."
        
        bias_names = [BIAS_DESCRIPTIONS[b]['name'] for b in detected_biases]
        return f"Multiple biases detected: {', '.join(bias_names)}."
    
    def get_bias_explanation(self, bias_type: str) -> Dict:
        """
        Get detailed explanation for a bias type.
        
        Args:
            bias_type: Type of bias
            
        Returns:
            Dictionary with name, description, examples, and indicators
        """
        if bias_type not in BIAS_DESCRIPTIONS:
            raise ValueError(f"Unknown bias type: {bias_type}")
        
        return BIAS_DESCRIPTIONS[bias_type]


def load_predictor(
    model_dir: str = "model/checkpoints",
    device: Optional[str] = None
) -> BiasPredictor:
    """
    Factory function to load the bias predictor.
    
    Args:
        model_dir: Path to model directory
        device: Target device
        
    Returns:
        Initialized BiasPredictor
    """
    return BiasPredictor(model_dir=model_dir, device=device)


if __name__ == "__main__":
    # Test inference
    print("Testing inference module...")
    
    # This will fail without a trained model, but shows usage
    try:
        predictor = load_predictor("model/checkpoints")
        
        test_text = """
        The government's visionary policies have transformed the nation's economy,
        while opposition leaders continue their baseless criticism and disruptive
        protests. The Prime Minister's bold decisions have been praised by experts
        worldwide.
        """
        
        result = predictor.predict(test_text)
        
        print("\nPrediction Result:")
        print(json.dumps(result, indent=2))
        
    except FileNotFoundError as e:
        print(f"\nExpected error (no trained model): {e}")
        print("Train a model first using: python model/src/train.py")
