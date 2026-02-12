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
from typing import Dict, List, Optional, Union
import torch
from transformers import BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification

# Setup logging
logger = logging.getLogger(__name__)

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.schema import BIAS_LABELS, BIAS_DESCRIPTIONS


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
    
    def predict(
        self,
        text: str,
        return_probabilities: bool = True
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
        
        # Inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Handle both HuggingFace and custom model outputs
        if hasattr(outputs, 'logits'):
            # HuggingFace format
            logits = outputs.logits
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        elif isinstance(outputs, dict) and 'probabilities' in outputs:
            # Custom model format
            probabilities = outputs['probabilities'].squeeze(0).cpu().numpy()
        else:
            # Fallback
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        
        predictions = (probabilities >= self.threshold).astype(int)
        
        # Format results
        result = {
            'biases': {},
            'detected_biases': [],
            'summary': ''
        }
        
        for i, label_name in enumerate(BIAS_LABELS):
            result['biases'][label_name] = {
                'detected': bool(predictions[i]),
                'score': float(probabilities[i])
            }
            
            if predictions[i]:
                result['detected_biases'].append(label_name)
        
        # Generate summary
        result['summary'] = self._generate_summary(result['detected_biases'])
        
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
            
            # Inference
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Handle both HuggingFace and custom model outputs
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                probabilities = torch.sigmoid(logits).cpu().numpy()
            elif isinstance(outputs, dict) and 'probabilities' in outputs:
                probabilities = outputs['probabilities'].cpu().numpy()
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                probabilities = torch.sigmoid(logits).cpu().numpy()
            
            # Process each result
            for j, probs in enumerate(probabilities):
                predictions = (probs >= self.threshold).astype(int)
                
                result = {
                    'biases': {},
                    'detected_biases': [],
                    'summary': ''
                }
                
                for k, label_name in enumerate(BIAS_LABELS):
                    result['biases'][label_name] = {
                        'detected': bool(predictions[k]),
                        'score': float(probs[k])
                    }
                    
                    if predictions[k]:
                        result['detected_biases'].append(label_name)
                
                result['summary'] = self._generate_summary(result['detected_biases'])
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
