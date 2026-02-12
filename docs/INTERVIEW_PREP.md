# Interview & Viva Preparation

## Quick Project Summary

**Perspective** is a media bias detection system for Indian news articles. It uses a fine-tuned BERT model to classify articles across 7 bias types: Political, Gender, Entity, Racial, Religious, Regional, and Sensationalism.

---

## Common Interview Questions

### 1. "Why did you choose this project?"

**Strong Answer:**
> "We noticed that while tools like Ground News exist for Western media, there's no dedicated solution for Indian news context. Indian media has unique characteristics - complex political landscape, diverse religious and regional identities, and specific linguistic patterns. We wanted to build something that addresses these nuances.
>
> From a technical standpoint, this project combines NLP, deep learning, and full-stack development - giving us hands-on experience with a complete ML pipeline from data collection to deployment."

**Key points to mention:**
- Gap in existing solutions for Indian context
- Real-world relevance (media literacy, journalism)
- Technical depth (NLP, deep learning, full-stack)

---

### 2. "What makes your project different from existing bias detection tools?"

**Strong Answer:**
> "Three main differentiators:
>
> 1. **India-specific**: We focus on biases relevant to Indian context - religious bias, regional bias, caste-based stereotypes - which Western tools don't capture.
>
> 2. **Multi-label approach**: Unlike tools that give a single 'left/right' score, we detect 7 independent bias types. An article can be both politically biased AND sensationalist.
>
> 3. **LLM-assisted labeling**: We used GPT-4 with few-shot prompts and consensus voting to create our training data, which is more scalable than pure manual labeling while maintaining quality through verification."

---

### 3. "Why multi-label classification instead of multi-class?"

**Strong Answer:**
> "In reality, biases don't occur in isolation. A news article can simultaneously have political bias AND religious undertones AND sensationalist language. Multi-class classification forces us to pick ONE label, which doesn't reflect reality.
>
> With multi-label classification using sigmoid activations, each bias type is an independent binary decision. We use Binary Cross-Entropy loss instead of Cross-Entropy, and apply a threshold (0.5) to each output independently.
>
> This gives us richer, more actionable insights. A user can see 'this article has high political bias and moderate sensationalism' rather than just 'this article is biased.'"

---

### 4. "Explain your model architecture."

**Strong Answer:**
> "We use BERT (bert-base-uncased) as our base model:
>
> 1. **Tokenization**: Articles are tokenized using WordPiece, truncated/padded to 512 tokens. We add [CLS] and [SEP] tokens.
>
> 2. **BERT Encoder**: 12 transformer layers with 768 hidden dimensions and 12 attention heads. ~110M parameters.
>
> 3. **Classification Head**: We take the [CLS] token embedding (which represents the whole sequence), apply dropout (0.3) for regularization, then a linear layer mapping 768 dimensions to 7 outputs.
>
> 4. **Output**: Sigmoid activation gives us probabilities [0, 1] for each bias type. We apply a threshold of 0.5 to make binary predictions.
>
> Loss function is BCEWithLogitsLoss, which combines sigmoid and binary cross-entropy for numerical stability."

---

### 5. "Why LLM-assisted labeling? Isn't that cheating?"

**Strong Answer:**
> "Great question! We're not using LLMs for inference - we're using them only for labeling training data. There are several reasons:
>
> 1. **Scale**: Manual labeling 5000+ articles with 7 labels each would take months. LLM-assisted labeling lets us scale efficiently.
>
> 2. **Consistency**: Human annotators have varying interpretations. LLMs with well-crafted prompts provide consistent label application.
>
> 3. **Cost**: Professional annotation is expensive. LLM API calls are more affordable.
>
> But we added safeguards:
> - **Consensus voting**: 3 LLM calls per article, majority vote
> - **Confidence filtering**: Only high-confidence labels (≥0.7) used for training
> - **Human verification**: 10% random sample manually verified
> - **Diverse few-shot examples**: Prevents anchoring on one viewpoint
>
> The final model is still a fine-tuned BERT that runs independently - no LLM needed for inference, making it fast and cost-free to run."

---

### 6. "What loss function did you use and why?"

**Strong Answer:**
> "We use BCEWithLogitsLoss (Binary Cross-Entropy with Logits).
>
> For multi-label classification, each output is an independent binary classification problem. The loss is:
>
> $L = -\frac{1}{N \times K} \sum_{i=1}^{N} \sum_{j=1}^{K} [y_{ij} \log(\sigma(x_{ij})) + (1-y_{ij}) \log(1-\sigma(x_{ij}))]$
>
> Where:
> - N = batch size
> - K = 7 labels
> - $y_{ij}$ = ground truth (0 or 1)
> - $x_{ij}$ = raw logit from model
> - $\sigma$ = sigmoid function
>
> BCEWithLogitsLoss combines sigmoid + BCE in one operation, which is numerically more stable than applying sigmoid first."

---

### 7. "How did you handle class imbalance?"

**Strong Answer:**
> "Class imbalance is a real concern - not every article has every type of bias. We addressed it in three ways:
>
> 1. **pos_weight in BCEWithLogitsLoss**: We calculated the ratio of negative to positive samples for each label and used it as pos_weight. This gives more weight to positive examples.
>
> 2. **Threshold tuning**: Instead of using 0.5 for all labels, we can optimize thresholds per label based on validation F1 scores.
>
> 3. **Evaluation metrics**: We focus on F1-score (harmonic mean of precision and recall) rather than accuracy, as accuracy can be misleading with imbalanced data."

---

### 8. "Explain your training process."

**Strong Answer:**
> "Standard fine-tuning setup:
>
> 1. **Optimizer**: AdamW with learning rate 2e-5 and weight decay 0.01
>
> 2. **Learning rate schedule**: Linear warmup for first 10% of steps, then linear decay
>
> 3. **Epochs**: 4 epochs (typical for BERT fine-tuning - more can cause overfitting)
>
> 4. **Batch size**: 16 (limited by GPU memory with 512 token sequences)
>
> 5. **Gradient clipping**: max_norm=1.0 to prevent exploding gradients
>
> 6. **Validation**: After each epoch, evaluate on validation set. Save best model based on macro F1 score.
>
> We split data 70/15/15 for train/validation/test with stratified sampling to maintain label distribution."

---

### 9. "What metrics do you use for evaluation?"

**Strong Answer:**
> "For multi-label classification, we report:
>
> 1. **Per-label metrics**: Precision, Recall, F1 for each of the 7 bias types
>
> 2. **Macro-averaged**: Average of per-label metrics (treats all labels equally)
>
> 3. **Micro-averaged**: Aggregate TP/FP/FN across all labels (gives more weight to common labels)
>
> We prioritize **macro F1** as our primary metric because:
> - It balances precision and recall
> - Macro averaging ensures we care about performance on rare bias types too
> - It's a single number for model comparison
>
> We also examine the **confusion matrix per label** to understand false positive vs false negative tradeoffs."

---

### 10. "What are the limitations of your project?"

**Strong Answer:**
> "Being honest about limitations:
>
> 1. **Language**: Currently English only. Indian news is heavily multilingual (Hindi, Tamil, Bengali, etc.). Extending to other languages requires multilingual models and language-specific training data.
>
> 2. **Explainability**: We output probabilities but don't highlight *which phrases* caused the bias detection. Adding attention visualization or LIME/SHAP explanations would help.
>
> 3. **LLM labeling bias**: Despite safeguards, LLMs have their own biases. Our model inherits whatever biases exist in the training labels. More human verification would improve this.
>
> 4. **Temporal context**: News context changes over time. A statement that's neutral today might be biased in a different political context. Our model doesn't capture this.
>
> 5. **Satire and opinion pieces**: The model is trained on news articles. It may not perform well on editorials, satire, or opinion pieces which have different conventions."

---

### 11. "What's your future scope?"

**Strong Answer:**
> "Several directions:
>
> 1. **Multilingual support**: Fine-tune multilingual BERT (mBERT) or IndicBERT for Hindi and regional languages.
>
> 2. **Explainability**: Add attention visualization to highlight specific phrases contributing to bias detection. Integrate LIME or SHAP for interpretability.
>
> 3. **Browser extension**: Build a Chrome extension for real-time bias detection while browsing news.
>
> 4. **Comparative analysis**: Show how different outlets cover the same story, like Ground News does.
>
> 5. **Temporal tracking**: Monitor how a source's bias patterns change over time.
>
> 6. **Fine-grained political bias**: Instead of binary 'political bias', detect left vs right vs centrist positioning."

---

### 12. "Walk me through a complete request flow."

**Strong Answer:**
> "Let me trace a request from frontend to model:
>
> 1. **User inputs text** in React frontend, clicks 'Analyze'
>
> 2. **Frontend sends POST request** to `http://localhost:5000/api/analyze` with JSON body `{text: "..."}`
>
> 3. **Flask receives request**, validates input (minimum length, maximum length)
>
> 4. **Model is loaded lazily** - first request loads model into memory, subsequent requests reuse it
>
> 5. **Tokenization**: BERT tokenizer converts text to token IDs, adds special tokens, creates attention mask
>
> 6. **Forward pass**: Token tensors go through BERT encoder, get [CLS] embedding
>
> 7. **Classification head**: [CLS] embedding → dropout → linear → 7 logits
>
> 8. **Sigmoid + threshold**: Convert logits to probabilities, apply 0.5 threshold
>
> 9. **Flask constructs JSON response** with biases, scores, detected_biases, summary
>
> 10. **React displays results** with colored bars and summary card"

---

## Technical Deep-Dive Questions

### "Why BERT and not GPT or other models?"

> "BERT is designed for understanding/classification tasks (encoder-only), while GPT is for generation (decoder-only). For classification, we need rich bidirectional representations, which BERT provides.
>
> Also, BERT is smaller and faster for inference compared to GPT-4. We don't need generation capabilities for classification."

### "Why bert-base-uncased specifically?"

> "We chose uncased because:
> - Indian news often has inconsistent capitalization
> - Bias patterns don't depend on case (SHOCKING vs shocking)
> - Smaller vocabulary, faster tokenization
>
> We chose base (vs large) because:
> - Faster inference (important for real-time API)
> - Less prone to overfitting with our dataset size
> - Can run on consumer hardware"

### "How do you handle long articles?"

> "BERT has a 512 token limit. For longer articles, we:
> - Truncate to first 512 tokens (contains headline + lead which are often most biased)
> - Alternative: Sliding window with aggregation (we can implement this as future work)"

---

## Questions to Ask the Panel

1. "What aspects of our implementation would you suggest improving for production deployment?"

2. "Are there specific Indian news sources you'd recommend for expanding our dataset?"

3. "What's your perspective on using LLMs for training data labeling vs. fully manual annotation?"

---

## Quick Stats to Remember

| Metric | Value |
|--------|-------|
| Model | bert-base-uncased |
| Parameters | ~110M |
| Max tokens | 512 |
| Bias types | 7 |
| Hidden size | 768 |
| Attention heads | 12 |
| Transformer layers | 12 |
| Learning rate | 2e-5 |
| Epochs | 4 |
| Batch size | 16 |
| Threshold | 0.5 |

---

## Confidence Boosters

- You built a complete end-to-end ML system
- You understand the architecture deeply
- You made practical engineering decisions
- You know the limitations and can discuss them honestly
- You have clear ideas for future work

**Good luck with your viva!** 🎓
