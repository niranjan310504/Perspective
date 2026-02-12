# System Architecture

## Overview

Perspective is a 3-tier application for detecting media bias in Indian news articles. The system uses a fine-tuned BERT model for multi-label classification across 7 bias categories.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      React Frontend                              │   │
│  │  • Text/URL Input    • Bias Visualization    • Explanations     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                        │
│                                 ▼                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                           REST API LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Flask Backend                               │   │
│  │  • /api/analyze      • /api/bias-types     • /api/health        │   │
│  │  • Request validation    • Error handling   • CORS              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                        │
│                                 ▼                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                          ML MODEL LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   BERT Classifier                                │   │
│  │  • Tokenization (512 tokens)                                     │   │
│  │  • BERT Encoder (bert-base-uncased)                             │   │
│  │  • Classification Head (7 outputs)                               │   │
│  │  • Sigmoid + Thresholding                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    Raw       │    │   Cleaned    │    │   Labeled    │    │   Split      │
│   Articles   │───▶│   Articles   │───▶│   Articles   │───▶│  Train/Val   │
│   (CSV)      │    │   (CSV)      │    │   (CSV)      │    │   /Test      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
  Web Scraping       clean_data.py       llm_labeler.py      split_dataset.py
  (newspaper3k)      • Remove noise      • GPT-4 prompts     • 70/15/15 split
                     • Deduplication     • Consensus voting  • Stratified
                     • Length filter     • Confidence scores
```

---

## Model Architecture

### BERT for Multi-Label Classification

```
Input Text: "The government's visionary policies have transformed..."
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    BERT Tokenizer                            │
│  • Add [CLS] and [SEP] tokens                               │
│  • Pad/truncate to 512 tokens                               │
│  • Create attention mask                                     │
└─────────────────────────────────────────────────────────────┘
                                    │
                      [CLS] token₁ token₂ ... token₅₁₂ [SEP]
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    BERT Encoder                              │
│  • 12 Transformer layers                                     │
│  • 768 hidden dimensions                                     │
│  • 12 attention heads                                        │
│  • 110M parameters                                           │
└─────────────────────────────────────────────────────────────┘
                                    │
                     [CLS] embedding (768-dim)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                 Classification Head                          │
│  • Dropout (p=0.3)                                          │
│  • Linear layer: 768 → 7                                    │
│  • Sigmoid activation                                        │
└─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Probabilities                      │
│  Political:     0.89  ──▶  ✓ Detected (> 0.5)               │
│  Gender:        0.12                                         │
│  Entity:        0.67  ──▶  ✓ Detected (> 0.5)               │
│  Racial:        0.08                                         │
│  Religious:     0.15                                         │
│  Regional:      0.11                                         │
│  Sensationalism: 0.72  ──▶  ✓ Detected (> 0.5)              │
└─────────────────────────────────────────────────────────────┘
```

---

## Training Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                                │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  for epoch in range(4):                                            │
│      ┌─────────────────────────────────────────────────────────┐   │
│      │  Training Phase                                          │   │
│      │  • Load batches (batch_size=16)                         │   │
│      │  • Forward pass through BERT                            │   │
│      │  • Calculate BCEWithLogitsLoss                          │   │
│      │  • Backward pass                                         │   │
│      │  • Gradient clipping (max_norm=1.0)                     │   │
│      │  • Optimizer step (AdamW, lr=2e-5)                      │   │
│      │  • Learning rate scheduler                               │   │
│      └─────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│      ┌─────────────────────────────────────────────────────────┐   │
│      │  Validation Phase                                        │   │
│      │  • Evaluate on validation set                           │   │
│      │  • Calculate F1, Precision, Recall per label            │   │
│      │  • Save best model if F1 improves                       │   │
│      └─────────────────────────────────────────────────────────┘   │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## LLM-Assisted Labeling

### Why LLM Labeling?

1. **Scale**: Manual labeling is time-consuming
2. **Consistency**: LLMs provide consistent label application
3. **Cost**: More affordable than large-scale human annotation

### Safeguards Against LLM Bias

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LABELING SAFEGUARDS                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. CONSENSUS VOTING                                                 │
│     • Run 3 LLM calls per article                                   │
│     • Majority vote determines final label                          │
│     • Reduces single-call errors                                    │
│                                                                      │
│  2. CONFIDENCE THRESHOLDING                                         │
│     • LLM provides confidence score (0-1)                           │
│     • Only include high-confidence labels (≥0.7) in training        │
│     • Low-confidence articles flagged for human review              │
│                                                                      │
│  3. DIVERSE FEW-SHOT EXAMPLES                                       │
│     • Include positive AND negative examples for each bias          │
│     • Examples from different political perspectives                │
│     • Prevents anchoring on one viewpoint                           │
│                                                                      │
│  4. HUMAN VERIFICATION                                              │
│     • Random sample (10%) manually verified                         │
│     • Edge cases reviewed by team                                   │
│     • Calibration of thresholds                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Request Flow

```
User enters text
        │
        ▼
┌───────────────────┐
│   React Frontend  │
│   (localhost:3000)│
└─────────┬─────────┘
          │ POST /api/analyze
          │ {text: "..."}
          ▼
┌───────────────────┐
│   Flask Backend   │
│   (localhost:5000)│
│                   │
│  1. Validate input│
│  2. Preprocess    │
│  3. Load model    │◀─── Model loaded once
│     (lazy load)   │     at first request
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   BERT Tokenizer  │
│  • Tokenize text  │
│  • Create tensors │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   BERT Encoder    │
│  • Forward pass   │
│  • Get embeddings │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Classification    │
│  • 7 probabilities│
│  • Apply threshold│
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   JSON Response   │
│  • biases{}       │
│  • detected_biases│
│  • summary        │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   React Frontend  │
│  • Display bars   │
│  • Show summary   │
└───────────────────┘
```

---

## Deployment Architecture (Future)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION SETUP                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │   Nginx      │────▶│   Gunicorn   │────▶│   Flask      │        │
│  │   (Reverse   │     │   (WSGI)     │     │   Workers    │        │
│  │    Proxy)    │     │              │     │   (x4)       │        │
│  └──────────────┘     └──────────────┘     └──────────────┘        │
│         │                                          │                 │
│         ▼                                          ▼                 │
│  ┌──────────────┐                         ┌──────────────┐          │
│  │   React      │                         │   Model      │          │
│  │   Static     │                         │   (GPU/CPU)  │          │
│  │   Files      │                         │              │          │
│  └──────────────┘                         └──────────────┘          │
│                                                                      │
│  Optional:                                                          │
│  • Redis for caching                                                │
│  • PostgreSQL for storing analysis history                          │
│  • Docker containerization                                          │
│  • Kubernetes for scaling                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
Perspective/
├── data/
│   ├── schema.py              # Dataset schema definitions
│   ├── raw/                   # Raw scraped articles
│   ├── processed/             # Cleaned and labeled data
│   └── splits/                # Train/val/test CSVs
│
├── model/
│   ├── config/
│   │   └── model_config.py    # Hyperparameters
│   ├── src/
│   │   ├── bert_classifier.py # Model architecture
│   │   ├── dataset.py         # PyTorch Dataset
│   │   ├── train.py           # Training script
│   │   └── inference.py       # Inference module
│   └── checkpoints/           # Saved models
│
├── backend/
│   ├── app/
│   │   ├── __init__.py        # Flask app factory
│   │   ├── config.py          # Flask config
│   │   └── routes.py          # API endpoints
│   └── run.py                 # Entry point
│
├── frontend/
│   ├── public/
│   │   └── index.html         # HTML template
│   └── src/
│       ├── App.js             # Main component
│       ├── components/        # React components
│       └── services/          # API client
│
├── scripts/
│   ├── preprocessing/
│   │   ├── clean_data.py      # Data cleaning
│   │   └── split_dataset.py   # Train/val/test split
│   └── labeling/
│       └── llm_labeler.py     # LLM-based labeling
│
├── docs/
│   ├── API.md                 # API documentation
│   ├── ARCHITECTURE.md        # This file
│   └── INTERVIEW_PREP.md      # Viva preparation
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview
```
