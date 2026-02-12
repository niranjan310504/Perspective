# Perspective - Google Colab Training Notebook

This notebook is designed to train the BERT bias detection model on Google Colab's free GPU (T4 16GB).

## 📋 Instructions

1. **Upload to Google Colab**: Upload this `.ipynb` file to [Google Colab](https://colab.research.google.com)
2. **Enable GPU**: Runtime → Change runtime type → GPU (T4)
3. **Run all cells**: Runtime → Run all

---

## Cell 1: Setup Environment

```python
# Check GPU availability
!nvidia-smi

# Install required packages
!pip install -q transformers datasets torch pandas scikit-learn tqdm

# Mount Google Drive (to save model)
from google.colab import drive
drive.mount('/content/drive')

print("✓ Environment ready!")
```

---

## Cell 2: Upload Training Data

```python
from google.colab import files
import pandas as pd

# Option 1: Upload from local machine
print("Upload your training data (train.csv, val.csv)")
uploaded = files.upload()

# Option 2: Load from Google Drive
# train_df = pd.read_csv('/content/drive/MyDrive/Perspective/data/splits/train.csv')
# val_df = pd.read_csv('/content/drive/MyDrive/Perspective/data/splits/val.csv')

# Check uploaded files
for filename in uploaded.keys():
    df = pd.read_csv(filename)
    print(f"{filename}: {len(df)} rows")
```

---

## Cell 3: Define Dataset Class

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

BIAS_LABELS = [
    "political", "gender", "entity", "racial",
    "religious", "regional", "sensationalism"
]
LABEL_COLUMNS = [f"label_{b}" for b in BIAS_LABELS]

class BiasDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Combine headline and content
        text = f"{row.get('headline', '')} [SEP] {row.get('content', '')}"
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = torch.tensor(
            [row[col] for col in LABEL_COLUMNS],
            dtype=torch.float
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }

print("✓ Dataset class defined")
```

---

## Cell 4: Define Model

```python
import torch.nn as nn
from transformers import BertModel, BertConfig

class BertBiasClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=7, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'probabilities': torch.sigmoid(logits)
        }

print("✓ Model class defined")
```

---

## Cell 5: Training Configuration

```python
# Hyperparameters (optimized for Colab T4)
CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_length': 256,        # Reduced from 512 to save memory
    'batch_size': 16,         # Fits in T4 16GB
    'learning_rate': 2e-5,
    'num_epochs': 3,          # Usually enough for BERT
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'num_labels': 7
}

# Paths
TRAIN_PATH = 'train.csv'  # or your uploaded filename
VAL_PATH = 'val.csv'
SAVE_PATH = '/content/drive/MyDrive/Perspective/model_checkpoint'

print("Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
```

---

## Cell 6: Load Data and Create DataLoaders

```python
import pandas as pd
from transformers import BertTokenizer

# Load data
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])

# Create datasets
train_dataset = BiasDataset(train_df, tokenizer, CONFIG['max_length'])
val_dataset = BiasDataset(val_df, tokenizer, CONFIG['max_length'])

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=2
)

print(f"\n✓ DataLoaders created")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
```

---

## Cell 7: Training Loop

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm
import numpy as np
import os

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model
model = BertBiasClassifier(
    model_name=CONFIG['model_name'],
    num_labels=CONFIG['num_labels']
).to(device)

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)

# Scheduler
total_steps = len(train_loader) * CONFIG['num_epochs']
warmup_steps = int(total_steps * CONFIG['warmup_ratio'])

scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)

# Training history
history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
best_f1 = 0

# Training loop
for epoch in range(CONFIG['num_epochs']):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")
    print('='*50)
    
    # Training
    model.train()
    train_loss = 0
    
    progress = tqdm(train_loader, desc='Training')
    for batch in progress:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        progress.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_train_loss = train_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)
    
    # Validation
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            val_loss += outputs['loss'].item()
            
            probs = outputs['probabilities']
            preds = (probs >= 0.5).int()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    
    # Calculate F1
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    history['val_loss'].append(avg_val_loss)
    history['val_f1'].append(val_f1)
    
    print(f"\n  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f}")
    print(f"  Val F1:     {val_f1:.4f}")
    
    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        os.makedirs(SAVE_PATH, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'config': CONFIG
        }, f'{SAVE_PATH}/best_model.pt')
        print(f"  ★ New best model saved! F1: {best_f1:.4f}")

print(f"\n{'='*50}")
print(f"Training complete! Best F1: {best_f1:.4f}")
print(f"Model saved to: {SAVE_PATH}")
```

---

## Cell 8: Evaluate Per-Label Performance

```python
from sklearn.metrics import classification_report

# Load best model
checkpoint = torch.load(f'{SAVE_PATH}/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get predictions on validation set
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        
        outputs = model(input_ids, attention_mask)
        probs = outputs['probabilities']
        preds = (probs >= 0.5).int()
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

# Print per-label report
print("Per-Label Classification Report:")
print("="*60)
for i, label in enumerate(BIAS_LABELS):
    print(f"\n{label.upper()}:")
    print(classification_report(
        all_labels[:, i], 
        all_preds[:, i], 
        target_names=['No Bias', 'Bias'],
        zero_division=0
    ))
```

---

## Cell 9: Save Final Model for Deployment

```python
# Save model for deployment
DEPLOY_PATH = '/content/drive/MyDrive/Perspective/model_deploy'
os.makedirs(DEPLOY_PATH, exist_ok=True)

# Save model weights
torch.save(model.state_dict(), f'{DEPLOY_PATH}/model.pt')

# Save tokenizer
tokenizer.save_pretrained(f'{DEPLOY_PATH}/tokenizer')

# Save config
import json
with open(f'{DEPLOY_PATH}/config.json', 'w') as f:
    json.dump({
        'model_name': CONFIG['model_name'],
        'num_labels': CONFIG['num_labels'],
        'max_length': CONFIG['max_length'],
        'label_names': BIAS_LABELS,
        'threshold': 0.5
    }, f, indent=2)

print(f"✓ Model saved for deployment at: {DEPLOY_PATH}")
print("\nFiles created:")
print("  - model.pt (model weights)")
print("  - tokenizer/ (BERT tokenizer)")
print("  - config.json (configuration)")
print("\nDownload these files and copy to: model/checkpoints/")
```

---

## Cell 10: Test Inference

```python
# Test the trained model

def predict_bias(text, threshold=0.5):
    model.eval()
    
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=CONFIG['max_length'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = outputs['probabilities'].squeeze().cpu().numpy()
    
    results = {}
    for i, label in enumerate(BIAS_LABELS):
        results[label] = {
            'score': float(probs[i]),
            'detected': probs[i] >= threshold
        }
    
    return results

# Test examples
test_texts = [
    "The government's visionary policies have transformed the nation while opposition continues baseless criticism.",
    "Parliament passed the budget with 312 votes in favor and 245 against. Both parties presented their arguments.",
    "SHOCKING! You won't BELIEVE what this celebrity did! The EXPLOSIVE truth finally revealed!!!"
]

print("Test Predictions:")
print("="*60)

for text in test_texts:
    print(f"\nText: {text[:80]}...")
    result = predict_bias(text)
    
    detected = [k for k, v in result.items() if v['detected']]
    if detected:
        print(f"Detected: {', '.join(detected)}")
    else:
        print("Detected: No bias")
    
    print("Scores:", {k: f"{v['score']:.2f}" for k, v in result.items()})
```

---

## 📥 Download Model

After training, download the model files from Google Drive:
1. `model_deploy/model.pt`
2. `model_deploy/tokenizer/`
3. `model_deploy/config.json`

Copy these to your local project: `model/checkpoints/`
