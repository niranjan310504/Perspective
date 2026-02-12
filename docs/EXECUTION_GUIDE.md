# 📋 Step-by-Step Execution Guide

Complete guide to build and run the Perspective - Indian Media Bias Detection system.

---

## 📌 Prerequisites

| Item | Details |
|------|---------|
| **Python** | 3.9 or higher |
| **Node.js** | 18.x or higher |
| **NewsAPI Key** | Free at [newsapi.org](https://newsapi.org) |
| **Gemini API Key** | Free at [makersuite.google.com](https://makersuite.google.com/app/apikey) |
| **Google Account** | For Colab training |

---

## 🚀 PHASE 1: Project Setup (Local Machine)

### Step 1.1: Create Virtual Environment

```cmd
cd N:\projects\Perspective

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 1.2: Create Environment File

Create a file named `.env` in the project root:

```env
# N:\projects\Perspective\.env

# Gemini API (get from https://makersuite.google.com)
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 📰 PHASE 2: Data Collection

### ⭐⭐ BEST OPTION: Indian News Multi-Source Collector

For a **Ground News clone for India**, you need articles from **diverse political perspectives**.

Our custom collector includes:
- ✅ **15+ Indian news sources** (left, center, right)
- ✅ **Opinion sections** (where bias is strongest)
- ✅ **No API key needed** (uses RSS feeds)
- ✅ **Balanced coverage** (Swarajya, The Wire, NDTV, etc.)
- ✅ **FREE and legal**

### Step 2.1: Collect News Articles (RECOMMENDED)

```cmd
# Make sure you're in the project folder with venv activated
cd N:\projects\Perspective
venv\Scripts\activate

# Install newspaper3k for better article extraction (recommended)
pip install newspaper3k lxml

# Run the Indian News Collector (BEST for Indian context)
python scripts/data_collection/indian_news_collector.py
```

**Expected Output:**
```
�🇳 INDIAN NEWS MULTI-SOURCE COLLECTOR 🇮🇳
Balanced Left/Center/Right Coverage

📰 Swarajya (right)
   RSS feeds: 2
   Collected: 30 articles

📰 The Wire (left)
   RSS feeds: 3
   Collected: 28 articles

📰 The Print (center)
   RSS feeds: 2
   Collected: 25 articles
...

📊 COLLECTION SUMMARY
Total articles: 550

By Political Lean:
  right          180
  center         200
  left           170

✅ Saved to: data/raw/news_articles.csv
```

### Step 2.2: Verify Collected Data

```cmd
python -c "import pandas as pd; df = pd.read_csv('data/raw/news_articles.csv'); print(f'Articles: {len(df)}'); print(df['source'].value_counts().head(10))"
```

### Step 2.3: Check Political Balance

```cmd
python -c "import pandas as pd; df = pd.read_csv('data/raw/news_articles.csv'); print(df['political_lean'].value_counts())"
```

### Alternative Options (if main collector fails)

```cmd
# Option B: GDELT (international database, fewer Indian sources)
python scripts/data_collection/gdelt_collector.py

# Option C: NewsAPI (limited: 100/day, requires API key)
python scripts/data_collection/news_api_collector.py
```

---

## 🏷️ PHASE 3: Data Labeling with Gemini

### Step 3.1: Run Gemini Labeler

```cmd
python scripts/labeling/gemini_labeler.py --input data/raw/news_articles.csv --output data/processed/labeled_articles.csv
```

**Options:**
- `--batch-size 10`: Articles per batch (default: 10)
- `--resume`: Resume from checkpoint if interrupted

**Expected Output:**
```
Perspective - Gemini Bias Labeler
==================================
Loaded 250 articles
Processing batch 1/25...
  Article 1: political, sensationalism
  Article 2: no bias detected
...
Saved checkpoint: 100/250
...
Complete! Saved to: data/processed/labeled_articles.csv
```

**⚠️ Gemini Free Tier:** 60 requests/minute. The script handles rate limiting automatically.

### Step 3.2: Verify Labeled Data

```cmd
python -c "import pandas as pd; df = pd.read_csv('data/processed/labeled_articles.csv'); print(f'Labeled: {len(df)}'); print(df[['headline', 'label_political', 'label_sensationalism']].head())"
```

---

## ✂️ PHASE 4: Split Dataset

### Step 4.1: Clean Data

```cmd
python scripts/preprocessing/clean_data.py --input data/processed/labeled_articles.csv --output data/processed/cleaned_articles.csv
```

### Step 4.2: Split into Train/Val/Test

```cmd
python scripts/preprocessing/split_dataset.py --input data/processed/cleaned_articles.csv --output-dir data/splits
```

**Expected Output:**
```
Dataset Statistics:
  Total samples: 500
  Training: 350 (70%)
  Validation: 75 (15%)
  Test: 75 (15%)

Files created:
  data/splits/train.csv
  data/splits/val.csv
  data/splits/test.csv
```

---

## 🧠 PHASE 5: Train Model on Google Colab

### Step 5.1: Upload to Google Drive

1. Open [Google Drive](https://drive.google.com)
2. Create folder: `Perspective`
3. Upload these files:
   - `data/splits/train.csv`
   - `data/splits/val.csv`

### Step 5.2: Open Colab Notebook

1. Open [Google Colab](https://colab.research.google.com)
2. Create new notebook
3. Copy contents from `notebooks/colab_training.md`
4. Run each cell in order

**Or directly copy this starter code:**

```python
# Cell 1: Mount Drive & Install
!pip install -q transformers torch pandas scikit-learn tqdm

from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Load your data
import pandas as pd
train_df = pd.read_csv('/content/drive/MyDrive/Perspective/train.csv')
val_df = pd.read_csv('/content/drive/MyDrive/Perspective/val.csv')
print(f"Train: {len(train_df)}, Val: {len(val_df)}")
```

### Step 5.3: Run Training

- Training takes ~20-30 minutes on T4 GPU
- Model saves to: `/content/drive/MyDrive/Perspective/model_deploy/`

### Step 5.4: Download Trained Model

After training completes, download from Google Drive:
1. `model_deploy/model.pt`
2. `model_deploy/tokenizer/` (entire folder)
3. `model_deploy/config.json`

### Step 5.5: Copy to Local Project

Copy downloaded files to:
```
N:\projects\Perspective\model\checkpoints\
├── model.pt
├── config.json
└── tokenizer\
    ├── config.json
    ├── tokenizer.json
    └── vocab.txt
```

---

## 🖥️ PHASE 6: Run Backend

### Step 6.1: Start Flask Server

```cmd
cd N:\projects\Perspective
venv\Scripts\activate

cd backend
python run.py
```

**Expected Output:**
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
Loading model...
Model loaded successfully!
```

### Step 6.2: Test API

Open new terminal:
```cmd
curl http://localhost:5000/health
```

Expected: `{"status": "healthy"}`

Test analysis:
```cmd
curl -X POST http://localhost:5000/analyze -H "Content-Type: application/json" -d "{\"text\": \"The visionary PM's bold policies will transform India forever!\"}"
```

---

## 🌐 PHASE 7: Run Frontend

### Step 7.1: Install Node Dependencies

Open new terminal:
```cmd
cd N:\projects\Perspective\frontend

npm install
```

### Step 7.2: Start React App

```cmd
npm start
```

**Expected Output:**
```
Compiled successfully!

Local:            http://localhost:3000
On Your Network:  http://192.168.x.x:3000
```

### Step 7.3: Open in Browser

1. Open http://localhost:3000
2. Paste news article text
3. Click "Analyze"
4. See bias detection results!

---

## ✅ Quick Verification Checklist

| Step | Check | Status |
|------|-------|--------|
| 1 | Virtual environment created | ⬜ |
| 2 | API keys in `.env` file | ⬜ |
| 3 | News articles collected (500+) | ⬜ |
| 4 | Articles labeled with Gemini | ⬜ |
| 5 | Dataset split (train/val/test) | ⬜ |
| 6 | Model trained on Colab | ⬜ |
| 7 | Model files downloaded | ⬜ |
| 8 | Backend running (port 5000) | ⬜ |
| 9 | Frontend running (port 3000) | ⬜ |
| 10 | End-to-end test passed | ⬜ |

---

## 🔧 Troubleshooting

### Issue: "CUDA out of memory" on Colab
**Solution:** Reduce batch size in training config:
```python
CONFIG['batch_size'] = 8  # or even 4
```

### Issue: NewsAPI rate limit
**Solution:** Wait 24 hours or upgrade to paid plan. Collect data over multiple days.

### Issue: Gemini API errors
**Solution:** Check rate limits. Add delay between requests:
```python
time.sleep(1.5)  # 1.5 seconds between requests
```

### Issue: Model not loading in backend
**Solution:** Verify paths in `model/config/model_config.py`:
```python
MODEL_PATH = "model/checkpoints/model.pt"
```

### Issue: Frontend can't connect to backend
**Solution:** 
1. Ensure backend is running on port 5000
2. Check CORS in `backend/app/__init__.py`
3. Verify `frontend/src/services/api.js` has correct URL

---

## 📊 Project Timeline

| Day | Tasks |
|-----|-------|
| Day 1 | Setup, collect ~100 articles |
| Day 2 | Collect more articles (~200) |
| Day 3 | Complete collection, start labeling |
| Day 4 | Complete labeling, clean data |
| Day 5 | Train model on Colab |
| Day 6 | Test backend + frontend |
| Day 7 | Polish, document, prepare demo |

---

## 🎤 Demo Script (For Interview)

1. **Show Architecture** (1 min)
   - Open `docs/ARCHITECTURE.md`
   - Explain flow: Data → LLM Labeling → BERT Training → Flask API → React UI

2. **Show Data Pipeline** (2 min)
   - Show sample of `data/raw/news_articles.csv`
   - Explain Gemini labeling process
   - Show labeled data distribution

3. **Show Model Code** (2 min)
   - Open `model/src/bert_classifier.py`
   - Explain BERT architecture
   - Show multi-label classification head

4. **Live Demo** (3 min)
   - Open frontend at localhost:3000
   - Analyze biased text: "PM's visionary leadership transforms nation!"
   - Analyze neutral text: "Parliament passed the bill with 312 votes"
   - Show different bias types detected

5. **Q&A Ready** (2 min)
   - Reference `docs/INTERVIEW_PREP.md` for common questions

---

**Good luck with your project! 🚀**
