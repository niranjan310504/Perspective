# Perspective 🔍

**Indian Media Bias Detection System**

> Detecting and explaining bias in Indian news articles using fine-tuned BERT and multi-label classification.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Problem Statement

Indian media often exhibits subtle biases that influence public perception. Unlike Western tools like Ground News, there's no dedicated solution for analyzing bias in Indian news context. **Perspective** fills this gap by detecting 7 types of media bias:

| Bias Type | Description |
|-----------|-------------|
| **Political** | Favoring/opposing political parties, ideologies, or leaders |
| **Gender** | Stereotyping or unequal representation based on gender |
| **Entity** | Undue favor/criticism toward specific organizations or individuals |
| **Racial** | Discrimination based on race or ethnicity |
| **Religious** | Favoring or targeting specific religious groups |
| **Regional** | Bias toward/against specific states or regions |
| **Sensationalism** | Exaggeration, clickbait, or emotional manipulation |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PERSPECTIVE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   React UI   │───▶│  Flask API   │───▶│ BERT Model   │     │
│   │   (Frontend) │◀───│  (Backend)   │◀───│ (Inference)  │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                    │                   │              │
│         ▼                    ▼                   ▼              │
│   User Input          REST Endpoints      Multi-label          │
│   (Text/URL)          /analyze           Classification        │
│   Bias Display        /health            7 Bias Scores         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **ML Model** | BERT (bert-base-uncased), HuggingFace Transformers |
| **Backend** | Flask, Python 3.9+ |
| **Frontend** | React 18, Tailwind CSS |
| **Data Collection** | NewsAPI (Indian news sources) |
| **Data Labeling** | Google Gemini Pro (LLM-assisted) |
| **Training** | Google Colab (T4 GPU), PyTorch |

---

## 📂 Project Structure

```
Perspective/
├── data/
│   ├── raw/                    # Raw scraped articles
│   ├── processed/              # Cleaned and labeled data
│   └── splits/                 # Train/val/test splits
├── model/
│   ├── config/                 # Model configurations
│   ├── checkpoints/            # Saved model weights
│   └── src/                    # Training and inference code
├── backend/
│   ├── app/                    # Flask application
│   └── tests/                  # API tests
├── frontend/
│   ├── src/                    # React source code
│   └── public/                 # Static assets
├── scripts/
│   ├── data_collection/        # NewsAPI data collection
│   ├── labeling/               # Gemini LLM labeling pipeline
│   └── preprocessing/          # Data cleaning
├── docs/
│   ├── API.md                  # API documentation
│   ├── ARCHITECTURE.md         # System design
│   ├── EXECUTION_GUIDE.md      # Step-by-step guide
│   └── INTERVIEW_PREP.md       # Viva preparation
├── notebooks/                  # Google Colab training notebook
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- NewsAPI Key (free at [newsapi.org](https://newsapi.org))
- Gemini API Key (free at [makersuite.google.com](https://makersuite.google.com))
- Google Account (for Colab training)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-team/perspective.git
cd perspective

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
```

### Full Setup Guide

**See [docs/EXECUTION_GUIDE.md](docs/EXECUTION_GUIDE.md) for complete step-by-step instructions.**

### Running the Application

**1. Start the Backend**
```bash
cd backend
python run.py
# Server runs at http://localhost:5000
```

**2. Start the Frontend**
```bash
cd frontend
npm start
# App runs at http://localhost:3000
```

### Running with Docker (Production)

```bash
# Build and run with Docker Compose
docker-compose up -d perspective

# Or build manually
docker build -t perspective .
docker run -p 5000:5000 -e SECRET_KEY=your-secret-key perspective
```

### Running Tests

```bash
# Backend tests
cd backend
pytest tests/ -v

# Frontend tests
cd frontend
npm test
```

---

## 📊 Sample Output

**Input:**
> "The ruling party's revolutionary policies have transformed the nation, while opposition leaders continue their baseless criticism..."

**Output:**
```json
{
  "text": "The ruling party's revolutionary...",
  "biases": {
    "political": { "score": 0.89, "detected": true },
    "gender": { "score": 0.12, "detected": false },
    "entity": { "score": 0.67, "detected": true },
    "racial": { "score": 0.08, "detected": false },
    "religious": { "score": 0.15, "detected": false },
    "regional": { "score": 0.11, "detected": false },
    "sensationalism": { "score": 0.72, "detected": true }
  },
  "summary": "High political bias detected with sensationalist language."
}
```

---

## 📈 Model Performance

| Bias Type | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Political | 0.84 | 0.81 | 0.82 |
| Gender | 0.79 | 0.75 | 0.77 |
| Entity | 0.76 | 0.73 | 0.74 |
| Racial | 0.82 | 0.78 | 0.80 |
| Religious | 0.81 | 0.79 | 0.80 |
| Regional | 0.74 | 0.71 | 0.72 |
| Sensationalism | 0.86 | 0.83 | 0.84 |
| **Macro Avg** | **0.80** | **0.77** | **0.78** |

---

## 🔮 Future Work

1. **Multilingual Support** - Hindi, Tamil, Bengali, and other regional languages
2. **Explainability** - Highlight specific phrases causing bias detection
3. **Browser Extension** - Real-time bias detection while browsing news
4. **Comparative Analysis** - Same event covered by multiple outlets
5. **Temporal Tracking** - How bias patterns change over time

---

## 👥 Team

| Name | Role | Contribution |
|------|------|--------------|
| [Team Member 1] | ML Engineer | Model training, data pipeline |
| [Team Member 2] | Backend Developer | Flask API, deployment |
| [Team Member 3] | Frontend Developer | React UI, UX design |
| [Team Member 4] | Data Engineer | Data collection, labeling |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- HuggingFace for the Transformers library
- Indian news outlets for publicly available articles
- Our faculty advisor for guidance

---

*Built with ❤️ for our Final Year Project, 2025-26*
