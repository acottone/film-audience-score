# Filmlytics: Movie Audience Score Prediction System

Predicting movie audience reception using **66,000+ films**, **150+ engineered features**, and an **ensemble of graph-based and tabular ML models**.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=for-the-badge&logo=mongodb&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![HuggingFace](https://img.shields.io/badge/huggingface-%23FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)

---

## TL;DR
- Built an **end-to-end ML system** predicting movie audience scores
- Integrated **TMDB, Rotten Tomatoes, and Youtube** data
- Trained **GNN, KGCN, and XGBoost** models
- Achieved **RMSE = 0.1085** using a stacking ensemble
- Deployed an **interactive Streamlit application**
- Analyzed **representation & gender diversity effects** in film reception

---

## Project Overview

This project develops a **scalable movie audience score prediction system**, combining, structured metadata, graph relationships, sentiment analysis, and engagement metrics.

The system supports:
- Predicting audience scores for **unreleased films**
- Comparing modeling approaches (**graph vs tabular ML**)
- Exploring how **representation and diversity** correlate with audience reception
- Interactive exploration via a production-style web interface

---

## Key Findings

### Model Performance
- **Ensemble RMSE: 0.1085** (best overall)
- **XGBoost** outperforms individual graph models
- **Graph-based models** capture relational structure missed by tabular features
- **80.2%** of predictions fall within +- 10 percentage points

### Feature Insights
- Budget, engagement metrics, and sentiment are dominant predictors
- Gender diversity metrics show **measurable but secondary effects**
- Temporal release patterns (seasonality) influence reception

### Modeling Insight
- Combining **graph structure** with **high-dimensional tabular features** yields consistent performance gains over either approach alone.

---

## Dataset

| Source | Data Type | Volume |
|--------|-----------|--------|
| **TMDB** | Movie metadata, cast, crew, budget | 66,000+ films |
| **Rotten Tomatoes** | Critic/audience scores, reviews | 50,000+ reviews |
| **YouTube** | Trailer views, likes, engagement | 40,000+ trailers |

**Coverage:** 2010-2015

---

## Methodology

The project follows a **modular, end-to-end ML pipeline**, separating data engineering, feature construction, modeling, and deployment.

### 1. Data Collection & Integration
- Pulled structured metadata from **TMDB API**
- Scraped reviews and scores from **Rotten Tomatoes**
- Collected trailer engagement metrics via **YouTube API**
- Matched entities across platforms using fuzzy and ID-based joins

All data sources were merged into a **unified film-level dataset**.

### 2. Feature Engineering (150+ Features)
Feature groups include:
- **Temporal:** release timing, seasonality
- **Budget:** log-budget, budget per minute
- **Cast & Crew:** gender composition, diversity indices
- **Engagement:** views, likes, engagement ratios
- **Sentiment:** BERT-based review sentiment
- **Content:** genre encoding, multi-genre indicators

Missing values were handled using indicator variables and robust defaults.

### 3. Graph Construction
Two graph representations were built:
- **Film similarity graph** (shared genres, cast, crew)
- **Knowledge graph** with heterogeneous node types:
    - Movies, actors, directors, genres, production companies

These graphs enabled relational learning beyond tabular features.

### 4. Modeling
| Model | Purpose |
|--------|-----------|
| **GNN** | Learns film similarity via graph convolution |
| **KGCN** | Captures semantic relations in knowledge graph |
| **XGBoost** | High-performance tabular baseline |
| **Stacking Ensemble** | Combines all models |

The ensemble uses **ridge regression** as a meta-learner.

### 5. Evaluation
- Train/validation/test split by release year
- Metrics: RMSE, MAE, R²
- Calibration checked via prediction error bands

### Results

#### Model Comparison
| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| **Stacking Ensemble** | **0.1085** | 0.0821 | 0.892 |
| XGBoost | 0.110 | 0.0835 | 0.887 |
| KGCN | 0.171 | 0.128 | 0.751 |
| GNN | 0.195 | 0.145 | 0.682 |

**Prediction Accuracy**
- **80.2%** within +- 10 points
- **95.1%** within +- 20 points

### System Architecture
TMDB ──┐
       ├─ Feature Engineering ──┐
RT  ───┤                          ├─ Models ── Ensemble ── Prediction
       ├─ Graph Construction ───┘
YT  ───┘

### Web Application
Built with **Streamlit**, the app supports:
- Movie search & score prediction
- Side-by-side film comparison
- Dataset analytics dashboards
- Feature importance visualization
- Interactive graph exploration

---

##  Reproducibility

### Environment
- Python 3.8+
- PyTorch, XGBoost, Scikit-learn
- MongoDB Atlas
- API Keys:
  - [TMDB API](https://www.themoviedb.org/settings/api) (for data collection)
  - [YouTube Data API](https://console.cloud.google.com/apis/credentials) (for trailer metrics)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd "film audience prediction (python)"
```

2. **Install dependencies**
```bash
# Backend dependencies
cd team15_code_data/backend
pip install -r requirements.txt

# Data pipeline dependencies
cd ../data
pip install pandas numpy requests beautifulsoup4 transformers torch tqdm python-dotenv
```

3. **Set up environment variables**

Create a `.env` file in the `team15_code_data/data` directory:
```bash
TMDB_BEARER_TOKEN=your_tmdb_token_here
YOUTUBE_API_KEY=your_youtube_key_here
```

### Data Pipeline
```
cd team15_code_data/data
python data_pipeline.py --step all
```

### Model Training
```
python xgb_model.py
python kgcn.py
jupyter notebook gnn.ipynb
python ensemble_model.py
```

### Website Deployment
```
cd team15_code_data/backend
streamlit run streamlit_app_clara.py
```

---

## Project Structure
```
film audience prediction (python)/
├── team15_code_data/
│   ├── data/                      # Data collection & preprocessing
│   │   ├── data_pipeline.py       # Complete data pipeline script
│   │   ├── complete_data.csv      # Final merged dataset
│   │   ├── tmdb_data.csv          # TMDB movie metadata
│   │   ├── rt_reviews.json        # Rotten Tomatoes reviews
│   │   └── rt_sentiment.csv       # BERT sentiment analysis
│   │
│   ├── model_training/            # Model training scripts
│   │   ├── gnn/                   # Graph Neural Network
│   │   │   └── gnn.ipynb          # GNN training notebook
│   │   ├── KGCN/                  # Knowledge Graph CNN
│   │   │   └── kgcn.py            # KGCN training script
│   │   ├── xgboost/               # XGBoost models
│   │   │   ├── xgb_model.py       # XGBoost training
│   │   │   └── feature_importance.csv
│   │   └── ensemble_model.py      # Stacking ensemble
│   │
│   ├── backend/                   # Backend services
│   │   ├── streamlit_app_clara.py # Main Streamlit app
│   │   ├── mongodb_setup.py       # Database initialization
│   │   ├── model_artifacts/       # Trained models & scalers
│   │   └── requirements.txt       # Backend dependencies
│   │
│   └── frontend/                  # Frontend (Streamlit UI)
│       └── streamlit_app_clara.py
│
├── Team15_ProjectReport_STA160.pdf
├── Reproducible Deployment Document.pdf
└── README.md
```

---

## Limitations
- Performance degrades for low-budget or niche films
- Rotten Tomatoes scraping is structurally fragile
- Graph models require GPU for efficient training
- Predictions are most reliable for mainstream releases

---

## Technologies Used

### Core
- **Python 3.8+** - Primary language
- **PyTorch** - Deep learning framework (GNN, KGCN)
- **XGBoost** - Gradient boosting
- **Scikit-learn** - ML utilities, ensemble
- **Streamlit** - Web interface
- **MongoDB Atlas** - Database

### Data Processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **BeautifulSoup4** - Web scraping
- **Transformers (HuggingFace)** - BERT sentiment analysis

### Visualization
- **Plotly** - Interactive charts
- **NetworkX** - Graph visualization
- **PyVis** - Network graphs

### APIs
- **TMDB API** - Movie metadata
- **YouTube Data API v3** - Trailer metrics
- **Rotten Tomatoes** - Scores and reviews (scraped)

---

## Author
Developed by a 5-person team for STA 160.

**Angelina Cottone, Matthew Ward, Clara Wei, Dylan Sidhu, Nidhi Deshmukh**
UC Davis, 2025

---
*Last Updated: December 2025*
