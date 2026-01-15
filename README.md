# Filmlytics: Movie Audience Score Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Filmlytics** is an advanced machine learning system that predicts movie audience scores before release by combining multiple data sources and state-of-the-art models. Built for studios, marketers, and film researchers to make data-driven decisions.

Team: Angelina Cottone, Matthew Ward, Clara Wei, Dylan Sidhu, Nidhi Deshmukh

---

## Project Overview

### What It Does
Filmlytics predicts how audiences will rate films **before release** using:
- **66,000+ films** from 2010-2025
- **150+ engineered features** from multiple data sources
- **Ensemble of 3 ML models** (GNN, KGCN, XGBoost)
- **Interactive web interface** for predictions and analytics

### Why It Matters
- **Studios**: Guide marketing strategy and financial forecasting
- **Researchers**: Understand what film characteristics resonate with audiences
- **Industry**: Analyze representation factors (gender diversity, cast composition)

### Performance Metrics
- **Ensemble RMSE**: 0.1085 (best overall)
- **XGBoost RMSE**: 0.110
- **KGCN RMSE**: 0.171
- **GNN RMSE**: 0.195
- **80.2%** of predictions within ±10 percentage points

---

## Architecture

### Data Pipeline
```
TMDB API → Rotten Tomatoes → YouTube API
    ↓           ↓                ↓
  Metadata   Reviews/Scores   Trailer Metrics
    ↓           ↓                ↓
        Feature Engineering
                ↓
    Unified Dataset (66K films)
```

### Model Ensemble
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│     GNN     │  │    KGCN     │  │   XGBoost   │
│  (Graph)    │  │ (Knowledge  │  │  (Tabular)  │
│             │  │   Graph)    │  │             │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        ↓
              Stacking Meta-Learner
                        ↓
           Audience Score Prediction
```

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- MongoDB Atlas account (free tier works)
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

## Usage

### 1. Data Collection (Optional - Pre-collected data included)

Run the complete data pipeline:
```bash
cd team15_code_data/data
python data_pipeline.py --step all
```

Run specific steps:
```bash
# Step 1: Collect TMDB data
python data_pipeline.py --step 1

# Steps 1-3: TMDB + cleaning + RT mapping
python data_pipeline.py --step 1,2,3

# Skip API calls (use cached data)
python data_pipeline.py --step all --skip-api
```

### 2. Set Up MongoDB

```bash
cd team15_code_data/backend
python mongodb_setup.py
```

Update `CONNECTION_STRING` and `CSV_PATH` in `mongodb_setup.py` before running.

### 3. Train Models (Optional - Pre-trained models included)

**XGBoost:**
```bash
cd team15_code_data/model_training/xgboost
python xgb_modelMATT.py
```

**KGCN:**
```bash
cd team15_code_data/model_training/KGCN
python kgcn.py
```

**GNN:**
```bash
cd team15_code_data/model_training/gnn
jupyter notebook gnn.ipynb
```

**Ensemble:**
```bash
cd team15_code_data/model_training
python ensemble_model.py
```

### 4. Run the Web Application

```bash
cd team15_code_data/backend
streamlit run streamlit_app_clara.py
```

The app will open at `http://localhost:8501`

---

## Features

### Web Application Pages

1. **Home** - Project overview and methodology
2. **Movie Search** - Search movies and get predictions
3. **Compare Movies** - Side-by-side movie comparison
4. **Analytics Dashboard** - Dataset insights and trends
5. **Modeling** - Model performance and feature importance
6. **Visual Graph Explorer** - Interactive film relationship graph
7. **Acknowledgements** - Data sources and credits

### Key Capabilities

- Predict audience scores for unreleased films
- Analyze 150+ features (budget, cast, genres, sentiment, diversity)
- Visualize film similarity networks
- Compare model predictions (GNN vs KGCN vs XGBoost)
- Explore gender diversity impact on audience reception
- Interactive charts and analytics

---

## Data Sources

| Source | Data Type | Volume |
|--------|-----------|--------|
| **TMDB** | Movie metadata, cast, crew, budget | 66,000+ films |
| **Rotten Tomatoes** | Critic/audience scores, reviews | 50,000+ reviews |
| **YouTube** | Trailer views, likes, engagement | 40,000+ trailers |

---

## Models

### 1. Graph Neural Network (GNN)
- **Purpose**: Captures film similarity through shared attributes
- **Architecture**: Multi-layer graph convolution
- **Features**: Genre, cast, director, production company relationships
- **RMSE**: 0.195

### 2. Knowledge Graph Convolutional Network (KGCN)
- **Purpose**: Learns semantic relationships in film knowledge graph
- **Architecture**: Heterogeneous graph with multiple node types
- **Node Types**: Movies, genres, directors, companies, cast, countries
- **Features**: 16 numeric features + categorical embeddings
- **Training**: 20 epochs with early stopping, gradient clipping
- **RMSE**: 0.171

### 3. XGBoost
- **Purpose**: Feature-based tabular prediction
- **Features**: 150+ engineered features including:
  - Temporal (release year, month, quarter, seasonality)
  - Budget (log-transformed, budget-per-minute)
  - Cast/Crew (diversity metrics, star power)
  - Engagement (YouTube views, likes, comments)
  - Sentiment (BERT-based review sentiment)
  - Representation (gender balance, female director)
- **RMSE**: 0.110 (best individual model)

### 4. Stacking Ensemble
- **Meta-Learner**: Ridge regression
- **Weights**: Optimized via grid search
- **RMSE**: 0.1085 (best overall)
- **Improvement**: 0.0015 over best individual model

---

## Feature Engineering

### Categories (150+ features total)

**Temporal Features**
- Release year, month, quarter
- Summer/holiday release indicators
- Days since epoch

**Budget & Revenue**
- Log-transformed budget
- Budget per minute
- Has budget indicator

**Cast & Crew**
- Female/male cast counts and percentages
- Gender balance score
- Female director indicator
- Cast diversity metrics
- Star power indicators

**Engagement Metrics**
- YouTube trailer views, likes, comments
- View-to-like ratio
- Engagement rate
- Days between trailer and release

**Sentiment Analysis**
- BERT sentiment scores on reviews
- Description sentiment (overview text)
- Positive/negative review ratios

**Genre & Content**
- One-hot encoded top genres
- Genre count
- Multi-genre indicators

**Representation**
- Female cast percentage
- Gender balance score
- Director gender
- Cast gender distribution

---

## Results & Insights

### Model Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| **Stacking Ensemble** | **0.1085** | 0.0821 | 0.892 |
| XGBoost | 0.110 | 0.0835 | 0.887 |
| KGCN | 0.171 | 0.128 | 0.751 |
| GNN | 0.195 | 0.145 | 0.682 |

### Key Findings

1. **Ensemble Advantage**: Combining graph-based and tabular models yields 1.4% improvement
2. **Feature Importance**: Budget, critic scores, and engagement metrics are top predictors
3. **Representation Matters**: Gender diversity shows measurable impact on audience scores
4. **Temporal Patterns**: Summer and holiday releases show different audience behavior
5. **Graph Benefits**: Film relationships capture nuances missed by tabular features

### Prediction Accuracy
- **80.2%** of predictions within ±10 percentage points
- **95.1%** within ±20 percentage points
- Mean absolute error: **8.21 percentage points**

---

## Technical Stack

### Core Technologies
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

## Configuration

### MongoDB Setup

1. Create a MongoDB Atlas cluster (free tier)
2. Whitelist your IP address
3. Get connection string from Atlas dashboard
4. Update `MONGODB_URI` in:
   - `team15_code_data/backend/streamlit_app_clara.py`
   - `team15_code_data/backend/mongodb_setup.py`
   - `team15_code_data/model_training/ensemble_model.py`

### Streamlit Secrets (for deployment)

Create `.streamlit/secrets.toml`:
```toml
MONGODB_URI = "mongodb+srv://username:password@cluster.mongodb.net/?appName=filmlytics"
```

### API Keys

For data collection, set environment variables:
```bash
export TMDB_BEARER_TOKEN="your_token"
export YOUTUBE_API_KEY="your_key"
```

Or create `.env` file in `team15_code_data/data/`:
```
TMDB_BEARER_TOKEN=your_token
YOUTUBE_API_KEY=your_key
```

---

## Data Pipeline Details

### Pipeline Steps

1. **TMDB Collection** - Fetch movie metadata (2010-2025)
2. **Data Cleaning** - Filter invalid entries, handle missing values
3. **RT URL Mapping** - Match TMDB movies to Rotten Tomatoes URLs
4. **RT Scraping** - Collect reviews and scores
5. **Sentiment Analysis** - BERT-based sentiment on reviews
6. **Gender Diversity** - Extract cast/crew gender via TMDB API
7. **YouTube Metrics** - Fetch trailer engagement data
8. **Final Merge** - Combine all sources into unified dataset

### Running Individual Steps

```bash
# Full pipeline
python data_pipeline.py --step all

# Individual steps
python data_pipeline.py --step 1  # TMDB collection
python data_pipeline.py --step 2  # Cleaning
python data_pipeline.py --step 3  # RT mapping
python data_pipeline.py --step 4  # RT scraping
python data_pipeline.py --step 5  # Sentiment analysis
python data_pipeline.py --step 6  # Gender diversity
python data_pipeline.py --step 7  # YouTube metrics
python data_pipeline.py --step 8  # Final merge

# Multiple steps
python data_pipeline.py --step 1,2,3

# Skip API calls (use cached data)
python data_pipeline.py --step all --skip-api
```

---

## Academic Context

**Course**: STA 160 - Statistical Data Science
**Institution**: UC Davis
**Team**: Team 15 (Cinemaniacs)
**Date**: December 2025

### Project Goals
1. Build end-to-end ML pipeline from data collection to deployment
2. Compare traditional ML (XGBoost) vs graph-based methods (GNN, KGCN)
3. Investigate representation factors in film success
4. Create production-ready web application

### Deliverables
- Comprehensive data pipeline
- Three trained models + ensemble
- Interactive web application
- Technical report (Team15_ProjectReport_STA160.pdf)
- Deployment documentation
- Video demonstration

---

## Documentation

- **Project Report**: `Team15_ProjectReport_STA160.pdf`
- **Deployment Guide**: `Reproducible Deployment Document.pdf`
- **Demo Video**: `Final Demo Presentation - Trimmed.mov`
- **Demo Script**: `Final Demo Presentation Script.pdf`

---

## Known Limitations

1. **Data Coverage**: Limited to 2010-2025 films with TMDB/RT data
2. **API Dependencies**: Requires active TMDB and YouTube API keys
3. **Computational Requirements**: KGCN and GNN training require GPU for reasonable speed
4. **Scraping Fragility**: Rotten Tomatoes scraper may break if site structure changes
5. **Prediction Scope**: Best for mainstream films; indie/foreign films may have less accurate predictions

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

### Data Sources
- **The Movie Database (TMDB)** - Movie metadata and API
- **Rotten Tomatoes** - Critic and audience scores
- **YouTube** - Trailer engagement metrics

### Technologies
- **Streamlit** - Web framework
- **MongoDB Atlas** - Database hosting
- **HuggingFace** - BERT models for sentiment analysis
- **PyTorch Geometric** - Graph neural network library

### Course & Support
- **UC Davis STA 160** - Statistical Data Science
- **Instructors & TAs** - Guidance and feedback
- **Team 15 Members** - Collaborative development

---

## Contact

For questions about this project, please refer to the project report or deployment documentation.

---
