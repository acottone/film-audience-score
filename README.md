# Filmlytics: Movie Audience Score Prediction System

Predicting movie audience scores using pre-release data with ensemble machine learning on 66,000+ films.

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
- **0.1085 RMSE** ensemble model combining GNN, KGCN, and XGBoost for audience score prediction
- **66,000+ films** (2010-2025) with 150+ engineered features from TMDB, Rotten Tomatoes, and YouTube
- **80.2% accuracy** within ±10 percentage points of actual audience scores
- **Interactive web app** for real-time predictions, analytics, and film network visualization
- **Representation analysis** showing measurable impact of gender diversity on audience reception
- **Production-ready** deployment with MongoDB backend and Streamlit frontend

---

## Project Overview

Filmlytics is an advanced machine learning system that predicts how audiences will rate films **based on pre-release data**. By integrating data from The Movie Database (TMDB), Rotten Tomatoes, and YouTube, the system analyzes movie metadata, critic reviews, trailer engagement, and representation metrics to forecast audience scores with high accuracy.

The system employs a novel ensemble approach combining three complementary models: a Graph Neural Network (GNN) that captures film similarity through shared attributes, a Knowledge Graph Convolutional Network (KGCN) that learns semantic relationships in the film knowledge graph, and XGBoost for feature-based tabular prediction. A stacking meta-learner integrates these models to achieve superior performance (RMSE: 0.1085) compared to any individual model.

Built for film studios, marketers, and researchers, Filmlytics provides actionable insights for marketing strategy, financial forecasting, and understanding what film characteristics resonate with audiences. The interactive web interface enables users to search movies, compare predictions, explore analytics dashboards, and visualize film relationship networks.

The project emphasizes:
- **Multi-source data integration** from TMDB, Rotten Tomatoes, and YouTube APIs
- **Advanced feature engineering** with 150+ features including sentiment analysis and diversity metrics
- **Hybrid modeling approach** combining graph-based and tabular machine learning
- **Representation research** analyzing gender diversity impact on audience reception
- **Production deployment** with scalable MongoDB backend and interactive Streamlit frontend

---

## Key Findings

### Model Performance

- **Ensemble outperforms individual models**: Stacking meta-learner achieves 0.1085 RMSE vs 0.110 (XGBoost), 0.171 (KGCN), 0.195 (GNN)
- **Graph models capture unique signals**: KGCN and GNN provide complementary information to tabular features
- **High practical accuracy**: 80.2% of predictions within ±10 points, 95.1% within ±20 points

### Feature Importance

- **Top predictors**: Critic scores (0.23), budget (0.18), YouTube engagement (0.15), sentiment (0.12)
- **Temporal patterns**: Summer releases average 3.2 points higher than winter releases
- **Genre effects**: Action and Adventure films show higher variance in audience reception

### Representation Impact

- **Gender diversity correlation**: Films with >40% female cast score 2.8 points higher on average
- **Female directors**: Associated with 1.5 point increase in audience scores (controlling for other factors)
- **Balanced representation**: Gender balance score shows positive correlation (r=0.31, p<0.001)

### Performance Summary

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| **Stacking Ensemble** | **0.1085** | **0.0821** | **0.892** |
| XGBoost | 0.110 | 0.0835 | 0.887 |
| KGCN | 0.171 | 0.128 | 0.751 |
| GNN | 0.195 | 0.145 | 0.682 |

---

## Dataset

- **Source:** The Movie Database (TMDB), Rotten Tomatoes, YouTube Data API

- **Size/Scope:** 66,247 films released between 2010-2025

- **Key Statistics:**
  - 50,000+ critic and audience reviews with BERT sentiment scores
  - 40,000+ YouTube trailers with engagement metrics
  - 150+ engineered features per film
  - 85% of films have complete metadata (budget, runtime, cast)
  - Average of 12.3 cast members and 2.8 genres per film
  - Median budget: $15M, median audience score: 68/100

---

## Technical Implementation

### Algorithm: Stacking Ensemble with Heterogeneous Models

```
Ensemble Prediction = α·GNN(G) + β·KGCN(KG) + γ·XGB(X) + ε

Where:
  G  = Film similarity graph (nodes: movies, edges: shared attributes)
  KG = Knowledge graph (heterogeneous: movies, genres, cast, directors)
  X  = Feature matrix (150+ engineered features)
  α, β, γ = Learned weights via Ridge meta-learner
  
Optimization:
  min Σ(y_true - y_pred)² + λ||w||²
  
Optimal weights: α=0.15, β=0.28, γ=0.57
```

### Key Technical Decisions

#### 1. Heterogeneous Graph Architecture (KGCN)

Multi-node-type graph with movies, genres, directors, production companies, cast members, and countries. Uses separate message-passing functions for each edge type to preserve semantic relationships.

**Result:**
- 12% improvement over homogeneous GNN
- Captures complex film industry relationships
- Enables interpretable predictions via graph attention

#### 2. BERT Sentiment Analysis on Reviews

Applied DistilBERT fine-tuned on SST-2 to 50,000+ Rotten Tomatoes reviews, aggregating sentiment scores as features rather than using raw text.

**Result:**
- Sentiment features rank 4th in XGBoost feature importance
- 0.015 RMSE improvement over bag-of-words approach
- Computationally efficient (pre-computed offline)

#### 3. Temporal Train-Test Split

Chronological split (80% train, 10% validation, 10% test) rather than random split to simulate real-world prediction scenario.

**Result:**
- More realistic performance estimates
- Prevents data leakage from future films
- Test RMSE 0.008 higher than random split (more honest evaluation)

#### 4. Stacking Meta-Learner with Ridge Regression

Ridge regression (α=1.0) as meta-learner instead of simple weighted average or complex neural network.

**Result:**
- Prevents overfitting to validation set
- Interpretable learned weights
- 0.0025 RMSE improvement over grid-search weighted average

---

## Methodology

1. **Data Collection**: Fetch movie metadata from TMDB API for 66,000+ films (2010-2025)

2. **Data Cleaning**: Filter invalid entries, handle missing values, standardize formats

3. **Rotten Tomatoes Integration**: Map TMDB movies to RT URLs, scrape reviews and scores

4. **Sentiment Analysis**: Apply BERT (DistilBERT-SST2) to 50,000+ reviews for sentiment scores

5. **Gender Diversity Metrics**: Extract cast/crew gender data via TMDB API, compute diversity scores

6. **YouTube Engagement**: Fetch trailer metrics (views, likes, comments) via YouTube Data API

7. **Feature Engineering**: Create 150+ features across temporal, budget, engagement, sentiment, and representation categories

8. **Graph Construction**: Build heterogeneous knowledge graph with 6 node types and 5 edge types

9. **Model Training**: Train GNN, KGCN, and XGBoost models independently with cross-validation

10. **Ensemble Optimization**: Train Ridge meta-learner on validation set to combine model predictions

11. **Evaluation**: Test on held-out chronological test set (most recent 10% of films)

12. **Deployment**: Load models and data into MongoDB, deploy Streamlit web application

---

## Results & Visualizations

### Prediction Accuracy Distribution

80.2% of predictions fall within ±10 percentage points of actual audience scores. The ensemble model shows tighter error distribution compared to individual models, with fewer extreme outliers.

### Feature Importance (XGBoost)

Top 10 features by SHAP value:
1. Critic score (0.23)
2. Budget (log-transformed) (0.18)
3. YouTube view count (0.15)
4. Review sentiment score (0.12)
5. Release year (0.08)
6. Runtime (0.06)
7. Female cast percentage (0.05)
8. Genre: Action (0.04)
9. YouTube like ratio (0.04)
10. Vote average (TMDB) (0.03)

### Model Comparison

Ensemble predictions show lower variance and better calibration than individual models. KGCN excels on films with strong genre/cast relationships, while XGBoost performs better on high-budget blockbusters.

### Representation Analysis

Films with balanced gender representation (40-60% female cast) show:
- 2.8 point higher average audience score
- Lower prediction error (RMSE: 0.095 vs 0.115)
- Stronger correlation between critic and audience scores (r=0.78 vs r=0.65)

---

### Environment

- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **Hardware**:
  - Minimum: 8GB RAM, 4-core CPU
  - Recommended: 16GB RAM, 8-core CPU, NVIDIA GPU (for KGCN/GNN training)
- **API Keys**: TMDB Bearer Token, YouTube Data API Key (for data collection)
- **Database**: MongoDB Atlas account (free tier sufficient)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd "film audience prediction (python)"

# Install backend dependencies
cd team15_code_data/backend
pip install -r requirements.txt

# Install data pipeline dependencies
cd ../data
pip install pandas numpy requests beautifulsoup4 transformers torch tqdm python-dotenv

# Install model training dependencies (if training from scratch)
cd ../model_training
pip install torch torch-geometric xgboost scikit-learn joblib
```

### Usage

```bash
# 1. Set up environment variables (create .env file)
cd team15_code_data/data
echo "TMDB_BEARER_TOKEN=your_token_here" > .env
echo "YOUTUBE_API_KEY=your_key_here" >> .env

# 2. Run data pipeline (optional - pre-collected data included)
python data_pipeline.py --step all

# 3. Set up MongoDB database
cd ../backend
# Edit mongodb_setup.py with your MongoDB connection string
python mongodb_setup.py

# 4. Train models (optional - pre-trained models included)
cd ../model_training/xgboost
python xgb_modelMATT.py

cd ../KGCN
python kgcn.py

cd ../
python ensemble_model.py

# 5. Run web application
cd ../backend
streamlit run streamlit_app_clara.py
```

### Outputs

- **Data files**: `team15_code_data/data/complete_data.csv` (final merged dataset)
- **Trained models**: `team15_code_data/backend/model_artifacts/*.pkl`
- **Predictions**: `*_preds_all_movies.csv` files with model predictions
- **Visualizations**: Generated in web app and saved in `visualizations/` directories
- **Web app**: Accessible at `http://localhost:8501` after running Streamlit

---

## Project Structure
```
film-audience-score/
├── backend/                   # Backend services
│   ├── streamlit_app_clara.py # Main Streamlit app
│   ├── mongodb_setup.py       # Database initialization
│   ├── model_artifacts/       # Trained models & scalers
│   └── requirements.txt       # Backend dependencies
├── data/                      # Data collection & preprocessing
│   ├── data_pipeline.py       # Complete data pipeline script
│   ├── complete_data.csv      # Final merged dataset
│   ├── tmdb_data.csv          # TMDB movie metadata
│   ├── tmdb_with_sentiment.csv  # TMDB movie sentiment analysis
│   ├── tmdb_with_urls.csv     # Youtube trailer urls
│   ├── rt_reviews.json        # Rotten Tomatoes reviews
│   └── rt_sentiment.csv       # BERT sentiment analysis
├── model_training/            # Model training scripts
│   ├── gnn/                   # Graph Neural Network
│   │   └── gnn.ipynb          # GNN training notebook
│   ├── KGCN/                  # Knowledge Graph CNN
│   │   └── kgcn.py            # KGCN training script
│   ├── xgboost/               # XGBoost models
│   │   ├── xgb_model.py       # XGBoost training
│   │   └── feature_importance.csv
│   └── ensemble_model.py      # Stacking ensemble
└── frontend/                  # Frontend (Streamlit UI)
│   └── streamlit_app_clara.py
├── Team15_ProjectReport_STA160.pdf
├── Reproducible Deployment Document.pdf
└── README.md
```

---

## Challenges & Limitations

### Technical Challenges Faced

- **API rate limiting**: TMDB and YouTube APIs have strict rate limits; implemented exponential backoff and caching
- **Web scraping fragility**: Rotten Tomatoes HTML structure changes required robust parsing with fallbacks
- **Graph memory constraints**: KGCN with 66K nodes exceeded GPU memory; implemented mini-batch training
- **Missing data**: ~15% of films lack budget/revenue data; used imputation and indicator features
- **Computational cost**: Full pipeline takes ~8 hours on single machine; parallelized where possible

### Current Limitations

- **Temporal scope**: Limited to 2010-2025 films; older films have sparse data
- **English-language bias**: Dataset predominantly English-language films; international films underrepresented
- **Genre imbalance**: Action/Drama overrepresented; Documentary/Foreign underrepresented
- **Prediction lag**: Requires trailer release; cannot predict for films without trailers
- **API dependencies**: Requires active API keys; rate limits constrain real-time updates

### Scope Boundaries

- **No box office prediction**: Focuses on audience scores, not financial performance
- **No streaming metrics**: Limited to theatrical releases; streaming-only films excluded
- **No real-time updates**: Models trained on static snapshot; requires retraining for new data
- **No causal inference**: Correlational analysis only; cannot claim causation for representation effects

---

## Technologies Used

### Core Stack

- **Python 3.8+**: Primary programming language
- **PyTorch 2.0+**: Deep learning framework for GNN and KGCN
- **XGBoost**: Gradient boosting library for tabular prediction
- **Scikit-learn**: ML utilities, preprocessing, ensemble methods
- **Streamlit**: Web application framework
- **MongoDB Atlas**: Cloud database for film data storage

### Data Processing

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **BeautifulSoup4**: HTML parsing for web scraping
- **Requests**: HTTP library for API calls

### Machine Learning Methods

- **Graph Neural Networks (GNN)**: Message-passing on film similarity graph
- **Knowledge Graph Convolutional Networks (KGCN)**: Heterogeneous graph learning
- **Gradient Boosting (XGBoost)**: Tree-based ensemble for tabular data
- **Stacking Ensemble**: Ridge regression meta-learner
- **BERT (DistilBERT)**: Transformer-based sentiment analysis

### Natural Language Processing

- **Transformers (HuggingFace)**: Pre-trained BERT models
- **DistilBERT-SST2**: Sentiment classification on reviews
- **Tokenization**: Text preprocessing for sentiment analysis

### Visualization

- **Plotly**: Interactive charts and graphs
- **NetworkX**: Graph visualization and analysis
- **PyVis**: Interactive network visualizations
- **Matplotlib/Seaborn**: Static plots for model evaluation

### APIs & Data Sources

- **TMDB API**: Movie metadata, cast, crew, budget
- **YouTube Data API v3**: Trailer engagement metrics
- **Rotten Tomatoes**: Critic and audience scores (scraped)

---

## Author
Developed by a 5-person team for STA 160.

**Angelina Cottone, Matthew Ward, Clara Wei, Dylan Sidhu, Nidhi Deshmukh**

*Course*: STA 160 - Statistical Data Science

*Institution*: University of California, Davis

*Date*: December 2025

---

## References

### Data Sources

- The Movie Database (TMDB): https://www.themoviedb.org/
- Rotten Tomatoes: https://www.rottentomatoes.com/
- YouTube Data API: https://developers.google.com/youtube/v3

### Documentation

- Project Report: `Team15_ProjectReport_STA160.pdf`
- Deployment Guide: `Reproducible Deployment Document.pdf`

---
*Last Updated: January 2026*
