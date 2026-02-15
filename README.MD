# Insurance Enrollment Prediction

Predict employee insurance enrollment using Machine Learning.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/scikit-learn-1.8+-orange?style=flat-square" alt="scikit-learn">
  <img src="https://img.shields.io/badge/FastAPI-0.129+-blue?style=flat-square" alt="FastAPI">
  <img src="https://img.shields.io/badge/LightGBM-4.6+-yellow?style=flat-square" alt="LightGBM">
</p>

## 📌 Overview

ML system for predicting employee insurance enrollment using three models with hyperparameter tuning via GridSearchCV.

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Train model (default: logistic_regression)
uv run python main.py

# Train specific model
uv run python main.py logistic_regression
uv run python main.py random_forest
uv run python main.py lightgbm

# Run API server
uv run uvicorn api:app --reload

# Run Web UI
uv run python gradio_app.py
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/models` | GET | Available models |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions (max 1000) |

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 89.7% | 91.0% | 92.4% | 91.7% | 97.1% |
| Random Forest | 99.9% | 100% | 99.9% | 99.9% | 100% |
| LightGBM | 99.9% | 100% | 99.9% | 99.9% | 100% |

## 🛠 Tech Stack

- **Python** 3.12+
- **ML**: scikit-learn, LightGBM, pandas, numpy
- **API**: FastAPI, uvicorn
- **Web UI**: Gradio
- **Visualization**: matplotlib, seaborn, plotly

## 📁 Project Structure

```
.
├── main.py              # ML pipeline entry point
├── api.py               # FastAPI application
├── gradio_app.py        # Gradio web interface
├── config.py            # Configuration
├── src/                 # Source code
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/              # Saved models (.pkl)
├── data/                # Data directory
└── results/             # Evaluation results
```

## 🐳 Docker Deployment

Containerized application with FastAPI server for easy deployment.

```bash
# Build Docker image
docker build -t insurance-prediction .

# Run container (API available at http://localhost:8000)
docker run -p 8000:8000 insurance-prediction
```

---

**Author**: Umrav Singh Shekhawat