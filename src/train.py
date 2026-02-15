"""Model training and hyperparameter tuning."""

import os
import joblib
import logging
from sklearn.model_selection import cross_val_score, GridSearchCV
from src.model import get_model
from src.feature_engineering import create_pipeline
from config import (
    CV_FOLDS, 
    LOGISTIC_PARAM_GRID, 
    RANDOM_FOREST_PARAM_GRID,
    LIGHTGBM_PARAM_GRID,
    MODEL_PATH
)

logger = logging.getLogger(__name__)


def train_with_cross_validation(X_train, y_train, model_name='logistic_regression'):
    """Train model with cross-validation."""
    model = get_model(model_name)
    pipeline = create_pipeline(model, model_name)  # Pass model_name for correct preprocessing
    
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, 
        cv=CV_FOLDS, scoring='roc_auc'
    )
    
    logger.info(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    return cv_scores


def train_with_grid_search(X_train, y_train, model_name='logistic_regression'):
    """Train model with hyperparameter tuning using GridSearchCV."""
    model = get_model(model_name)
    pipeline = create_pipeline(model, model_name)  # Pass model_name for correct preprocessing
    
    # Get appropriate param grid based on model
    if model_name == 'logistic_regression':
        param_grid = LOGISTIC_PARAM_GRID
    elif model_name == 'random_forest':
        param_grid = RANDOM_FOREST_PARAM_GRID
    elif model_name == 'lightgbm':
        param_grid = LIGHTGBM_PARAM_GRID
    else:
        raise ValueError(f"No param grid defined for {model_name}")
    
    grid_search = GridSearchCV(
        pipeline, param_grid,
        cv=CV_FOLDS, scoring='roc_auc',
        n_jobs=-1, verbose=0
    )
    
    logger.info("Starting GridSearchCV hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def save_model(model, model_name='logistic_regression'):
    """Save trained model to disk."""
    os.makedirs(MODEL_PATH, exist_ok=True)
    model_path = os.path.join(MODEL_PATH, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")


def load_model(model_name='logistic_regression'):
    """Load trained model from disk."""
    model_path = os.path.join(MODEL_PATH, f"{model_name}.pkl")
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model
