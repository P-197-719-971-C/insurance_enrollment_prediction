"""Prediction module for making predictions with trained models."""

import logging
import pandas as pd
from config import FEATURES

logger = logging.getLogger(__name__)

# Module-level cache for models (used by API)
_model_cache = {}


def _set_model_cache(cache: dict):
    """Set the model cache from external source (API)."""
    global _model_cache
    _model_cache = cache


def predict(model_name='logistic_regression', data=None):
    """
    Make predictions using a trained model.
    
    Args:
        model_name (str): Name of the saved model to load
        data (pd.DataFrame): Input data with features
        
    Returns:
        tuple: (predictions, probabilities)
    """
    # Use cached model if available, otherwise load from disk
    if model_name in _model_cache:
        logger.info(f"Using cached model: {model_name}")
        model = _model_cache[model_name]
        
        # 🔍 LOG: Show which model we fetched from cache and its ID
        logger.info(f"=== PREDICT: Fetched model '{model_name}' with id={id(model)} from cache ===")
    else:
        # Import here to avoid circular imports
        from src.train import load_model
        logger.info(f"Loading model from disk: {model_name}")
        model = load_model(model_name)
    
    # Validate input data
    if data is None:
        raise ValueError("Input data cannot be None")
    
    # Check if required features are present
    missing_features = set(FEATURES) - set(data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select only required features in correct order
    X = data[FEATURES]
    
    # Make predictions
    logger.info(f"Making predictions on {len(X)} samples...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    logger.info(f"Predictions complete")
    logger.info(f"  Predicted class 0: {(predictions == 0).sum()}")
    logger.info(f"  Predicted class 1: {(predictions == 1).sum()}")
    
    return predictions, probabilities


def predict_single(model_name='logistic_regression', 
                   has_dependents=None, 
                   employment_type=None, 
                   age=None, 
                   salary=None):
    """
    Make prediction for a single instance.
    
    Args:
        model_name (str): Name of the saved model
        has_dependents: Value for has_dependents feature
        employment_type: Value for employment_type feature
        age: Value for age feature
        salary: Value for salary feature
        
    Returns:
        tuple: (prediction, probability)
    """
    # Create DataFrame from input
    data = pd.DataFrame([{
        'has_dependents': has_dependents,
        'employment_type': employment_type,
        'age': age,
        'salary': salary
    }])
    
    # Make prediction
    predictions, probabilities = predict(model_name, data)
    
    # Return single values
    prediction = predictions[0]
    probability = probabilities[0, 1]  # Probability of class 1
    
    logger.info(f"Single prediction: {prediction} (probability: {probability:.4f})")
    
    return prediction, probability


def predict_from_csv(model_name='logistic_regression', csv_path=None):
    """
    Make predictions on data from CSV file.
    
    Args:
        model_name (str): Name of the saved model
        csv_path (str): Path to CSV file with input data
        
    Returns:
        pd.DataFrame: Original data with predictions and probabilities added
    """
    if csv_path is None:
        raise ValueError("csv_path cannot be None")
    
    logger.info(f"Loading data from {csv_path}")
    data = pd.read_csv(csv_path)
    
    # Make predictions
    predictions, probabilities = predict(model_name, data)
    
    # Add predictions to dataframe
    result_df = data.copy()
    result_df['prediction'] = predictions
    result_df['probability_class_0'] = probabilities[:, 0]
    result_df['probability_class_1'] = probabilities[:, 1]
    
    return result_df


if __name__ == "__main__":
    """Example usage of prediction functions."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    # Example 1: Single prediction
    print("="*60)
    print("Example 1: Single Prediction")
    print("="*60)
    
    prediction, probability = predict_single(
        model_name='logistic_regression',
        has_dependents='Yes',
        employment_type='Full-time',
        age=35,
        salary=75000
    )
    
    print(f"\nResult:")
    print(f"  Prediction: {prediction}")
    print(f"  Probability: {probability:.4f}")
    print(f"  Interpretation: {'Will enroll' if prediction == 1 else 'Will not enroll'}")
    
    # Example 2: Batch predictions
    print("\n" + "="*60)
    print("Example 2: Batch Predictions")
    print("="*60)
    
    # Create sample data
    sample_data = pd.DataFrame([
        {'has_dependents': 'Yes', 'employment_type': 'Full-time', 'age': 30, 'salary': 60000},
        {'has_dependents': 'No', 'employment_type': 'Part-time', 'age': 25, 'salary': 35000},
        {'has_dependents': 'Yes', 'employment_type': 'Full-time', 'age': 45, 'salary': 95000},
    ])
    
    predictions, probabilities = predict(
        model_name='logistic_regression',
        data=sample_data
    )
    
    print(f"\nResults:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"  Sample {i+1}: Prediction={pred}, Probability={prob[1]:.4f}")