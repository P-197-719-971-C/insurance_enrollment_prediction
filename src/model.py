"""Model definitions and creation."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from config import RANDOM_STATE


def create_logistic_regression():
    """Create logistic regression model with default parameters."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    return model


def create_random_forest():
    """Create Random Forest model with default parameters."""
    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    return model


def create_lightgbm():
    """Create LightGBM model with default parameters."""
    model = LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    return model


def get_model(model_name='logistic_regression'):
    """Factory function to get model by name."""
    models = {
        'logistic_regression': create_logistic_regression,  # DEFAULT
        'random_forest': create_random_forest,
        'lightgbm': create_lightgbm,
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(models.keys())}")
    
    return models[model_name]()
