"""Feature engineering and preprocessing pipeline."""

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def create_preprocessor():
    """
    Create preprocessing pipeline for logistic regression.
    - OneHotEncoder for categorical features
    - StandardScaler for numeric features
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), 
             CATEGORICAL_FEATURES),
            ('num', StandardScaler(), NUMERIC_FEATURES)
        ],
        remainder='passthrough'
    )
    return preprocessor


def create_tree_preprocessor():
    """
    Create preprocessing pipeline for tree-based models (Random Forest, LightGBM).
    - OrdinalEncoder for categorical features (better for trees)
    - passthrough for numeric features (no scaling needed)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), 
             CATEGORICAL_FEATURES),
            ('num', 'passthrough', NUMERIC_FEATURES)  # No scaling needed for trees
        ],
        remainder='passthrough'
    )
    return preprocessor


def create_pipeline(model, model_name='logistic_regression'):
    """
    Create full pipeline with model-specific preprocessing and model.
    
    Args:
        model: The classifier model
        model_name: 'logistic_regression', 'random_forest', or 'lightgbm'
    """
    # Tree-based models use different preprocessing
    if model_name in ['random_forest', 'lightgbm']:
        preprocessor = create_tree_preprocessor()
    else:
        preprocessor = create_preprocessor()
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    return pipeline
