"""Configuration for the ML pipeline."""

# Data paths
DATA_PATH = "data/raw/employee_data.csv"
MODEL_PATH = "models"
RESULTS_PATH = "results"
METRICS_PATH = "results/metrics"
PLOTS_PATH = "results/plots"

# Features
FEATURES = ['has_dependents', 'employment_type', 'age', 'salary']
TARGET = 'enrolled'
CATEGORICAL_FEATURES = ['has_dependents', 'employment_type']
NUMERIC_FEATURES = ['age', 'salary']

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Hyperparameter grid for tuning (updated for sklearn 1.8+)
LOGISTIC_PARAM_GRID = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['lbfgs', 'liblinear']
}

# Random Forest hyperparameter grid
RANDOM_FOREST_PARAM_GRID = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__max_features': ['sqrt'],
    'classifier__max_samples': [0.8],
}

# LightGBM hyperparameter grid
LIGHTGBM_PARAM_GRID = {
    'classifier__n_estimators': [100, 200],
    'classifier__num_leaves': [31, 63],
    'classifier__max_depth': [5, 10],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__min_child_samples': [20],
    'classifier__subsample': [0.8],
    'classifier__colsample_bytree': [0.8],
    'classifier__reg_alpha': [0, 0.1],
    'classifier__reg_lambda': [0, 0.1],
}
