"""Data loading and preprocessing functions."""

import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_PATH, FEATURES, TARGET, TEST_SIZE, RANDOM_STATE


def load_data():
    """Load data from CSV file."""
    data = pd.read_csv(DATA_PATH)
    return data


def prepare_data(data):
    """Split data into features and target."""
    X = data[FEATURES]
    y = data[TARGET]
    return X, y


def split_data(X, y):
    """Split data into train and test sets with stratification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_train_test_data():
    """Main function to load and split data."""
    data = load_data()
    X, y = prepare_data(data)
    return split_data(X, y)