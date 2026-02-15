"""Comprehensive unit tests for ML pipeline."""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import load_data, prepare_data, split_data
from src.feature_engineering import create_preprocessor, create_pipeline
from src.model import get_model, create_logistic_regression
from src.train import train_with_cross_validation
from config import FEATURES, TARGET, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEST_SIZE


class TestDataProcessing(unittest.TestCase):
    """Test data processing functions."""
    
    def setUp(self):
        """Load data once for all tests."""
        self.data = load_data()
    
    def test_load_data_returns_dataframe(self):
        """Test that load_data returns a pandas DataFrame."""
        self.assertIsInstance(self.data, pd.DataFrame)
    
    def test_load_data_not_empty(self):
        """Test that loaded data is not empty."""
        self.assertGreater(len(self.data), 0)
        self.assertGreater(self.data.shape[1], 0)
    
    def test_load_data_has_required_columns(self):
        """Test that all required columns are present."""
        required_cols = FEATURES + [TARGET]
        for col in required_cols:
            self.assertIn(col, self.data.columns, 
                         f"Required column '{col}' not found in data")
    
    def test_prepare_data_splits_correctly(self):
        """Test that prepare_data returns X and y with correct shapes."""
        X, y = prepare_data(self.data)
        
        self.assertEqual(len(X), len(y))
        self.assertEqual(X.shape[1], len(FEATURES))
        self.assertEqual(list(X.columns), FEATURES)
    
    def test_prepare_data_no_nulls_in_target(self):
        """Test that target variable has no null values."""
        X, y = prepare_data(self.data)
        self.assertEqual(y.isnull().sum(), 0)
    
    def test_split_data_proportions(self):
        """Test that train-test split maintains correct proportions."""
        X, y = prepare_data(self.data)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        total = len(X)
        test_ratio = len(X_test) / total
        
        self.assertAlmostEqual(test_ratio, TEST_SIZE, places=2)
    
    def test_split_data_stratification(self):
        """Test that class distribution is maintained in train and test."""
        X, y = prepare_data(self.data)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Check class distribution is similar
        train_ratio = y_train.sum() / len(y_train)
        test_ratio = y_test.sum() / len(y_test)
        original_ratio = y.sum() / len(y)
        
        self.assertAlmostEqual(train_ratio, original_ratio, places=1,
                              msg="Train set class distribution differs from original")
        self.assertAlmostEqual(test_ratio, original_ratio, places=1,
                              msg="Test set class distribution differs from original")
    
    def test_split_data_no_data_leakage(self):
        """Test that train and test sets don't overlap."""
        X, y = prepare_data(self.data)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Check indices don't overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        
        self.assertEqual(len(train_indices.intersection(test_indices)), 0,
                        "Train and test sets have overlapping indices")


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functions."""
    
    def test_create_preprocessor_returns_transformer(self):
        """Test that preprocessor is created successfully."""
        preprocessor = create_preprocessor()
        self.assertIsNotNone(preprocessor)
        self.assertTrue(hasattr(preprocessor, 'fit'))
        self.assertTrue(hasattr(preprocessor, 'transform'))
    
    def test_create_preprocessor_has_correct_features(self):
        """Test that preprocessor handles correct feature types."""
        preprocessor = create_preprocessor()
        transformers = preprocessor.transformers
        
        # Check categorical transformer
        cat_transformer, cat_features = transformers[0][1], transformers[0][2]
        self.assertEqual(cat_features, CATEGORICAL_FEATURES)
        
        # Check numeric transformer
        num_transformer, num_features = transformers[1][1], transformers[1][2]
        self.assertEqual(num_features, NUMERIC_FEATURES)
    
    def test_create_pipeline_structure(self):
        """Test that pipeline has correct structure."""
        model = get_model('logistic_regression')
        pipeline = create_pipeline(model)
        
        self.assertIsInstance(pipeline, Pipeline)
        self.assertIn('preprocessor', pipeline.named_steps)
        self.assertIn('classifier', pipeline.named_steps)
    
    def test_pipeline_fit_transform(self):
        """Test that pipeline can fit and predict."""
        from src.data_processing import get_train_test_data
        X_train, X_test, y_train, y_test = get_train_test_data()
        
        model = get_model('logistic_regression')
        pipeline = create_pipeline(model)
        
        # Should not raise error
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        self.assertEqual(len(predictions), len(y_test))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))


class TestModel(unittest.TestCase):
    """Test model creation and configuration."""
    
    def test_create_logistic_regression(self):
        """Test logistic regression creation."""
        model = create_logistic_regression()
        self.assertIsInstance(model, LogisticRegression)
    
    def test_get_model_logistic_regression(self):
        """Test model factory for logistic regression."""
        model = get_model('logistic_regression')
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'fit'))
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(hasattr(model, 'predict_proba'))
    
    def test_get_model_invalid_name(self):
        """Test that invalid model name raises error."""
        with self.assertRaises(ValueError):
            get_model('invalid_model_name')
    
    def test_model_has_random_state(self):
        """Test that model has random_state for reproducibility."""
        model = get_model('logistic_regression')
        self.assertTrue(hasattr(model, 'random_state'))
        self.assertIsNotNone(model.random_state)


class TestTraining(unittest.TestCase):
    """Test training functions."""
    
    def test_cross_validation_returns_scores(self):
        """Test that cross-validation returns valid scores."""
        from src.data_processing import get_train_test_data
        X_train, X_test, y_train, y_test = get_train_test_data()
        
        cv_scores = train_with_cross_validation(X_train, y_train)
        
        self.assertEqual(len(cv_scores), 5)  # 5-fold CV
        self.assertTrue(all(0 <= score <= 1 for score in cv_scores))


class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data loading to prediction."""
        # Load and prepare data
        from src.data_processing import get_train_test_data
        X_train, X_test, y_train, y_test = get_train_test_data()
        
        # Create and train model
        model = get_model('logistic_regression')
        pipeline = create_pipeline(model)
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        
        # Validate outputs
        self.assertEqual(len(predictions), len(y_test))
        self.assertEqual(probabilities.shape, (len(y_test), 2))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        self.assertTrue(all(0 <= prob <= 1 for row in probabilities for prob in row))
    
    def test_pipeline_reproducibility(self):
        """Test that pipeline produces same results with same random seed."""
        from src.data_processing import get_train_test_data
        X_train, X_test, y_train, y_test = get_train_test_data()
        
        # Train two models with same config
        model1 = get_model('logistic_regression')
        pipeline1 = create_pipeline(model1)
        pipeline1.fit(X_train, y_train)
        pred1 = pipeline1.predict(X_test)
        
        model2 = get_model('logistic_regression')
        pipeline2 = create_pipeline(model2)
        pipeline2.fit(X_train, y_train)
        pred2 = pipeline2.predict(X_test)
        
        # Predictions should be identical
        np.testing.assert_array_equal(pred1, pred2, 
                                      err_msg="Pipeline not reproducible")


class TestDataQuality(unittest.TestCase):
    """Test data quality and validation."""
    
    def test_no_missing_values_in_features(self):
        """Test that features have no missing values after preparation."""
        data = load_data()
        X, y = prepare_data(data)
        
        missing_counts = X.isnull().sum()
        self.assertEqual(missing_counts.sum(), 0,
                        f"Features have missing values: {missing_counts[missing_counts > 0]}")
    
    def test_categorical_features_valid(self):
        """Test that categorical features have expected values."""
        data = load_data()
        X, y = prepare_data(data)
        
        for col in CATEGORICAL_FEATURES:
            unique_vals = X[col].unique()
            self.assertGreater(len(unique_vals), 0,
                             f"Categorical feature '{col}' has no values")
            self.assertLess(len(unique_vals), 100,
                           f"Categorical feature '{col}' has too many unique values")
    
    def test_numeric_features_valid_range(self):
        """Test that numeric features have reasonable values."""
        data = load_data()
        X, y = prepare_data(data)
        
        for col in NUMERIC_FEATURES:
            self.assertFalse(X[col].isnull().any(),
                           f"Numeric feature '{col}' has null values")
            self.assertTrue(np.isfinite(X[col]).all(),
                           f"Numeric feature '{col}' has infinite values")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
