"""Main entry point for ML pipeline."""

import sys
import logging
from src.data_processing import get_train_test_data
from src.train import train_with_grid_search, save_model
from src.evaluate import evaluate_model, save_metrics, plot_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)
#

def main(model_name='logistic_regression'): # it supports 'logistic_regression', 'random_forest', 'lightgbm'
    """Run the complete ML pipeline."""
    logger.info("="*60)
    logger.info(f"ML Pipeline - {model_name.replace('_', ' ').title()}")
    logger.info("="*60)
    
    # Load and split data
    logger.info("\n1. Loading and splitting data...")
    X_train, X_test, y_train, y_test = get_train_test_data()
    logger.info(f"   Train: {len(X_train)} samples")
    logger.info(f"   Test: {len(X_test)} samples")
    
    # Train model with hyperparameter tuning
    logger.info("\n2. Training model with GridSearchCV...")
    model = train_with_grid_search(X_train, y_train, model_name)
    
    # Evaluate model
    logger.info("\n3. Evaluating model...")
    results, y_pred, y_prob = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Save results
    logger.info("\n4. Saving results...")
    save_model(model, model_name)
    save_metrics(results, model_name)
    plot_results(y_test, y_pred, y_prob, model_name)
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    # Get model name from command line args if provided
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'logistic_regression'
    main(model_name)