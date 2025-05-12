import os
import sys
import logging
import argparse
import mlflow
import pandas as pd
import joblib
from pathlib import Path

# Add current directory to path to handle imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    setup_mlflow, evaluate_model, log_metrics_to_mlflow, 
    log_figure_to_mlflow, plot_confusion_matrix,
    TOPIC_MAPPING
)
from data_processing import (
    load_data, download_nltk_resources, TextPreprocessor, 
    create_submission_file
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate model with MLflow tracking')
    
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing the dataset files')
    parser.add_argument('--model-uri', type=str, required=True,
                       help='URI of the MLflow model to evaluate (e.g., "runs:/run_id/models")')
    parser.add_argument('--vectorizer-path', type=str, default=None,
                       help='Path to the vectorizer joblib file')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for the MLflow run')
    parser.add_argument('--create-submission', action='store_true',
                       help='Whether to create a submission file for the test set')
    
    return parser.parse_args()

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup MLflow
    setup_mlflow()
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}...")
    train_df, test_df = load_data(args.data_dir)
    
    # Create run name if not provided
    run_name = args.run_name or f"evaluation_{Path(args.model_uri).name}"
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        
        # Log model URI
        mlflow.log_param("model_uri", args.model_uri)
        
        # Load model
        logger.info(f"Loading model from {args.model_uri}...")
        model = mlflow.sklearn.load_model(args.model_uri)
        
        # Load vectorizer if provided
        vectorizer = None
        if args.vectorizer_path:
            logger.info(f"Loading vectorizer from {args.vectorizer_path}...")
            vectorizer = joblib.load(args.vectorizer_path)
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
        
        # Check if test set has labels
        has_test_labels = 'label' in test_df.columns
        
        if has_test_labels:
            logger.info("Evaluating model on labeled test set...")
            
            # Prepare test data
            X_test_text = test_df['Question'].values
            y_test = test_df['label'].values
            
            # Preprocess text
            X_test_processed = preprocessor.preprocess(X_test_text)
            
            # Vectorize if vectorizer provided
            if vectorizer:
                X_test = vectorizer.transform(X_test_processed)
            else:
                # Assume model is a pipeline with vectorizer
                X_test = X_test_processed
            
            # Evaluate model
            metrics, y_pred = evaluate_model(model, X_test, y_test)
            
            # Log metrics
            log_metrics_to_mlflow(metrics)
            
            # Generate and log confusion matrix
            cm_fig = plot_confusion_matrix(y_test, y_pred)
            log_figure_to_mlflow(cm_fig, "figures/confusion_matrix.png")
            
            # Print results
            logger.info(f"F1-micro score: {metrics['f1_micro']:.4f}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            
            # Create submission file if requested
            if args.create_submission:
                submission_path = os.path.join(args.data_dir, f"submission_{run_id}.csv")
                create_submission_file(test_df, y_pred, submission_path)
        
        else:
            # Only generate predictions if no labels available
            logger.info("Generating predictions for unlabeled test set...")
            
            # Prepare test data
            X_test_text = test_df['Question'].values
            
            # Preprocess text
            X_test_processed = preprocessor.preprocess(X_test_text)
            
            # Vectorize if vectorizer provided
            if vectorizer:
                X_test = vectorizer.transform(X_test_processed)
            else:
                # Assume model is a pipeline with vectorizer
                X_test = X_test_processed
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Create submission file
            submission_path = os.path.join(args.data_dir, f"submission_{run_id}.csv")
            create_submission_file(test_df, y_pred, submission_path)
            
            logger.info(f"Predictions saved to {submission_path}")
        
        # Log class distribution
        pred_class_dist = pd.Series(y_pred).value_counts().sort_index()
        class_dist_str = ", ".join([f"{TOPIC_MAPPING[i]}: {count}" for i, count in pred_class_dist.items()])
        logger.info(f"Predicted class distribution: {class_dist_str}")
        
        # Log MLflow run info
        logger.info(f"MLflow run: {run_id}")
        logger.info(f"MLflow UI: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{run_id}")

if __name__ == "__main__":
    main() 