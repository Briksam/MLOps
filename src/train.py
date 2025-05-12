import os
import sys
import logging
import argparse
import time
import mlflow


# Add current directory to path to handle imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    setup_mlflow, evaluate_model, log_metrics_to_mlflow, 
    log_params_to_mlflow, log_figure_to_mlflow,
    plot_confusion_matrix, save_model, TOPIC_MAPPING, ARTIFACT_PATH
)
from data_processing import (
    load_data, prepare_data, download_nltk_resources, 
    create_submission_file
)
from models import (
    get_model_with_default_params, 
    get_ensemble_model, get_stacking_model
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train math problem classification model with MLflow tracking')
    
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing the dataset files')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--model-type', type=str, default='lr',
                       choices=['lr', 'rf', 'gb', 'svm', 'nb', 'ensemble', 'stacking'],
                       help='Type of model to train')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for the MLflow run')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Size of validation split')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--register-model', action='store_true',
                       help='Register model to MLflow registry')
    parser.add_argument('--model-name', type=str, default='math-topic-classifier',
                       help='Name for registered model')
    
    return parser.parse_args()

def train_model(X_train, X_val, y_train, y_val, model_type='lr', model_params=None):
    """Train and evaluate a model."""
    logger.info(f"Training {model_type} model...")
    
    # Get model
    if model_type == 'ensemble':
        model = get_ensemble_model()
    elif model_type == 'stacking':
        model = get_stacking_model()
    else:
        model = get_model_with_default_params(model_type, model_params)
    
    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f} seconds")
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics, y_pred = evaluate_model(model, X_val, y_val)
    
    # Log training time
    metrics['training_time'] = train_time
    
    # Generate confusion matrix
    cm_fig = plot_confusion_matrix(y_val, y_pred)
    
    return model, metrics, cm_fig, y_pred

def main():
    """Main training function."""
    args = parse_args()
    
    # Setup MLflow
    setup_mlflow()
    
    # Ensure no active run interferes with our new run
    if mlflow.active_run():
        logger.info(f"Ending active run: {mlflow.active_run().info.run_id}")
        mlflow.end_run()
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}...")
    train_df, test_df = load_data(args.data_dir)
    
    # Create run name if not provided
    run_name = args.run_name or f"{args.model_type}_model_{int(time.time())}"
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        
        # Prepare data
        logger.info("Preparing data...")
        X_train, X_val, y_train, y_val, vectorizer, preprocessor = prepare_data(
            train_df, 
            test_size=args.test_size, 
            random_state=args.random_state
        )
        
        # Log dataset info
        mlflow.log_param("train_examples", X_train.shape[0])
        mlflow.log_param("val_examples", X_val.shape[0])
        mlflow.log_param("features", X_train.shape[1])
        mlflow.log_param("classes", len(TOPIC_MAPPING))
        
        # Train model
        model, metrics, cm_fig, y_pred = train_model(
            X_train, X_val, y_train, y_val,
            model_type=args.model_type
        )
        
        # Log model parameters
        try:
            model_params = model.get_params()
            log_params_to_mlflow(model_params)
        except:
            logger.warning("Could not extract model parameters")
        
        # Log metrics
        log_metrics_to_mlflow(metrics)
        
        # Log confusion matrix
        log_figure_to_mlflow(cm_fig, "figures/confusion_matrix.png")
        
        # Save and log model
        model_dir = os.path.join(args.model_dir, run_id)
        os.makedirs(model_dir, exist_ok=True)
        model_path = save_model(model, model_dir, f"{args.model_type}_model")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=ARTIFACT_PATH,
            signature=mlflow.models.signature.infer_signature(X_train, model.predict(X_train[:1])),
            input_example=X_train[:1]
        )
        logger.info(f"Model logged to MLflow at {ARTIFACT_PATH}")
        
        # Save vectorizer
        vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
        import joblib
        joblib.dump(vectorizer, vectorizer_path)
        mlflow.log_artifact(vectorizer_path, "models")
        
        # Generate predictions on test set
        if 'label' not in test_df.columns:
            logger.info("Generating predictions on test set...")
            # Process test data
            test_text = test_df['Question'].values
            test_processed = preprocessor.preprocess(test_text)
            X_test = vectorizer.transform(test_processed)
            
            # Predict
            test_preds = model.predict(X_test)
            
            # Create submission file
            submission_path = os.path.join(args.data_dir, f"submission_{args.model_type}.csv")
            create_submission_file(test_df, test_preds, submission_path)
            
            # Log the submission file
            mlflow.log_artifact(submission_path, "submissions")
            logger.info(f"Submission file created: {submission_path}")
        
        # Register model if requested
        if args.register_model:
            from utils import register_model_to_registry, transition_model_stage
            
            logger.info(f"Registering model '{args.model_name}' to MLflow Registry...")
            registered_model = register_model_to_registry(run_id, args.model_name)
            
            # Transition to staging by default
            transition_model_stage(args.model_name, registered_model.version, "Staging")
            
            logger.info(f"Model registered and moved to Staging: {args.model_name}, version: {registered_model.version}")
        
        # Print summary
        logger.info("Training complete!")
        logger.info(f"F1-micro score: {metrics['f1_micro']:.4f}")
        logger.info(f"MLflow run: {run_id}")
        logger.info(f"MLflow UI: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{run_id}")

    # Ensure any active run is ended
    if mlflow.active_run():
        logger.info(f"Ending active run: {mlflow.active_run().info.run_id}")
        mlflow.end_run()

if __name__ == "__main__":
    main() 