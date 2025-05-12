import os
import sys
import logging
import argparse
import time
import mlflow
import joblib

# Add current directory to path to handle imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    setup_mlflow, log_metrics_to_mlflow, 
    log_figure_to_mlflow, plot_confusion_matrix, TOPIC_MAPPING
) 
from data_processing import (
    load_data, download_nltk_resources,
    create_submission_file
)
from models import (
    get_model_with_default_params
)
from transformer_model import (
    DebertaClassifier, HybridEnsembleClassifier, FeatureEngineeringPreprocessor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train enhanced math problem classification model with DeBERTa')
    
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Directory containing the dataset files')
    parser.add_argument('--model-dir', type=str, default='models',
                      help='Directory to save trained models')
    parser.add_argument('--model-approach', type=str, default='hybrid',
                      choices=['deberta', 'hybrid'],
                      help='Type of model approach to use (deberta or hybrid ensemble)')
    parser.add_argument('--run-name', type=str, default=None,
                      help='Name for the MLflow run')
    parser.add_argument('--experiment-name', type=str, default=None,
                      help='Name for the MLflow experiment (defaults to transformer-models for deberta, math-problem-classification for hybrid)')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Size of validation split')
    parser.add_argument('--random-state', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--register-model', action='store_true',
                      help='Register model to MLflow registry')
    parser.add_argument('--model-name', type=str, default='math-topic-classifier-enhanced',
                      help='Name for registered model')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=8,
                      help='Number of training epochs (recommended: 8-10 for better performance)')
    parser.add_argument('--max-length', type=int, default=128,
                      help='Maximum sequence length for tokenization')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                      help='Learning rate for transformer model (recommended: 1e-5)')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda or cpu, default: auto-detect)')
    parser.add_argument('--traditional-models', type=str, default='lr,rf',
                      help='Comma-separated list of traditional models to include in hybrid ensemble')
    parser.add_argument('--use-feature-engineering', type=bool, default=True,
                      help='Whether to use feature engineering integration')
    parser.add_argument('--feature-integration-mode', type=str, default='hybrid',
                      choices=['none', 'hybrid', 'ensemble'],
                      help='How to integrate engineered features with transformer')
    parser.add_argument('--feature-weight', type=float, default=0.3,
                      help='Weight given to engineered features (0-1) in hybrid mode')
    
    return parser.parse_args()

def train_deberta_model(
    X_train, X_val, y_train, y_val, 
    batch_size=16, 
    epochs=8, 
    max_length=128,
    learning_rate=1e-5,
    device=None,
    num_labels=8,
    random_state=42,
    use_feature_engineering=True,
    feature_integration_mode='hybrid',
    feature_weight=0.3
):
    """Train and evaluate a DeBERTa model.
    
    Feature engineering integration modes:
    - 'none': No feature engineering is used
    - 'hybrid': Features are combined with transformer output using weighted average
    - 'ensemble': Features are used in a separate model and results are ensembled
    """
    logger.info("Training DeBERTa model...")
    
    # Apply feature engineering if requested
    feature_eng = None
    if use_feature_engineering:
        logger.info(f"Applying feature engineering with integration mode: {feature_integration_mode}")
        
        # Validate integration mode
        valid_modes = ['none', 'hybrid', 'ensemble']
        assert feature_integration_mode in valid_modes, f"Invalid feature integration mode: {feature_integration_mode}. Must be one of {valid_modes}"
        
        feature_eng = FeatureEngineeringPreprocessor()
        
        # Fit and transform training data
        train_features_df = feature_eng.fit_transform(X_train)
        val_features_df = feature_eng.transform(X_val)
        
        # Log feature information
        logger.info(f"Generated {train_features_df.shape[1]} engineered features")
        logger.info("Feature engineering applied - extracted math-specific patterns and features")
    
    # Initialize model
    model = DebertaClassifier(
        num_labels=num_labels,
        batch_size=batch_size,
        epochs=epochs,
        max_length=max_length,
        learning_rate=learning_rate,
        device=device,
        random_state=random_state,
        feature_preprocessor=feature_eng,
        feature_integration_mode=feature_integration_mode if use_feature_engineering else "none",
        feature_weight=feature_weight
    )
    
    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f} seconds")
    
    # Evaluate model
    logger.info("Evaluating model...")
    y_pred = model.predict(X_val)
    
    # Calculate metrics directly from the predictions we already have
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'f1_micro': f1_score(y_val, y_pred, average='micro'),
        'f1_macro': f1_score(y_val, y_pred, average='macro'),
        'f1_weighted': f1_score(y_val, y_pred, average='weighted'),
        'precision_micro': precision_score(y_val, y_pred, average='micro'),
        'precision_macro': precision_score(y_val, y_pred, average='macro'),
        'precision_weighted': precision_score(y_val, y_pred, average='weighted'),
        'recall_micro': recall_score(y_val, y_pred, average='micro'),
        'recall_macro': recall_score(y_val, y_pred, average='macro'),
        'recall_weighted': recall_score(y_val, y_pred, average='weighted')
    }
    metrics['training_time'] = train_time
    
    # Generate confusion matrix
    cm_fig = plot_confusion_matrix(y_val, y_pred)
    
    return model, metrics, cm_fig, y_pred, feature_eng

def train_hybrid_model(
    X_train, X_val, y_train, y_val,
    traditional_models=None,
    batch_size=16,
    epochs=3,
    max_length=128,
    learning_rate=2e-5,
    device=None,
    num_labels=8,
    random_state=42
):
    """Train and evaluate a hybrid ensemble model."""
    logger.info("Training hybrid ensemble model...")
    
    # Initialize DeBERTa classifier
    deberta_model = DebertaClassifier(
        num_labels=num_labels,
        batch_size=batch_size,
        epochs=epochs,
        max_length=max_length,
        learning_rate=learning_rate,
        device=device,
        random_state=random_state
    )
    
    # Initialize traditional models
    if traditional_models is None:
        traditional_models = ['lr', 'rf']
        
    trad_models = []
    for model_type in traditional_models:
        model = get_model_with_default_params(model_type)
        trad_models.append((model_type, model))
    
    # Initialize hybrid ensemble
    model = HybridEnsembleClassifier(
        transformer_model=deberta_model,
        traditional_models=trad_models,
        device=device
    )
    
    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f} seconds")
    
    # Evaluate model
    logger.info("Evaluating model...")
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'f1_micro': f1_score(y_val, y_pred, average='micro'),
        'f1_macro': f1_score(y_val, y_pred, average='macro'),
        'f1_weighted': f1_score(y_val, y_pred, average='weighted'),
        'precision_micro': precision_score(y_val, y_pred, average='micro'),
        'precision_macro': precision_score(y_val, y_pred, average='macro'),
        'precision_weighted': precision_score(y_val, y_pred, average='weighted'),
        'recall_micro': recall_score(y_val, y_pred, average='micro'),
        'recall_macro': recall_score(y_val, y_pred, average='macro'),
        'recall_weighted': recall_score(y_val, y_pred, average='weighted')
    }
    
    # Log training time
    metrics['training_time'] = train_time
    
    # Generate confusion matrix
    cm_fig = plot_confusion_matrix(y_val, y_pred)
    
    return model, metrics, cm_fig, y_pred

def log_model_with_vectorizer(model, vectorizer, artifact_path, model_dir='models', run_id=None):
    """Log model and vectorizer to MLflow."""
    # Use models directory instead of temp directory
    if run_id:
        # Create model directory if it doesn't exist
        model_path = os.path.join(model_dir, run_id, "model.joblib")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save vectorizer if provided
        if vectorizer:
            vectorizer_path = os.path.join(model_dir, run_id, "vectorizer.joblib")
            joblib.dump(vectorizer, vectorizer_path)
            mlflow.log_artifact(vectorizer_path, artifact_path)

    # Using PyFunc to handle custom models
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=mlflow.pyfunc.PythonModel(),
        artifacts={"model_path": model_path},
        code_path=[os.path.join("src", "transformer_model.py")]
    )
    
    logger.info(f"Model logged to MLflow at {artifact_path}")

def main():
    """Main training function."""
    args = parse_args()
    
    # Determine experiment name based on model approach if not provided
    if args.experiment_name is None:
        if args.model_approach == 'deberta':
            experiment_name = "transformer-models"
        else:
            experiment_name = "math-problem-classification"
    else:
        experiment_name = args.experiment_name
    
    # Setup MLflow
    setup_mlflow(experiment_name=experiment_name)
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}...")
    train_df, test_df = load_data(args.data_dir)
    
    # Create run name if not provided
    run_name = args.run_name or f"{args.model_approach}_enhanced_{int(time.time())}"
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        
        # Basic data preparation - no vectorization needed for transformer models
        logger.info("Preparing data...")
        # Get train-val split
        from sklearn.model_selection import train_test_split
        train_data, val_data = train_test_split(
            train_df, 
            test_size=args.test_size, 
            random_state=args.random_state,
            stratify=train_df['label']
        )
        
        # Extract features and labels
        X_train = train_data['Question'].values
        y_train = train_data['label'].values
        X_val = val_data['Question'].values
        y_val = val_data['label'].values
        
        # Log dataset info
        mlflow.log_param("train_examples", len(X_train))
        mlflow.log_param("val_examples", len(X_val))
        mlflow.log_param("classes", len(TOPIC_MAPPING))
        
        # Parse traditional models list if provided
        if args.traditional_models:
            traditional_models = [m.strip() for m in args.traditional_models.split(',')]
        else:
            traditional_models = None
        
        try:
            # Train appropriate model based on selected approach
            if args.model_approach == 'deberta':
                model, metrics, cm_fig, y_pred, feature_eng = train_deberta_model(
                    X_train, X_val, y_train, y_val,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    max_length=args.max_length,
                    learning_rate=args.learning_rate,
                    device=args.device,
                    num_labels=len(TOPIC_MAPPING),
                    random_state=args.random_state,
                    use_feature_engineering=args.use_feature_engineering,
                    feature_integration_mode=args.feature_integration_mode,
                    feature_weight=args.feature_weight
                )
            else:  # hybrid ensemble
                model, metrics, cm_fig, y_pred = train_hybrid_model(
                    X_train, X_val, y_train, y_val,
                    traditional_models=traditional_models,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    max_length=args.max_length,
                    learning_rate=args.learning_rate,
                    device=args.device,
                    num_labels=len(TOPIC_MAPPING),
                    random_state=args.random_state
                )
            
            # Log hyperparameters
            hyperparams = {
                'model_approach': args.model_approach,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'max_length': args.max_length,
                'learning_rate': args.learning_rate,
                'random_state': args.random_state
            }
            
            if args.model_approach == 'deberta':
                hyperparams['use_feature_engineering'] = args.use_feature_engineering
                hyperparams['feature_integration_mode'] = args.feature_integration_mode
                hyperparams['feature_weight'] = args.feature_weight
            elif args.model_approach == 'hybrid':
                hyperparams['traditional_models'] = args.traditional_models
            
            mlflow.log_params(hyperparams)
            
            # Log metrics
            log_metrics_to_mlflow(metrics)
            
            # Log confusion matrix
            log_figure_to_mlflow(cm_fig, "figures/confusion_matrix.png")
        except Exception as e:
            logger.error(f"Error during training or evaluation: {e}")
            # Continue to save model even if evaluation fails
        
        # Save and log model
        try:
            model_dir = os.path.join(args.model_dir, run_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # For DeBERTa or Hybrid models, use a custom save method
            if args.model_approach == 'deberta':
                model_path = os.path.join(model_dir, "deberta_model")
                model.save(model_path)
                
                # Save feature engineering preprocessor if available
                if feature_eng is not None:
                    feature_eng_path = os.path.join(model_dir, "feature_eng.joblib")
                    joblib.dump(feature_eng, feature_eng_path)
                    logger.info(f"Feature engineering preprocessor saved to {feature_eng_path}")
                
                # Log to MLflow - using a simpler approach due to complexity of transformer models
                mlflow.pyfunc.log_model(
                    artifact_path="models/deberta",
                    python_model=CustomPythonModel(),
                    artifacts={
                        "model_dir": model_path,
                        "feature_eng_path": os.path.join(model_dir, "feature_eng.joblib") if feature_eng is not None else None
                    },
                    code_path=["src/transformer_model.py"]
                )
                
                # Log feature engineering details
                if feature_eng is not None:
                    mlflow.log_param("feature_engineering", "enabled")
                    mlflow.log_artifact(os.path.join(model_dir, "feature_eng.joblib"), "feature_engineering")
            else:
                model_path = os.path.join(model_dir, "hybrid_model")
                model.save(model_path)
                
                # Log to MLflow - using a simpler approach due to complexity of transformer models
                mlflow.pyfunc.log_model(
                    artifact_path="models/hybrid",
                    python_model=CustomPythonModel(),
                    artifacts={"model_dir": model_path},
                    code_path=["src/transformer_model.py"]
                )
        except Exception as e:
            logger.error(f"Error saving model: {e}")
        
        try:
            # Generate predictions for submission
            logger.info("Generating predictions on test set...")
            try:
                test_preds = model.predict(test_df['Question'].values)
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                # Try batch prediction if normal prediction fails
                logger.info("Attempting batch prediction...")
                # Use the same batch size as in model training
                batch_size = args.batch_size
                all_preds = []
                for i in range(0, len(test_df), batch_size):
                    batch = test_df['Question'].values[i:i+batch_size]
                    batch_preds = model.predict(batch)
                    all_preds.extend(batch_preds)
                test_preds = all_preds
            
            submission_path = os.path.join(args.data_dir, f"submission_{args.model_approach}_enhanced.csv")
            create_submission_file(test_df, test_preds, submission_path)
            
            # Log submission file as artifact
            mlflow.log_artifact(submission_path, "submissions")
            logger.info(f"Submission file created and logged: {submission_path}")
        except Exception as e:
            logger.error(f"Error creating predictions or submission file: {e}")
        
        # Register model if requested
        if args.register_model:
            try:
                from mlflow.models.signature import infer_signature
                
                # Create model signature for better serving
                signature = infer_signature(
                    X_train[:1],  # Sample input
                    model.predict(X_train[:1])  # Sample output
                )
                
                # Register the model
                model_name = f"{args.model_name}-{args.model_approach}"
                mlflow.register_model(
                    f"runs:/{run_id}/models/{args.model_approach}",
                    model_name
                )
                
                # Transition to staging
                client = mlflow.tracking.MlflowClient()
                latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version,
                    stage="Staging"
                )
                
                logger.info(f"Model registered as {model_name} version {latest_version} and moved to Staging")
            except Exception as e:
                logger.error(f"Error registering model: {e}")
        
        # Print summary
        logger.info("Training complete!")
        if 'f1_micro' in metrics:
            logger.info(f"F1-micro score: {metrics['f1_micro']:.4f}")
        logger.info(f"MLflow run: {run_id}")
        logger.info(f"MLflow UI: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{run_id}")


# Custom Python model for MLflow
class CustomPythonModel(mlflow.pyfunc.PythonModel):
    """Custom Python model for MLflow model serialization."""
    
    def load_context(self, context):
        """Load model from artifacts."""
        import os
        from pathlib import Path
        
        # Load model
        model_dir = context.artifacts["model_dir"]
        model_type = Path(model_dir).name
        
        # Import transformer models
        from transformer_model import DebertaClassifier, HybridEnsembleClassifier
        
        # Load appropriate model type
        if "deberta" in model_type:
            self.model = DebertaClassifier.load(model_dir)
        elif "hybrid" in model_type:
            self.model = HybridEnsembleClassifier.load(model_dir)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load feature engineering preprocessor if available
        self.feature_eng = None
        if "feature_eng_path" in context.artifacts and context.artifacts["feature_eng_path"]:
            import joblib
            feature_eng_path = context.artifacts["feature_eng_path"]
            if os.path.exists(feature_eng_path):
                self.feature_eng = joblib.load(feature_eng_path)
                print(f"Loaded feature engineering preprocessor from {feature_eng_path}")
    
    def predict(self, context, model_input):
        """Make predictions."""
        import pandas as pd
        import numpy as np
        
        # Handle different input types
        if isinstance(model_input, pd.DataFrame):
            if "Question" in model_input.columns:
                texts = model_input["Question"].values
            else:
                texts = model_input.iloc[:, 0].values
        else:
            texts = model_input
        
        # Apply feature engineering if available
        if self.feature_eng is not None:
            # Feature engineering is handled internally by the model predict method
            # We don't need to manually apply it here
            pass
        
        # Make predictions
        return self.model.predict(texts)


if __name__ == "__main__":
    main() 