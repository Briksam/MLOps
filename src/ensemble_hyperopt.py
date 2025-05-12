import os
import sys
import logging
import argparse
import time
import mlflow
import joblib
from sklearn.ensemble import VotingClassifier, StackingClassifier
from mlflow.models.signature import infer_signature
from sklearn.svm import LinearSVC

# Add current directory to path to handle imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    setup_mlflow, evaluate_model, log_metrics_to_mlflow, 
     log_figure_to_mlflow,
    plot_confusion_matrix, save_model, register_model_to_registry, 
    transition_model_stage
)
from data_processing import load_data, prepare_data, download_nltk_resources, create_submission_file
from models import get_model_with_default_params

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create ensemble models from hyperparameter-tuned models')
    
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing the dataset files')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory containing tuned models')
    parser.add_argument('--ensemble-type', type=str, default='voting',
                       choices=['voting', 'stacking', 'both'],
                       help='Type of ensemble to create')
    parser.add_argument('--voting-strategy', type=str, default='soft',
                       choices=['soft', 'hard'],
                       help='Voting strategy for voting ensemble')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Size of test split')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--register-model', action='store_true',
                       help='Register model to MLflow registry')
    parser.add_argument('--model-name-prefix', type=str, default='math-topic-classifier-ensemble',
                       help='Prefix for registered model names')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for the MLflow run')
    parser.add_argument('--model-types', type=str, default='lr,rf,gb',
                       help='Comma-separated list of model types to include (lr,rf,gb,svm,nb)')
    return parser.parse_args()

def find_best_model_runs(client, experiment_id, model_type):
    """Find best hyperparameter-tuned runs for a specific model type."""
    # Query MLflow for runs of the specified model type
    query = f"params.model_type = '{model_type}' and tags.mlflow.runName LIKE 'hyperopt_%'"
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=query,
        order_by=["metrics.f1_micro DESC"]
    )
    
    if not runs:
        logger.warning(f"No hyperopt runs found for model type: {model_type}")
        return None
    
    # Return best run
    best_run = runs[0]
    logger.info(f"Found best run for {model_type}: {best_run.info.run_id} (F1: {best_run.data.metrics.get('f1_micro', 0):.4f})")
    return best_run

def load_model_from_run(client, run_id, artifact_path="models"):
    """Load model from MLflow run."""
    try:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model from run: {run_id}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from run {run_id}: {e}")
        return None

def create_voting_ensemble(models, voting='soft', weights=None):
    """Create a voting ensemble from tuned models."""
    # Check if any SVM models are included (they don't have predict_proba)
    has_svm = any('svm' in model_name.lower() or isinstance(model, LinearSVC) 
                 for model_name, model in models)
    
    # If SVM is included and soft voting requested, switch to hard voting
    if has_svm and voting == 'soft':
        logger.warning("LinearSVC doesn't support predict_proba required for soft voting.")
        logger.warning("Switching to hard voting since SVM models are included.")
        voting = 'hard'
    
    return VotingClassifier(
        estimators=models,
        voting=voting,
        weights=weights
    )

def create_stacking_ensemble(base_models, meta_model=None, cv=5, passthrough=False):
    """Create a stacking ensemble from tuned models."""
    if meta_model is None:
        # Default to gradient boosting as meta-learner
        meta_model = get_model_with_default_params('gb')
        
    return StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=cv,
        passthrough=passthrough
    )

def main():
    """Main function to create ensemble models from tuned models."""
    args = parse_args()
    
    # Force end any MLflow runs that might be active
    try:
        mlflow.end_run()
    except Exception as e:
        logger.warning(f"Error ending MLflow run: {e}")
    
    # Setup MLflow
    setup_mlflow()
    
    # Double check - end any active run again to be safe
    try:
        if mlflow.active_run():
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Ending active run: {run_id}")
            mlflow.end_run()
    except Exception as e:
        logger.warning(f"Error ending active run: {e}")
    
    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name("math-problem-classification")
    
    if not experiment:
        logger.error("Experiment not found. Please run some training runs first.")
        return
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}...")
    train_df, test_df = load_data(args.data_dir)
    
    # Prepare data
    logger.info("Preparing data...")
    X_train, X_val, y_train, y_val, vectorizer, preprocessor = prepare_data(
        train_df, 
        test_size=args.test_size, 
        random_state=args.random_state
    )
    
    # Prepare test data for predictions
    X_test = None
    if 'Question' in test_df.columns:
        logger.info("Preparing test data for predictions...")
        test_text = test_df['Question'].values
        if preprocessor:
            test_processed = preprocessor.preprocess(test_text)
            X_test = vectorizer.transform(test_processed)
        else:
            X_test = vectorizer.transform(test_text)
    
    # Parse model types to include
    model_types = [mt.strip() for mt in args.model_types.split(',')]
    logger.info(f"Including model types: {model_types}")
    
    # Find best runs for each model type
    tuned_models = []
    for model_type in model_types:
        best_run = find_best_model_runs(client, experiment.experiment_id, model_type)
        if best_run:
            model = load_model_from_run(client, best_run.info.run_id)
            if model:
                # Check if it's an SVM model and replacement is requested
                tuned_models.append((model_type, model))
    
    if len(tuned_models) < 2:
        logger.warning("Not enough tuned models found. Need at least 2 models for ensemble.")
        # Fall back to using default models with tuned parameters
        if 'lr' not in [m[0] for m in tuned_models] and 'lr' in model_types:
            logger.info("Adding default LR model")
            tuned_models.append(('lr', get_model_with_default_params('lr')))
        if 'rf' not in [m[0] for m in tuned_models] and 'rf' in model_types:
            logger.info("Adding default RF model")
            tuned_models.append(('rf', get_model_with_default_params('rf')))
        if 'gb' not in [m[0] for m in tuned_models] and 'gb' in model_types:
            logger.info("Adding default GB model")
            tuned_models.append(('gb', get_model_with_default_params('gb')))
    
    if len(tuned_models) < 2:
        logger.error("Cannot create ensemble with less than 2 models.")
        return
    
    # Generate run name if not provided
    if not args.run_name:
        run_name = f"ensemble_{'-'.join([m[0] for m in tuned_models])}_{int(time.time())}"
    else:
        run_name = args.run_name
    
    # Make absolutely sure there are no active runs
    try:
        if mlflow.active_run():
            mlflow.end_run()
    except Exception as e:
        logger.warning(f"Error ending active run: {e}")
    
    # Try to start a new run in a separate block
    run = None
    run_id = None
    try:
        # Use explicit run creation instead of context manager
        run = mlflow.start_run(run_name=run_name)
        run_id = run.info.run_id
        logger.info(f"Started MLflow Run ID: {run_id}")
        
        # Log ensemble parameters
        mlflow.log_params({
            'ensemble_type': args.ensemble_type,
            'model_types': args.model_types,
            'random_state': args.random_state,
            'model_count': len(tuned_models)
        })
        
        # Create directory for ensemble model
        model_dir = os.path.join(args.model_dir, f"ensemble_{run_id}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Create and evaluate ensembles based on specified type
        if args.ensemble_type in ['voting', 'both']:
            logger.info(f"Creating voting ensemble with {len(tuned_models)} models...")
            
            # Create and train voting ensemble
            voting_ensemble = create_voting_ensemble(
                tuned_models, 
                voting=args.voting_strategy
            )
            
            # Log additional voting parameters - use the actual voting strategy from the ensemble
            actual_voting_strategy = voting_ensemble.voting
            mlflow.log_param('voting_strategy', actual_voting_strategy)
            if actual_voting_strategy != args.voting_strategy:
                logger.info(f"Note: Voting strategy changed from '{args.voting_strategy}' to '{actual_voting_strategy}'")
            
            # Fit ensemble
            voting_ensemble.fit(X_train, y_train)
            
            # Evaluate ensemble
            voting_metrics, voting_y_pred = evaluate_model(voting_ensemble, X_val, y_val)
            
            # Log metrics with prefix
            voting_metrics_prefixed = {f"voting_{k}": v for k, v in voting_metrics.items()}
            log_metrics_to_mlflow(voting_metrics_prefixed)
            
            # Generate and log confusion matrix
            voting_cm_fig = plot_confusion_matrix(y_val, voting_y_pred)
            log_figure_to_mlflow(voting_cm_fig, "figures/voting_confusion_matrix.png")
            
            # Save model
            voting_model_path = save_model(voting_ensemble, model_dir, "voting_ensemble")
            
            # Log model
            mlflow.sklearn.log_model(
                voting_ensemble,
                artifact_path="voting_model",
                signature=infer_signature(X_train[:1], voting_ensemble.predict(X_train[:1])),
                input_example=X_train[:1]
            )
            
            # Generate predictions for submission
            try:
                logger.info("Generating predictions on test set for voting ensemble...")
                if X_test is not None:
                    test_preds = voting_ensemble.predict(X_test)
                else:
                    # If X_test wasn't prepared earlier, process questions directly
                    test_preds = voting_ensemble.predict(test_df['Question'].values)
                
                submission_path = os.path.join(args.data_dir, f"submission_ensemble_voting.csv")
                create_submission_file(test_df, test_preds, submission_path)
                
                # Log submission file as artifact
                mlflow.log_artifact(submission_path, "submissions")
                logger.info(f"Submission file created and logged: {submission_path}")
            except Exception as e:
                logger.error(f"Error creating predictions or submission file for voting ensemble: {e}")
                logger.error(f"Exception details: {str(e)}")
            
            # Register model if requested
            if args.register_model:
                voting_model_name = f"{args.model_name_prefix}-voting"
                logger.info(f"Registering voting ensemble as '{voting_model_name}'...")
                voting_registered = register_model_to_registry(run_id, voting_model_name, artifact_path="voting_model")
                transition_model_stage(voting_model_name, voting_registered.version, "Staging")
            
            # Print results
            logger.info(f"Voting Ensemble F1-micro: {voting_metrics['f1_micro']:.4f}")
        
        if args.ensemble_type in ['stacking', 'both']:
            logger.info(f"Creating stacking ensemble with {len(tuned_models)} base models...")
            
            # Create and train stacking ensemble
            stacking_ensemble = create_stacking_ensemble(
                tuned_models, 
                cv=5,
                passthrough=False
            )
            
            # Fit ensemble
            stacking_ensemble.fit(X_train, y_train)
            
            # Evaluate ensemble
            stacking_metrics, stacking_y_pred = evaluate_model(stacking_ensemble, X_val, y_val)
            
            # Log metrics with prefix
            # stacking_metrics_prefixed = {f"stacking_{k}": v for k, v in stacking_metrics.items()}
            log_metrics_to_mlflow(stacking_metrics)
            
            # Generate and log confusion matrix
            stacking_cm_fig = plot_confusion_matrix(y_val, stacking_y_pred)
            log_figure_to_mlflow(stacking_cm_fig, "figures/stacking_confusion_matrix.png")
            
            # Save model
            stacking_model_path = save_model(stacking_ensemble, model_dir, "stacking_ensemble")
            
            # Log model
            mlflow.sklearn.log_model(
                stacking_ensemble,
                artifact_path="stacking_model",
                signature=infer_signature(X_train[:1], stacking_ensemble.predict(X_train[:1])),
                input_example=X_train[:1]
            )
            
            # Generate predictions for submission
            try:
                logger.info("Generating predictions on test set for stacking ensemble...")
                if X_test is not None:
                    test_preds = stacking_ensemble.predict(X_test)
                else:
                    # If X_test wasn't prepared earlier, process questions directly
                    test_preds = stacking_ensemble.predict(test_df['Question'].values)
                
                submission_path = os.path.join(args.data_dir, f"submission_ensemble_stacking.csv")
                create_submission_file(test_df, test_preds, submission_path)
                
                # Log submission file as artifact
                mlflow.log_artifact(submission_path, "submissions")
                logger.info(f"Submission file created and logged: {submission_path}")
            except Exception as e:
                logger.error(f"Error creating predictions or submission file for stacking ensemble: {e}")
                logger.error(f"Exception details: {str(e)}")
            
            # Register model if requested
            if args.register_model:
                stacking_model_name = f"{args.model_name_prefix}-stacking"
                logger.info(f"Registering stacking ensemble as '{stacking_model_name}'...")
                stacking_registered = register_model_to_registry(run_id, stacking_model_name, artifact_path="stacking_model")
                transition_model_stage(stacking_model_name, stacking_registered.version, "Staging")
            
            # Print results
            logger.info(f"Stacking Ensemble F1-micro: {stacking_metrics['f1_micro']:.4f}")
        
        # Save vectorizer for later use with the ensemble
        vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
        joblib.dump(vectorizer, vectorizer_path)
        mlflow.log_artifact(vectorizer_path)
        
        # Log MLflow run info
        logger.info(f"Ensemble creation complete!")
        logger.info(f"MLflow run: {run_id}")
        if mlflow.active_run():
            logger.info(f"MLflow UI: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{run_id}")
        
    except Exception as e:
        logger.error(f"Error during ensemble creation: {e}")
        raise
    finally:
        # Always try to end the run in the finally block
        try:
            if mlflow.active_run():
                logger.info("Ending active run")
                mlflow.end_run()
        except Exception as e:
            logger.warning(f"Error ending active run: {e}")


# Add a cleanup function to be called at exit
def cleanup():
    """Clean up function to ensure we don't leave active runs."""
    try:
        if mlflow.active_run():
            logger.info("Ending active run at exit")
            mlflow.end_run()
    except Exception as e:
        logger.warning(f"Error in cleanup: {e}")

if __name__ == "__main__":
    # Register the cleanup function to be called at exit
    import atexit
    atexit.register(cleanup)
    
    try:
        main()
    finally:
        # Try one more time to clean up
        try:
            if mlflow.active_run():
                mlflow.end_run()
        except:
            pass 