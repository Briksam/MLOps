import os
import sys
import logging
import argparse
import time
import mlflow
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from functools import partial
from sklearn.model_selection import cross_val_score
from mlflow.models.signature import infer_signature

# Add current directory to path to handle imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    setup_mlflow, evaluate_model, log_metrics_to_mlflow, 
    log_params_to_mlflow,log_figure_to_mlflow,
    plot_confusion_matrix, save_model, register_model_to_registry, 
    transition_model_stage
)
from data_processing import load_data, prepare_data, download_nltk_resources, create_submission_file
from models import get_model, get_model_with_default_params, get_ensemble_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hyperparameter search spaces
PARAM_SPACES = {
    'lr': {
        'C': hp.loguniform('C', np.log(0.01), np.log(10.0)),
        'solver': hp.choice('solver', ['liblinear', 'saga']),
        'max_iter': hp.choice('max_iter', [1000, 2000, 3000]),
        'penalty': hp.choice('penalty', ['l1', 'l2']),
    },
    'rf': {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
        'max_depth': hp.quniform('max_depth', 5, 50, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
    },
    'gb': {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
    },
    'svm': {
        'C': hp.loguniform('C', np.log(0.01), np.log(10.0)),
        'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-3)),
        'dual': hp.choice('dual', [True, False]),
    },
    'nb': {
        'alpha': hp.loguniform('alpha', np.log(0.01), np.log(10.0)),
        'fit_prior': hp.choice('fit_prior', [True, False]),
    },
    # New parameter spaces for ensemble and stacking models
    'ensemble': {
        'voting': hp.choice('voting', ['soft', 'hard']),
        'weights': hp.choice('weights', [None, [1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]),
        # Model selection params - which models to include in ensemble
        'use_lr': hp.choice('use_lr', [True, False]),
        'use_rf': hp.choice('use_rf', [True, False]),
        'use_gb': hp.choice('use_gb', [True, False]),
        'use_svm': hp.choice('use_svm', [True, False]),
        'use_nb': hp.choice('use_nb', [True, False]),
    },
    'stacking': {
        # Base model selection
        'use_lr': hp.choice('use_lr', [True, False]),
        'use_rf': hp.choice('use_rf', [True, False]),
        'use_gb': hp.choice('use_gb', [True, False]),
        'use_svm': hp.choice('use_svm', [True, False]),
        'use_nb': hp.choice('use_nb', [True, False]),
        # Meta-learner selection
        'meta_learner': hp.choice('meta_learner', ['lr', 'rf', 'gb']),
        'cv': hp.choice('cv', [3, 5, 10]),
        'passthrough': hp.choice('passthrough', [True, False])
    }
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with MLflow tracking')
    
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing the dataset files')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save best model')
    parser.add_argument('--model-type', type=str, default='lr',
                       choices=['lr', 'rf', 'gb', 'svm', 'nb', 'ensemble', 'stacking'],
                       help='Type of model to tune')
    parser.add_argument('--max-evals', type=int, default=50,
                       help='Maximum number of evaluations for hyperparameter tuning')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Size of test split')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--register-model', action='store_true',
                       help='Register best model to MLflow registry')
    parser.add_argument('--model-name', type=str, default='math-topic-classifier-tuned',
                       help='Name for registered model')
    
    return parser.parse_args()

def create_ensemble_model_from_params(params):
    """Create ensemble model from hyperopt parameters."""
    # Determine which models to include based on parameters
    models = []
    
    if params.pop('use_lr', True):
        # Use optimized LR model (simplified - in production you might load best LR model)
        lr_params = {'C': 1.0, 'solver': 'liblinear', 'max_iter': 1000, 'penalty': 'l2'}
        models.append(('lr', get_model_with_default_params('lr', lr_params)))
    
    if params.pop('use_rf', True):
        # Use optimized RF model
        rf_params = {'n_estimators': 100, 'max_depth': 30}
        models.append(('rf', get_model_with_default_params('rf', rf_params)))
    
    if params.pop('use_gb', True):
        # Use optimized GB model
        gb_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}
        models.append(('gb', get_model_with_default_params('gb', gb_params)))
    
    if params.pop('use_svm', False):
        # Use optimized SVM model
        svm_params = {'C': 1.0, 'dual': False}
        models.append(('svm', get_model_with_default_params('svm', svm_params)))
    
    if params.pop('use_nb', False):
        # Use optimized NB model
        nb_params = {'alpha': 1.0}
        models.append(('nb', get_model_with_default_params('nb', nb_params)))
    
    # Ensure at least 2 models are included
    if len(models) < 2:
        # Default to LR and RF if no models selected
        models = [
            ('lr', get_model_with_default_params('lr')),
            ('rf', get_model_with_default_params('rf'))
        ]
    
    # Create and return the ensemble model
    return get_ensemble_model(models=models, **params)

def create_stacking_model_from_params(params):
    """Create stacking model from hyperopt parameters."""
    # Determine which models to include based on parameters
    base_models = []
    
    if params.pop('use_lr', True):
        lr_params = {'C': 1.0, 'solver': 'liblinear', 'max_iter': 1000}
        base_models.append(('lr', get_model_with_default_params('lr', lr_params)))
    
    if params.pop('use_rf', True):
        rf_params = {'n_estimators': 100, 'max_depth': 30}
        base_models.append(('rf', get_model_with_default_params('rf', rf_params)))
    
    if params.pop('use_gb', False):
        gb_params = {'n_estimators': 100, 'learning_rate': 0.1}
        base_models.append(('gb', get_model_with_default_params('gb', gb_params)))
    
    if params.pop('use_svm', False):
        svm_params = {'C': 1.0, 'dual': False}
        base_models.append(('svm', get_model_with_default_params('svm', svm_params)))
    
    if params.pop('use_nb', True):
        nb_params = {'alpha': 1.0}
        base_models.append(('nb', get_model_with_default_params('nb', nb_params)))
    
    # Ensure at least 2 base models are included
    if len(base_models) < 2:
        # Default to LR and RF if no models selected
        base_models = [
            ('lr', get_model_with_default_params('lr')),
            ('rf', get_model_with_default_params('rf'))
        ]
    
    # Get meta-learner
    meta_learner_type = params.pop('meta_learner', 'gb')
    meta_learner = get_model_with_default_params(meta_learner_type)
    
    # Create and return the stacking model
    from sklearn.ensemble import StackingClassifier
    return StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        **params
    )

def objective(params, model_type, X_train, y_train, cv_folds):
    """Objective function for hyperparameter optimization."""
    # Convert hyperopt parameter format to scikit-learn format
    params = {k: int(v) if isinstance(v, float) and v.is_integer() else v for k, v in params.items()}
    
    # Create model with specified parameters
    if model_type == 'ensemble':
        model = create_ensemble_model_from_params(params.copy())
    elif model_type == 'stacking':
        model = create_stacking_model_from_params(params.copy())
    else:
        model = get_model(model_type, **params)
    
    # Cross-validation score (negative to convert to minimization problem)
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=cv_folds, 
        scoring='f1_micro', 
        n_jobs=-1
    )
    
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    
    # Log parameters and metrics for this trial
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric('f1_micro_mean', mean_score)
        mlflow.log_metric('f1_micro_std', std_score)
    
    # Return result as a dictionary for hyperopt
    return {
        'loss': -mean_score,  # Negative because hyperopt minimizes
        'status': STATUS_OK,
        'params': params,
        'f1_micro_mean': mean_score,
        'f1_micro_std': std_score
    }

def main():
    """Main hyperparameter tuning function."""
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
    
    # Start MLflow parent run
    run_name = f"hyperopt_{args.model_type}_{int(time.time())}"
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        
        # Log tuning parameters
        mlflow.log_params({
            'model_type': args.model_type,
            'max_evals': args.max_evals,
            'cv_folds': args.cv_folds,
            'random_state': args.random_state
        })
        
        # Prepare data
        logger.info("Preparing data...")
        X_train, X_val, y_train, y_val, vectorizer, preprocessor = prepare_data(
            train_df, 
            test_size=args.test_size, 
            random_state=args.random_state
        )
        
        # Define hyperparameter space
        param_space = PARAM_SPACES.get(args.model_type, {})
        if not param_space:
            logger.error(f"No parameter space defined for model type: {args.model_type}")
            return
        
        # Create objective function with fixed parameters
        obj_func = partial(
            objective,
            model_type=args.model_type,
            X_train=X_train,
            y_train=y_train,
            cv_folds=args.cv_folds
        )
        
        # Initialize trials object to store results
        trials = Trials()
        
        # Run hyperparameter optimization
        logger.info(f"Starting hyperparameter tuning for {args.model_type} model...")
        logger.info(f"Running {args.max_evals} evaluations...")
        
        best = fmin(
            fn=obj_func,
            space=param_space,
            algo=tpe.suggest,
            max_evals=args.max_evals,
            trials=trials,
            rstate=np.random.default_rng(args.random_state)
        )
        
        # Get best parameters
        best_params = space_eval(param_space, best)
        
        # Convert float to int for integer parameters
        best_params = {k: int(v) if isinstance(v, float) and v.is_integer() else v 
                      for k, v in best_params.items()}
        
        logger.info(f"Best parameters: {best_params}")
        
        # Train final model with best parameters
        logger.info("Training final model with best parameters...")
        
        if args.model_type == 'ensemble':
            final_model = create_ensemble_model_from_params(best_params.copy())
        elif args.model_type == 'stacking':
            final_model = create_stacking_model_from_params(best_params.copy())
        else:
            final_model = get_model(args.model_type, **best_params)
            
        final_model.fit(X_train, y_train)
        
        # Evaluate final model
        metrics, y_pred = evaluate_model(final_model, X_val, y_val)
        
        # Log best parameters and metrics
        # Use a prefix for best parameters to avoid conflicts with parameters already logged during optimization
        best_params_prefixed = {f"best_{k}": v for k, v in best_params.items()}
        log_params_to_mlflow(best_params_prefixed)
        log_metrics_to_mlflow(metrics)
        
        # Generate and log confusion matrix
        cm_fig = plot_confusion_matrix(y_val, y_pred)
        log_figure_to_mlflow(cm_fig, "figures/confusion_matrix.png")
        
        # Save best model
        model_dir = os.path.join(args.model_dir, f"hyperopt_{run_id}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = save_model(final_model, model_dir, f"{args.model_type}_best_model")
        
        # Log best model to MLflow with vectorizer included
        # First save the vectorizer to a temporary file
        vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
        import joblib
        joblib.dump(vectorizer, vectorizer_path)
        
        # Log model and include vectorizer as an artifact
        artifact_path = "models"
        mlflow.sklearn.log_model(
            final_model,
            artifact_path=artifact_path,
            signature=infer_signature(X_train[:1], final_model.predict(X_train[:1])),
            input_example=X_train[:1]
        )
        mlflow.log_artifact(vectorizer_path, artifact_path)
        
        # Save vectorizer reference for model registry
        mlflow.log_param("vectorizer_path", os.path.join(artifact_path, "vectorizer.joblib"))
        
        # Register model if requested
        if args.register_model:
            logger.info(f"Registering model '{args.model_name}' to MLflow Registry...")
            registered_model = register_model_to_registry(run_id, args.model_name)
            
            # Transition to staging by default
            transition_model_stage(args.model_name, registered_model.version, "Staging")
            
            logger.info(f"Model registered as {args.model_name}, version: {registered_model.version}")
        
        # Log trials as a JSON artifact
        trials_path = os.path.join(model_dir, "trials.csv")
        trials_df = pd.DataFrame([t['result'] for t in trials.trials if 'result' in t])
        
        if not trials_df.empty:
            # Select relevant columns
            cols = ['params', 'f1_micro_mean', 'f1_micro_std']
            cols = [c for c in cols if c in trials_df.columns]
            trials_df = trials_df[cols]
            
            # Save to CSV
            trials_df.to_csv(trials_path, index=False)
            mlflow.log_artifact(trials_path, "trials")
        
        # Generate predictions on test set and create submission file
        try:
            logger.info("Generating predictions on test set...")
            
            # Check if test data needs preprocessing
            if hasattr(vectorizer, 'transform') and hasattr(preprocessor, 'preprocess'):
                logger.info("Preprocessing test data...")
                test_text = test_df['Question'].values
                test_processed = preprocessor.preprocess(test_text)
                X_test = vectorizer.transform(test_processed)
                test_preds = final_model.predict(X_test)
            else:
                # Direct prediction on raw text (some models may handle this)
                logger.info("Attempting direct prediction without preprocessing...")
                test_preds = final_model.predict(test_df['Question'].values)
                
            submission_path = os.path.join(args.data_dir, f"submission_{args.model_type}_hyperopt.csv")
            create_submission_file(test_df, test_preds, submission_path)
            
            # Log the submission file
            mlflow.log_artifact(submission_path, "submissions")
            logger.info(f"Submission file created and logged: {submission_path}")
        except Exception as e:
            logger.error(f"Error creating predictions or submission file: {e}")
            logger.error(f"Exception details: {str(e)}")
        
        # Print summary
        logger.info("Hyperparameter tuning complete!")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best F1-micro score: {metrics['f1_micro']:.4f}")
        logger.info(f"MLflow run: {run_id}")
        logger.info(f"MLflow UI: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{run_id}")

    # Ensure any active run is ended
    try:
        if mlflow.active_run():
            logger.info(f"Ending active run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()
    except Exception as e:
        logger.warning(f"Error ending active run: {e}")

if __name__ == "__main__":
    main() 