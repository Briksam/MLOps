import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import logging
import joblib
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "math-problem-classification"
ARTIFACT_PATH = "models"
MLRUNS_DIR = "mlruns"

# Topic mapping for reference
TOPIC_MAPPING = {
    0: "Algebra",
    1: "Geometry and Trigonometry",
    2: "Calculus and Analysis",
    3: "Probability and Statistics",
    4: "Number Theory",
    5: "Combinatorics and Discrete Math",
    6: "Linear Algebra",
    7: "Abstract Algebra and Topology"
}

def setup_mlflow(experiment_name=EXPERIMENT_NAME):
    """Set up MLflow tracking."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # End any active run that might be open
    try:
        if mlflow.active_run():
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Ending active run: {run_id}")
            mlflow.end_run()
    except Exception as e:
        logger.warning(f"Error handling active run: {e}")
    
    # Create experiment if it doesn't exist
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name} with ID: {experiment_id}")
        else:
            logger.info(f"Using existing experiment: {experiment_name} with ID: {experiment.experiment_id}")
        
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.error(f"Error setting up MLflow experiment: {e}")
        raise
        
    logger.info(f"MLflow setup complete. Tracking URI: {MLFLOW_TRACKING_URI}")
    return MLFLOW_TRACKING_URI

def save_model(model, model_dir, model_name):
    """Save model to disk."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    return model_path

def load_model(model_path):
    """Load model from disk."""
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model

def plot_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 8), cmap=plt.cm.Blues):
    """Plot confusion matrix."""
    if classes is None:
        classes = [TOPIC_MAPPING[i] for i in range(len(TOPIC_MAPPING))]
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='d', cmap=cmap)
    
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(classes, rotation=45)
    ax.yaxis.set_ticklabels(classes, rotation=45)
    
    plt.tight_layout()
    return plt.gcf()

def plot_class_distribution(df, label_col='label', figsize=(10, 6)):
    """Plot the distribution of classes in the dataset."""
    plt.figure(figsize=figsize)
    counts = df[label_col].value_counts().sort_index()
    
    # Map numeric labels to text labels
    labels = [TOPIC_MAPPING[i] for i in counts.index]
    
    ax = sns.barplot(x=[str(i) for i in counts.index], y=counts.values)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title('Class Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Math Topic')
    
    for i, count in enumerate(counts.values):
        ax.text(i, count + 5, str(count), ha='center')
    
    plt.tight_layout()
    return plt.gcf()

def evaluate_model(model, X_test, y_test, get_predictions=True):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='micro')
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        'f1_micro': f1,
        'accuracy': report['accuracy'],
        'weighted_precision': report['weighted avg']['precision'],
        'weighted_recall': report['weighted avg']['recall'],
        'weighted_f1': report['weighted avg']['f1-score']
    }
    
    # Add per-class metrics
    for i in range(len(TOPIC_MAPPING)):
        if str(i) in report:
            metrics[f'f1_class_{i}'] = report[str(i)]['f1-score']
    
    return metrics, y_pred if get_predictions else metrics

def log_metrics_to_mlflow(metrics):
    """Log metrics to MLflow."""
    for metric_name, metric_value in metrics.items():
        # Skip non-numeric values
        if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
            mlflow.log_metric(metric_name, metric_value)
    logger.info("Metrics logged to MLflow")

def log_params_to_mlflow(params):
    """Log parameters to MLflow."""
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)
    logger.info("Parameters logged to MLflow")

def log_model_to_mlflow(model, artifact_path=ARTIFACT_PATH, X_sample=None):
    """
    Log model to MLflow with signature and input example.
    
    Args:
        model: The trained model to log
        artifact_path: Path where model artifacts will be stored
        X_sample: Sample input data for signature inference (optional)
    """
    # Create a model signature if X_sample is provided
    signature = None
    input_example = None
    
    if X_sample is not None:
        # Generate sample prediction to get output shape/type
        y_sample = model.predict(X_sample[:1])
        
        # Infer model signature
        signature = infer_signature(X_sample, y_sample)
        
        # Use the first example as input example
        if isinstance(X_sample, np.ndarray):
            input_example = X_sample[0].toarray() if hasattr(X_sample[0], 'toarray') else X_sample[0]
        else:
            input_example = X_sample[:1]
        
        logger.info("Created model signature from sample data")
    
    # Log the model with signature and input example
    mlflow.sklearn.log_model(
        model, 
        artifact_path=artifact_path,
        signature=signature,
        input_example=input_example
    )
    
    logger.info(f"Model logged to MLflow at {artifact_path}")

def log_figure_to_mlflow(fig, artifact_path):
    """Log figure to MLflow."""
    temp_path = f"temp_{artifact_path.replace('/', '_')}.png"
    fig.savefig(temp_path)
    mlflow.log_artifact(temp_path, artifact_path)
    os.remove(temp_path)
    logger.info(f"Figure logged to MLflow at {artifact_path}")

def register_model_to_registry(run_id, model_name, artifact_path=ARTIFACT_PATH):
    """Register a model from a run to the MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    registered_model = mlflow.register_model(model_uri, model_name)
    logger.info(f"Model registered with name: {model_name}, version: {registered_model.version}")
    return registered_model

def transition_model_stage(model_name, version, stage):
    """Transition a model to a specific stage in the Model Registry."""
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    logger.info(f"Model {model_name} version {version} transitioned to {stage}") 