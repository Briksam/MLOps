import os
import sys
import logging
import argparse
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import joblib
from scipy.sparse import hstack, csr_matrix

# Add current directory to path to handle imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    setup_mlflow, evaluate_model, log_metrics_to_mlflow, 
    log_figure_to_mlflow, plot_confusion_matrix, TOPIC_MAPPING
)
from data_processing import (
    download_nltk_resources, TextPreprocessor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the clean_text and engineer_features functions from train_embeddings
def clean_text(text):
    """Clean and normalize text for better embeddings."""
    import re
    import string
    
    if not isinstance(text, str):
        return ""
    
    # Extract LaTeX expressions for special processing
    latex_expressions = re.findall(r'\$+.*?\$+', text, re.DOTALL)
    
    # Lower case
    text = text.lower()
    
    # Remove punctuations (except $ which is used for LaTeX)
    exclude = set(string.punctuation) - set('$')
    text = ''.join([char for char in text if char not in exclude])
    
    # Replace numeric values with token
    text = re.sub(r'\b\d+\b', ' NUM ', text)
    
    # Process LaTeX expressions
    latex_tokens = []
    for expr in latex_expressions:
        math_expr = expr.lower()
        
        # Detailed LaTeX pattern recognition
        if '\\matrix' in math_expr or '\\begin{matrix}' in math_expr or '\\begin{pmatrix}' in math_expr:
            latex_tokens.append('MATRIX_EXPRESSION')
        elif '\\int' in math_expr:
            latex_tokens.append('INTEGRAL_EXPRESSION')
        elif '\\sum' in math_expr:
            latex_tokens.append('SUMMATION_EXPRESSION')
        elif '\\frac' in math_expr or '/' in math_expr:
            latex_tokens.append('FRACTION_EXPRESSION')
        elif '\\sqrt' in math_expr:
            latex_tokens.append('SQUARE_ROOT_EXPRESSION')
        elif '\\pi' in math_expr:
            latex_tokens.append('PI_EXPRESSION')
        elif any(trig in math_expr for trig in ['\\sin', '\\cos', '\\tan', '\\cot', '\\sec', '\\csc']):
            latex_tokens.append('TRIGONOMETRY_EXPRESSION')
        elif '\\log' in math_expr:
            latex_tokens.append('LOGARITHM_EXPRESSION')
        elif '\\lim' in math_expr:
            latex_tokens.append('LIMIT_EXPRESSION')
        elif any(geom in math_expr for geom in ['\\triangle', '\\angle', '\\perp', '\\parallel']):
            latex_tokens.append('GEOMETRY_EXPRESSION')
        elif '\\forall' in math_expr or '\\exists' in math_expr:
            latex_tokens.append('LOGICAL_EXPRESSION')
        elif '\\in' in math_expr or '\\subset' in math_expr:
            latex_tokens.append('SET_THEORY_EXPRESSION')
        elif any(prob in math_expr for prob in ['\\mathbb{p}', 'probability', 'expectation']):
            latex_tokens.append('PROBABILITY_EXPRESSION')
        else:
            latex_tokens.append('MATH_EXPRESSION')
    
    # Add special tokens to the end of text
    if latex_tokens:
        text += ' ' + ' '.join(latex_tokens)
    
    return text

def engineer_features(data):
    """Extract additional features from the math problems."""
    logger.info("Engineering features...")
    
    from sklearn.preprocessing import StandardScaler
    
    # Check if input is a DataFrame or a list/array of questions
    if isinstance(data, pd.DataFrame) and 'Question' in data.columns:
        # Already a DataFrame with 'Question' column
        df = data
    else:
        # Convert to DataFrame
        df = pd.DataFrame({"Question": data})
    
    features = pd.DataFrame(index=df.index)
    
    # Math domain-specific keywords
    math_keywords = {
        'algebra': ['equation', 'solve', 'variable', 'polynomial', 'expression', 'factor', 'simplify', 'expand', 'linear', 'quadratic', 'roots', 'solution'],
        'geometry': ['triangle', 'circle', 'square', 'rectangle', 'polygon', 'angle', 'degree', 'radius', 'diameter', 'perimeter', 'area', 'volume', 'perpendicular', 'parallel', 'point', 'line', 'plane'],
        'calculus': ['derivative', 'integral', 'limit', 'function', 'continuity', 'differentiable', 'maximum', 'minimum', 'extrema', 'inflection', 'rate', 'change', 'integration'],
        'statistics': ['probability', 'random', 'distribution', 'mean', 'median', 'mode', 'variance', 'standard', 'deviation', 'expected', 'value', 'sample', 'population', 'hypothesis', 'test', 'confidence', 'interval'],
        'number_theory': ['prime', 'divisible', 'divisor', 'factor', 'multiple', 'remainder', 'modulo', 'congruence', 'gcd', 'lcm', 'diophantine'],
        'combinatorics': ['combination', 'permutation', 'arrange', 'factorial', 'choose', 'counting', 'principle', 'pigeonhole', 'induction'],
        'linear_algebra': ['matrix', 'vector', 'determinant', 'eigenvalue', 'eigenvector', 'linear', 'transformation', 'basis', 'span', 'dimension', 'subspace', 'orthogonal', 'projection', 'rank', 'kernel', 'range'],
        'abstract_algebra': ['group', 'ring', 'field', 'homomorphism', 'isomorphism', 'topology', 'set', 'subset', 'element', 'operation', 'associative', 'commutative', 'identity', 'inverse', 'closure']
    }
    
    def check_keyword_presence(text, keyword_list):
        """Check if any keyword from the list is present in the text"""
        text_lower = text.lower()
        for keyword in keyword_list:
            if f' {keyword} ' in f' {text_lower} ' or f'-{keyword}' in text_lower or f'{keyword}-' in text_lower:
                return 1
        return 0
    
    # Text length features
    features['text_length'] = df['Question'].apply(lambda x: len(str(x)))
    features['word_count'] = df['Question'].apply(lambda x: len(str(x).split()))
    
    # LaTeX features
    features['has_latex'] = df['Question'].apply(lambda x: 1 if '$' in str(x) else 0)
    features['latex_count'] = df['Question'].apply(lambda x: str(x).count('$') // 2)  # Each LaTeX expression has opening and closing $
    
    # Check for specific math symbols and concepts
    features['has_equation'] = df['Question'].apply(lambda x: 1 if '=' in str(x) or 'equation' in str(x).lower() else 0)
    features['has_integral'] = df['Question'].apply(lambda x: 1 if '\\int' in str(x) or 'integral' in str(x).lower() else 0)
    features['has_derivative'] = df['Question'].apply(lambda x: 1 if '\\frac{d' in str(x) or 'derivative' in str(x).lower() else 0)
    features['has_summation'] = df['Question'].apply(lambda x: 1 if '\\sum' in str(x) or 'sum' in str(x).lower() else 0)
    features['has_matrix'] = df['Question'].apply(lambda x: 1 if 'matrix' in str(x).lower() or '\\begin{matrix}' in str(x) else 0)
    features['has_probability'] = df['Question'].apply(lambda x: 1 if 'probability' in str(x).lower() else 0)
    features['has_functions'] = df['Question'].apply(lambda x: 1 if 'function' in str(x).lower() else 0)
    
    # Add keyword features for each math domain
    logger.info("  Adding domain-specific keyword features...")
    for domain, keywords in math_keywords.items():
        features[f'has_{domain}'] = df['Question'].apply(lambda x: check_keyword_presence(str(x), keywords))
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['text_length', 'word_count', 'latex_count']
    features[numerical_cols] = scaler.fit_transform(features[numerical_cols])
    
    return features

def prepare_embedding_features(questions, vectorizer=None, embedding_model_name="all-MiniLM-L6-v2"):
    """
    Prepare features for embedding models in the same format as during training.
    This ensures compatibility with the monitoring data.
    """
    logger.info("Preparing features for embedding model evaluation...")
    
    # Step 1: Clean text for embeddings
    cleaned_text = [clean_text(q) for q in questions]
    logger.info("Cleaned text for embeddings")
    
    # Step 2: Preprocess text for TF-IDF with the preprocessor
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    processed_text = preprocessor.preprocess(questions)
    logger.info(f"Preprocessed text successfully: {len(processed_text)} questions")
    
    # Step 3: Generate TF-IDF features
    if vectorizer is None:
        logger.info("No vectorizer provided, creating a new one")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(
            max_features=20000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents='unicode'
        )
        X_tfidf = vectorizer.fit_transform(processed_text)
    else:
        logger.info("Using provided vectorizer")
        X_tfidf = vectorizer.transform(processed_text)
    
    logger.info(f"Generated TF-IDF features with shape: {X_tfidf.shape}")
    
    # Step 4: Generate sentence embeddings
    try:
        from sentence_transformers import SentenceTransformer
        sentence_model = SentenceTransformer(embedding_model_name)
        embeddings = sentence_model.encode(cleaned_text)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        logger.warning("Will proceed with zero embeddings")
        # Create dummy embeddings if the model fails
        embeddings = np.zeros((len(cleaned_text), 384))  # Default embedding size
    
    # Step 5: Generate engineered features
    features_df = engineer_features(questions)
    logger.info(f"Generated engineered features with shape: {features_df.shape}")
    
    # Step 6: Convert to sparse matrices for combining
    embeddings_sparse = csr_matrix(embeddings)
    features_sparse = csr_matrix(features_df.values)
    
    # Combine all features in the same order as training
    X_combined = hstack([X_tfidf, embeddings_sparse, features_sparse]).tocsr()
    logger.info(f"Combined all features with shape: {X_combined.shape}")
    
    return X_combined

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Monitor model performance over time')
    
    parser.add_argument('--model-name', type=str, required=True,
                       help='Name of the registered model to monitor')
    parser.add_argument('--stage', type=str, default='Production',
                       choices=['Staging', 'Production', 'None', 'Archived'],
                       help='Stage of the model to monitor')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to the monitoring data with ground truth labels')
    parser.add_argument('--output-dir', type=str, default='monitoring',
                       help='Directory to save monitoring results')
    parser.add_argument('--vectorizer-path', type=str, default=None,
                       help='Path to the vectorizer joblib file, if not included in the model')
    
    return parser.parse_args()

def plot_performance_trend(metrics_history, output_path):
    """Plot performance trend over time."""
    df = pd.DataFrame(metrics_history)
    
    if df.empty or 'timestamp' not in df.columns:
        logger.warning("No valid metrics history data for plotting trend")
        return None
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot F1 and Accuracy trends
    ax = sns.lineplot(x='timestamp', y='f1_micro', data=df, marker='o', label='F1-micro')
    sns.lineplot(x='timestamp', y='accuracy', data=df, marker='s', label='Accuracy')
    
    # Add annotations for the latest points
    latest = df.iloc[-1]
    ax.annotate(f'F1: {latest["f1_micro"]:.4f}', 
                xy=(latest['timestamp'], latest['f1_micro']),
                xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    ax.annotate(f'Acc: {latest["accuracy"]:.4f}', 
                xy=(latest['timestamp'], latest['accuracy']),
                xytext=(10, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    # Set labels and title
    plt.title('Model Performance Trend')
    plt.ylabel('Score')
    plt.xlabel('Time')
    plt.ylim(0, 1.05)  # Fixed y-axis for consistency
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    
    return plt.gcf()

def plot_class_performance_trend(metrics_history, output_path):
    """Plot per-class performance trend over time."""
    df = pd.DataFrame(metrics_history)
    
    if df.empty or 'timestamp' not in df.columns:
        logger.warning("No valid metrics history data for plotting class trend")
        return None
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Extract class metrics columns
    class_cols = [col for col in df.columns if col.startswith('f1_class_')]
    
    if not class_cols:
        logger.warning("No per-class metrics available for plotting")
        return None
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot each class F1 score
    for col in class_cols:
        class_idx = int(col.split('_')[-1])
        class_name = TOPIC_MAPPING.get(class_idx, f"Class {class_idx}")
        sns.lineplot(x='timestamp', y=col, data=df, marker='o', label=class_name)
    
    # Set labels and title
    plt.title('Per-Class F1 Score Trend')
    plt.ylabel('F1 Score')
    plt.xlabel('Time')
    plt.ylim(0, 1.05)  # Fixed y-axis for consistency
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Handle legend (if too many classes, place outside)
    if len(class_cols) > 5:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    
    return plt.gcf()

def plot_drift_detection(current_metrics, historical_metrics, output_path):
    """Plot drift detection metrics."""
    if not historical_metrics:
        logger.warning("No historical metrics for drift detection")
        return None
    
    # Calculate drifts from historical average
    historical_df = pd.DataFrame(historical_metrics)
    
    if historical_df.empty:
        logger.warning("Empty historical metrics dataframe")
        return None
    
    # Get metrics we want to compare (exclude timestamp and metadata)
    metrics_to_compare = [
        'f1_micro', 'accuracy', 'weighted_precision', 
        'weighted_recall', 'weighted_f1'
    ]
    
    metrics_to_compare = [m for m in metrics_to_compare if m in historical_df.columns]
    
    if not metrics_to_compare:
        logger.warning("No metrics found for comparison")
        return None
    
    # Calculate historical averages
    historical_avg = historical_df[metrics_to_compare].mean()
    historical_std = historical_df[metrics_to_compare].std()
    
    # Calculate drift scores (z-scores)
    current_values = pd.Series({m: current_metrics.get(m, 0) for m in metrics_to_compare})
    drift_scores = (current_values - historical_avg) / historical_std.replace(0, 1)  # Avoid div by zero
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot drift scores
    ax = sns.barplot(x=drift_scores.index, y=drift_scores.values, palette='RdBu_r')
    
    # Add threshold lines
    plt.axhline(y=2, color='r', linestyle='--', alpha=0.7, label='Drift Threshold (2σ)')
    plt.axhline(y=-2, color='r', linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add value annotations
    for i, v in enumerate(drift_scores):
        ax.text(i, v + 0.1 * np.sign(v), f'{v:.2f}', ha='center', va='center', fontweight='bold')
    
    # Set labels and title
    plt.title('Model Drift Detection (Z-Scores)')
    plt.ylabel('Drift Score (σ)')
    plt.xlabel('Metric')
    
    # Improve x-axis labels
    metric_labels = {
        'f1_micro': 'F1-Micro', 
        'accuracy': 'Accuracy',
        'weighted_precision': 'Precision',
        'weighted_recall': 'Recall',
        'weighted_f1': 'F1-Weighted'
    }
    plt.xticks(range(len(metrics_to_compare)), [metric_labels.get(m, m) for m in metrics_to_compare])
    
    # Add legend
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    
    return plt.gcf()

def load_monitoring_history(history_file):
    """Load monitoring history from file."""
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            logger.info(f"Loaded monitoring history with {len(history)} records")
            return history
        except Exception as e:
            logger.error(f"Error loading monitoring history: {e}")
    
    logger.info("No monitoring history found. Starting new history.")
    return []

def save_monitoring_history(history, history_file):
    """Save monitoring history to file."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        logger.info(f"Saved monitoring history with {len(history)} records")
    except Exception as e:
        logger.error(f"Error saving monitoring history: {e}")

# Helper function to find vectorizer in MLflow artifacts
def find_vectorizer_in_mlflow(run_id):
    """Find and load vectorizer from MLflow artifacts."""
    client = mlflow.tracking.MlflowClient()
    
    # List of potential locations to check
    potential_paths = [
        "models/vectorizer.joblib",
        "vectorizer.joblib",
        "artifacts/vectorizer.joblib",
        "artifacts/tfidf_vectorizer.joblib",
        "models/tfidf_vectorizer.joblib",
        "tfidf_vectorizer.joblib"
    ]
    
    for path in potential_paths:
        try:
            logger.info(f"Trying to load vectorizer from {path}...")
            local_path = client.download_artifacts(run_id, path)
            vec = joblib.load(local_path)
            logger.info(f"Successfully loaded vectorizer from {path}")
            return vec
        except Exception as e:
            logger.warning(f"Could not load vectorizer from {path}: {e}")
    
    # If still not found, try listing all artifacts and find any joblib file
    try:
        logger.info("Attempting to locate vectorizer by searching all artifacts...")
        artifacts = client.list_artifacts(run_id)
        joblib_files = [a.path for a in artifacts if a.path.endswith('.joblib')]
        
        for joblib_path in joblib_files:
            try:
                logger.info(f"Trying to load potential vectorizer from {joblib_path}...")
                local_path = client.download_artifacts(run_id, joblib_path)
                potential_vectorizer = joblib.load(local_path)
                
                # Check if it looks like a vectorizer (has transform method)
                if hasattr(potential_vectorizer, 'transform'):
                    logger.info(f"Found vectorizer in {joblib_path}")
                    return potential_vectorizer
            except Exception as inner_e:
                logger.warning(f"Could not load or verify {joblib_path}: {inner_e}")
    except Exception as e:
        logger.warning(f"Error searching for vectorizer in artifacts: {e}")
    
    return None

# Get embedding model name used in training
def get_embedding_model_name(run_id):
    """Get embedding model name from MLflow artifacts."""
    client = mlflow.tracking.MlflowClient()
    
    # List of potential locations to check
    potential_paths = [
        "embedding_model.txt",
        "embedding_model/embedding_model.txt",
        "models/embedding_model.txt",
        "artifacts/embedding_model.txt"
    ]
    
    for path in potential_paths:
        try:
            logger.info(f"Trying to get embedding model name from {path}...")
            local_path = client.download_artifacts(run_id, path)
            with open(local_path, 'r') as f:
                embedding_model = f.read().strip()
            logger.info(f"Successfully retrieved embedding model name: {embedding_model}")
            return embedding_model
        except Exception as e:
            logger.warning(f"Could not get embedding model name from {path}: {e}")
    
    # Check if it's logged as a parameter
    try:
        run = client.get_run(run_id)
        if 'embedding_model' in run.data.params:
            embedding_model = run.data.params['embedding_model']
            logger.info(f"Found embedding model in run parameters: {embedding_model}")
            return embedding_model
    except Exception as e:
        logger.warning(f"Error retrieving run parameters: {e}")
    
    # Return default if not found
    default_embedding_model = "all-MiniLM-L6-v2"
    logger.warning(f"Could not find embedding model name, using default: {default_embedding_model}")
    return default_embedding_model

def main():
    """Main monitoring function."""
    args = parse_args()
    
    # Setup MLflow with a dedicated monitoring experiment
    mlflow_tracking_uri = setup_mlflow()
    
    # Create or get dedicated monitoring experiment
    client = mlflow.tracking.MlflowClient()
    monitoring_experiment_name = "model-monitoring"
    
    try:
        experiment = client.get_experiment_by_name(monitoring_experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing monitoring experiment: {monitoring_experiment_name} with ID: {experiment_id}")
        else:
            experiment_id = client.create_experiment(monitoring_experiment_name)
            logger.info(f"Created new monitoring experiment: {monitoring_experiment_name} with ID: {experiment_id}")
    except Exception as e:
        logger.error(f"Error setting up monitoring experiment: {e}")
        # Fall back to default experiment
        experiment_id = "0"
        logger.warning(f"Falling back to default experiment with ID: {experiment_id}")
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create history file path
    history_file = os.path.join(args.output_dir, f"{args.model_name}_{args.stage}_history.json")
    
    # Start MLflow run in the monitoring experiment
    run_name = f"monitoring_{args.model_name}_{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        
        # Log parameters
        mlflow.log_params({
            'model_name': args.model_name,
            'stage': args.stage,
            'data_path': args.data_path
        })
        
        # Load model
        model_uri = f"models:/{args.model_name}/{args.stage}"
        logger.info(f"Loading model from {model_uri}...")
        model = mlflow.sklearn.load_model(model_uri)
        
        # Get model details to find original run_id
        model_details = client.get_latest_versions(args.model_name, stages=[args.stage])
        original_run_id = None
        
        if model_details and len(model_details) > 0:
            original_run_id = model_details[0].run_id
            logger.info(f"Found original run_id: {original_run_id} for model {args.model_name}")
        
        # Load monitoring data
        logger.info(f"Loading monitoring data from {args.data_path}...")
        try:
            data = pd.read_csv(args.data_path)
            logger.info(f"Loaded monitoring data: {data.shape}")
        except Exception as e:
            logger.error(f"Error loading monitoring data: {e}")
            return
        
        # Check if data has required columns
        if 'Question' not in data.columns or 'label' not in data.columns:
            logger.error("Monitoring data must have 'Question' and 'label' columns")
            return
        
        # Load vectorizer if provided
        vectorizer = None
        if args.vectorizer_path and os.path.exists(args.vectorizer_path):
            logger.info(f"Loading vectorizer from {args.vectorizer_path}...")
            try:
                vectorizer = joblib.load(args.vectorizer_path)
            except Exception as e:
                logger.error(f"Error loading vectorizer: {e}")
                vectorizer = None
        elif hasattr(model, 'named_steps') and 'vectorizer' in model.named_steps:
            vectorizer = model.named_steps['vectorizer']
            logger.info("Extracted vectorizer from model pipeline")
        
        # Try to find the original vectorizer in MLflow artifacts if not found
        if vectorizer is None and original_run_id:
            logger.info(f"Trying to find original vectorizer from MLflow run {original_run_id}...")
            vectorizer = find_vectorizer_in_mlflow(original_run_id)
            if vectorizer:
                logger.info("Successfully loaded original vectorizer from MLflow artifacts")
        
        # Determine if this is an embedding model
        is_embedding_model = "embedding" in args.model_name.lower()
        
        # Get embedding model name if this is an embedding model
        embedding_model_name = "all-MiniLM-L6-v2"  # Default
        if is_embedding_model and original_run_id:
            embedding_model_name = get_embedding_model_name(original_run_id)
            logger.info(f"Using embedding model: {embedding_model_name}")
        
        # Prepare data
        X_text = data['Question'].values
        y_true = data['label'].values
        
        # Process data based on model type
        if is_embedding_model:
            logger.info("Detected embedding model, using train_embeddings.py preprocessing logic")
            
            # Step 1: Clean text for embeddings (exactly as in train_embeddings.py)
            logger.info("Cleaning text...")
            data['cleaned_text'] = data['Question'].apply(clean_text)
            
            # Step 2: Generate engineered features
            logger.info("Generating engineered features...")
            features_df = engineer_features(data)
            
            # Step 3: Generate TF-IDF features
            logger.info("Generating TF-IDF features...")
            
            # If we have a vectorizer, use it. Otherwise, create a new one.
            if vectorizer is None:
                from sklearn.feature_extraction.text import TfidfVectorizer
                logger.info("Creating new TF-IDF vectorizer...")
                vectorizer = TfidfVectorizer(
                    max_features=20000,
                    min_df=2,
                    max_df=0.9,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    strip_accents='unicode'
                )
                X_tfidf = vectorizer.fit_transform(data['cleaned_text'])
            else:
                logger.info("Using original vectorizer from training...")
                X_tfidf = vectorizer.transform(data['cleaned_text'])
            
            logger.info(f"TF-IDF features shape: {X_tfidf.shape}")
            
            # Step 4: Generate sentence embeddings
            logger.info("Generating sentence embeddings...")
            try:
                from sentence_transformers import SentenceTransformer
                sentence_model = SentenceTransformer(embedding_model_name)
                embeddings = sentence_model.encode(data['cleaned_text'].values)
                logger.info(f"Embedding shape: {embeddings.shape}")
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                # Use default embedding dimension if we can't generate embeddings
                embedding_dim = 384
                embeddings = np.zeros((len(data), embedding_dim))
                logger.warning(f"Using zero embeddings with shape: {embeddings.shape}")
            
            # Step 5: Convert to sparse matrices for combining
            from scipy.sparse import hstack, csr_matrix
            embeddings_sparse = csr_matrix(embeddings)
            features_sparse = csr_matrix(features_df.values)
            
            # Step 6: Combine all features
            logger.info("Combining features...")
            X = hstack([X_tfidf, embeddings_sparse, features_sparse]).tocsr()
            logger.info(f"Combined features shape: {X.shape}")
            
            # Handle feature count mismatch
            if hasattr(model, 'coef_'):
                expected_feature_count = model.coef_.shape[1]
                actual_feature_count = X.shape[1]
                
                if expected_feature_count != actual_feature_count:
                    logger.warning(f"Feature count mismatch: got {actual_feature_count}, expected {expected_feature_count}")
                    
                    # Option 1: If we have too few features, pad with zeros
                    if actual_feature_count < expected_feature_count:
                        logger.info(f"Padding features with zeros to match expected count")
                        padding = csr_matrix((X.shape[0], expected_feature_count - actual_feature_count))
                        X = hstack([X, padding]).tocsr()
                        logger.info(f"New feature shape after padding: {X.shape}")
                    
                    # Option 2: If we have too many features, truncate
                    elif actual_feature_count > expected_feature_count:
                        logger.info(f"Truncating features to match expected count")
                        X = X[:, :expected_feature_count]
                        logger.info(f"New feature shape after truncation: {X.shape}")
            
        else:
            # Traditional model preprocessing
            # Preprocess text
            preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
            X_processed = preprocessor.preprocess(X_text)
            
            # Vectorize if needed
            if vectorizer:
                X = vectorizer.transform(X_processed)
            else:
                X = X_processed
        
        # Evaluate model
        metrics, y_pred = evaluate_model(model, X, y_true)
        
        # Add timestamp to metrics
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['run_id'] = run_id
        
        # Load monitoring history
        history = load_monitoring_history(history_file)
        
        # Check if we have enough history for drift detection
        drift_detected = False
        drift_metrics = {}
        
        if len(history) >= 3:
            # Plot performance trend
            trend_path = os.path.join(args.output_dir, f"{args.model_name}_{args.stage}_trend.png")
            trend_fig = plot_performance_trend(history + [metrics], trend_path)
            if trend_fig:
                log_figure_to_mlflow(trend_fig, "monitoring/performance_trend.png")
            
            # Plot class performance trend
            class_trend_path = os.path.join(args.output_dir, f"{args.model_name}_{args.stage}_class_trend.png")
            class_trend_fig = plot_class_performance_trend(history + [metrics], class_trend_path)
            if class_trend_fig:
                log_figure_to_mlflow(class_trend_fig, "monitoring/class_performance_trend.png")
            
            # Plot drift detection
            drift_path = os.path.join(args.output_dir, f"{args.model_name}_{args.stage}_drift.png")
            drift_fig = plot_drift_detection(metrics, history, drift_path)
            if drift_fig:
                log_figure_to_mlflow(drift_fig, "monitoring/drift_detection.png")
            
            # Calculate if drift is detected (z-score > 2 for any key metric)
            historical_df = pd.DataFrame(history)
            key_metrics = ['f1_micro', 'accuracy', 'weighted_f1']
            key_metrics = [m for m in key_metrics if m in historical_df.columns]
            
            if key_metrics:
                # Calculate historical averages and std devs
                historical_avg = historical_df[key_metrics].mean()
                historical_std = historical_df[key_metrics].std().replace(0, 1)  # Avoid div by zero
                
                # Current values
                current_values = pd.Series({m: metrics.get(m, 0) for m in key_metrics})
                
                # Calculate z-scores
                z_scores = (current_values - historical_avg) / historical_std
                
                # Check for drift
                drift_detected = any(abs(z) > 2 for z in z_scores)
                
                # Store drift metrics
                drift_metrics = {
                    'drift_detected': drift_detected,
                    'z_scores': z_scores.to_dict()
                }
                
                # Log drift metrics
                mlflow.log_metrics({f"drift_{k}": v for k, v in z_scores.items()})
                mlflow.log_param("drift_detected", drift_detected)
        
        # Generate and log confusion matrix
        cm_fig = plot_confusion_matrix(y_true, y_pred)
        cm_path = os.path.join(args.output_dir, f"{args.model_name}_{args.stage}_cm.png")
        cm_fig.savefig(cm_path)
        log_figure_to_mlflow(cm_fig, "monitoring/confusion_matrix.png")
        
        # Log metrics
        log_metrics_to_mlflow(metrics)
        
        # Add metrics to history
        history_entry = {**metrics, **drift_metrics}
        history.append(history_entry)
        
        # Save updated history
        save_monitoring_history(history, history_file)
        
        # Log monitoring summary
        summary = {
            'model_name': args.model_name,
            'model_stage': args.stage,
            'monitoring_timestamp': datetime.now().isoformat(),
            'data_samples': len(data),
            'f1_micro': metrics['f1_micro'],
            'accuracy': metrics['accuracy'],
            'drift_detected': drift_detected,
            'history_length': len(history)
        }
        
        # Log summary as artifact
        summary_path = os.path.join(args.output_dir, f"{args.model_name}_{args.stage}_latest.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        mlflow.log_artifact(summary_path, "monitoring")
        
        # Print summary
        logger.info(f"Monitoring complete for {args.model_name} ({args.stage})")
        logger.info(f"F1-micro score: {metrics['f1_micro']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        
        if drift_detected:
            logger.warning("MODEL DRIFT DETECTED! Consider retraining or investigating.")
        
        logger.info(f"MLflow run: {run_id}")
        logger.info(f"MLflow UI: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{run_id}")

if __name__ == "__main__":
    main() 