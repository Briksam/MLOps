import os
import sys
import logging
import argparse
import time
import joblib
import numpy as np
import pandas as pd
import mlflow
from tqdm import tqdm
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# Add current directory to path to handle imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    setup_mlflow, log_figure_to_mlflow, plot_confusion_matrix
)
from data_processing import (
    load_data, download_nltk_resources, create_submission_file
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train embedding-based math problem classification models')
    
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Directory containing the dataset files')
    parser.add_argument('--model-dir', type=str, default='models',
                      help='Directory to save trained models')
    parser.add_argument('--run-name', type=str, default=None,
                      help='Name for the MLflow run')
    parser.add_argument('--experiment-name', type=str, default='sentence-embeddings',
                      help='Name for the MLflow experiment')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Size of validation split')
    parser.add_argument('--random-state', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--register-model', action='store_true',
                      help='Register model to MLflow registry')
    parser.add_argument('--model-name', type=str, default='math-topic-classifier-embeddings',
                      help='Name for registered model')
    parser.add_argument('--embedding-model', type=str, default='all-MiniLM-L6-v2',
                      help='Name of the sentence transformer model to use')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for embedding generation')
    parser.add_argument('--n-folds', type=int, default=5,
                      help='Number of cross-validation folds')
    
    return parser.parse_args()

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

def engineer_features(df):
    """Extract additional features from the math problems."""
    logger.info("Engineering features...")
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

def generate_embeddings(texts, model_name, batch_size=32):
    """Generate sentence embeddings for the given texts."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.info("Installing sentence-transformers...")
        os.system("pip install -q sentence-transformers")
        from sentence_transformers import SentenceTransformer
    
    logger.info(f"Using embedding model: {model_name}")
    sentence_model = SentenceTransformer(model_name)
    
    embeddings = []
    logger.info("Generating embeddings in batches...")
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size].tolist()
        batch_embeddings = sentence_model.encode(batch)
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)

def train_cross_val_models(X, y, n_folds=5, random_state=42):
    """Train multiple models with cross-validation and return the best one."""
    logger.info("Training multiple models with cross-validation...")
    
    models = {
        'LogisticRegression': LogisticRegression(C=10, solver='liblinear', max_iter=1000, random_state=random_state),
        'LinearSVC': CalibratedClassifierCV(LinearSVC(C=1.0, random_state=random_state), cv=3),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=random_state)
    }
    
    # Perform cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_results = {}
    
    for name, model in models.items():
        logger.info(f"Cross-validating {name}...")
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            start_time = time.time()
            model.fit(X_train_fold, y_train_fold)
            train_time = time.time() - start_time
            
            # Get predictions
            y_val_pred = model.predict(X_val_fold)
            
            # Calculate F1 score
            f1 = f1_score(y_val_fold, y_val_pred, average='micro')
            cv_scores.append(f1)
            
            logger.info(f"  Fold {fold+1}: F1 = {f1:.4f} (Time: {train_time:.2f}s)")
        
        # Calculate mean CV score
        mean_f1 = np.mean(cv_scores)
        cv_results[name] = {
            'scores': cv_scores,
            'mean_f1': mean_f1,
            'model': models[name]  # Store the model class
        }
        logger.info(f"  Mean F1 for {name}: {mean_f1:.4f}")
    
    # Select best model
    best_model_name = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_f1'])
    best_mean_f1 = cv_results[best_model_name]['mean_f1']
    
    logger.info(f"Best model: {best_model_name} with mean F1: {best_mean_f1:.4f}")
    
    # Create a fresh instance of the best model
    if best_model_name == 'LogisticRegression':
        best_model = LogisticRegression(C=10, solver='liblinear', max_iter=1000, random_state=random_state)
    elif best_model_name == 'LinearSVC':
        best_model = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=random_state), cv=3)
    else:  # RandomForest
        best_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=random_state)
    
    # Train on the entire dataset
    logger.info(f"Training {best_model_name} on the entire dataset...")
    best_model.fit(X, y)
    
    return best_model, best_model_name, cv_results

def save_embedding_model(model, model_name, vectorizer, features_path, embedding_model_name, run_id, output_dir='models'):
    """Save the trained model and associated artifacts."""
    model_dir = os.path.join(output_dir, run_id, "embedding_model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    
    # Save vectorizer
    if vectorizer is not None:
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
        joblib.dump(vectorizer, vectorizer_path)
    
    # Save embedding model name
    with open(os.path.join(model_dir, "embedding_model.txt"), 'w') as f:
        f.write(embedding_model_name)
    
    # Save features path
    with open(os.path.join(model_dir, "features_path.txt"), 'w') as f:
        f.write(features_path)
    
    logger.info(f"Model and artifacts saved to {model_dir}")
    return model_dir

def main():
    """Main function to train embedding-based models."""
    args = parse_args()
    
    # Setup MLflow with custom experiment name
    setup_mlflow(experiment_name=args.experiment_name)
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}...")
    train_df, test_df = load_data(args.data_dir)
    
    # Clean text
    logger.info("Cleaning text...")
    train_df['cleaned_text'] = train_df['Question'].apply(clean_text)
    test_df['cleaned_text'] = test_df['Question'].apply(clean_text)
    
    # Create run name if not provided
    run_name = args.run_name or f"embedding_{args.embedding_model}_{int(time.time())}"
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        
        # Log parameters
        mlflow.log_param("embedding_model", args.embedding_model)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("train_examples", len(train_df))
        mlflow.log_param("test_examples", len(test_df))
        
        try:
            # Generate engineered features
            train_features = engineer_features(train_df)
            test_features = engineer_features(test_df)
            
            # Generate TF-IDF features
            logger.info("Generating TF-IDF features...")
            tfidf_vectorizer = TfidfVectorizer(
                max_features=20000,
                min_df=2,
                max_df=0.9,
                ngram_range=(1, 2),
                sublinear_tf=True,
                strip_accents='unicode'
            )
            X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['cleaned_text'])
            X_test_tfidf = tfidf_vectorizer.transform(test_df['cleaned_text'])
            logger.info(f"TF-IDF features shape: {X_train_tfidf.shape}")
            
            # Generate sentence embeddings
            logger.info("Generating sentence embeddings...")
            train_embeddings = generate_embeddings(
                train_df['cleaned_text'],
                args.embedding_model,
                args.batch_size
            )
            test_embeddings = generate_embeddings(
                test_df['cleaned_text'],
                args.embedding_model,
                args.batch_size
            )
            logger.info(f"Embedding shape: {train_embeddings.shape}")
            
            # Convert to sparse matrices for combining
            train_embeddings_sparse = csr_matrix(train_embeddings)
            test_embeddings_sparse = csr_matrix(test_embeddings)
            train_features_sparse = csr_matrix(train_features.values)
            test_features_sparse = csr_matrix(test_features.values)
            
            # Combine all features
            logger.info("Combining features...")
            X_train_combined = hstack([X_train_tfidf, train_embeddings_sparse, train_features_sparse]).tocsr()
            X_test_combined = hstack([X_test_tfidf, test_embeddings_sparse, test_features_sparse]).tocsr()
            logger.info(f"Combined features shape: {X_train_combined.shape}")
            
            # Save combined features
            feature_dir = os.path.join(args.model_dir, run_id, "features")
            os.makedirs(feature_dir, exist_ok=True)
            train_features_path = os.path.join(feature_dir, "train_features_combined.joblib")
            test_features_path = os.path.join(feature_dir, "test_features_combined.joblib")
            joblib.dump(X_train_combined, train_features_path)
            joblib.dump(X_test_combined, test_features_path)
            logger.info(f"Features saved to {feature_dir}")
            
            # Train models
            y_train = train_df['label']
            best_model, best_model_name, cv_results = train_cross_val_models(
                X_train_combined, 
                y_train,
                n_folds=args.n_folds,
                random_state=args.random_state
            )
            
            # Log cross-validation results
            for model_name, result in cv_results.items():
                mlflow.log_metric(f"{model_name}_mean_f1", result['mean_f1'])
                for i, score in enumerate(result['scores']):
                    mlflow.log_metric(f"{model_name}_fold{i+1}_f1", score)
            
            # Log the best model's mean F1 as f1_micro for easier comparison with other experiments
            best_mean_f1 = cv_results[best_model_name]['mean_f1']
            mlflow.log_metric("f1_micro", best_mean_f1)
            mlflow.log_param("best_model", best_model_name)
            logger.info(f"Logged best model's F1 score ({best_mean_f1:.4f}) as 'f1_micro' in MLflow")
            
            # Evaluate on test set if labels are available
            if 'label' in test_df.columns:
                logger.info("Evaluating on test set...")
                y_test = test_df['label']
                y_pred = best_model.predict(X_test_combined)
                test_f1 = f1_score(y_test, y_pred, average='micro')
                logger.info(f"Test F1 score: {test_f1:.4f}")
                mlflow.log_metric("test_f1_micro", test_f1)
                
                # Generate confusion matrix
                cm_fig = plot_confusion_matrix(y_test, y_pred)
                log_figure_to_mlflow(cm_fig, "figures/confusion_matrix_test.png")
            
            # Save model and artifacts
            model_dir = save_embedding_model(
                best_model,
                best_model_name,
                tfidf_vectorizer,
                train_features_path,
                args.embedding_model,
                run_id,
                args.model_dir
            )
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                best_model,
                f"models/{best_model_name}",
                registered_model_name=args.model_name if args.register_model else None
            )
            
            # Log vectorizer
            mlflow.log_artifact(os.path.join(model_dir, "tfidf_vectorizer.joblib"), "artifacts")
            
            # Generate predictions for submission
            logger.info("Generating predictions on test set...")
            test_preds = best_model.predict(X_test_combined)
            submission_path = os.path.join(args.data_dir, f"submission_embedding_{args.embedding_model}.csv")
            create_submission_file(test_df, test_preds, submission_path)
            
            # Log submission file as artifact
            mlflow.log_artifact(submission_path, "submissions")
            logger.info(f"Submission file logged to MLflow at submissions/submission_embedding_{args.embedding_model}.csv")
            
            # Print summary
            logger.info("Training complete!")
            logger.info(f"Best model: {best_model_name}")
            logger.info(f"Best CV F1 score: {cv_results[best_model_name]['mean_f1']:.4f}")
            logger.info(f"MLflow run: {run_id}")
            logger.info(f"MLflow UI: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{run_id}")
        
        except Exception as e:
            logger.error(f"Error during training: {e}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 