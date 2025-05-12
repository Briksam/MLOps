import pandas as pd
import os
import re
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
import mlflow
from utils import log_figure_to_mlflow, plot_class_distribution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK resources
def download_nltk_resources():
    """Download required NLTK resources."""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            download(resource)
            logger.info(f"NLTK resource '{resource}' downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading NLTK resource '{resource}': {e}")
            logger.warning(f"Some text processing features may not work without '{resource}'")

class TextPreprocessor:
    """Text preprocessing class for math problem classification."""
    
    def __init__(self, remove_stopwords=True, lemmatize=True):
        """Initialize the text preprocessor."""
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else None
        
        # Math-specific stopwords to keep
        self.math_terms = {
            'sum', 'difference', 'product', 'quotient', 'equals',
            'equal', 'equation', 'function', 'variable', 'solve',
            'find', 'calculate', 'evaluate', 'simplify', 'factor',
            'expand', 'graph', 'prove', 'theorem', 'matrix',
            'vector', 'determinant', 'eigenvalue', 'derivative',
            'integral', 'limit', 'probability', 'distribution'
        }
        
        if self.stop_words:
            # Remove math-specific terms from stopwords
            self.stop_words = self.stop_words - self.math_terms
    
    def clean_text(self, text):
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep math operators
        text = re.sub(r'[^a-zA-Z0-9\s+\-*/^()={}<>≤≥≠∫∂∑∏√]', ' ', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text."""
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text)
        
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words or t in self.math_terms]
        
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return tokens
    
    def preprocess(self, texts):
        """Preprocess a list of texts."""
        return [' '.join(self.tokenize(text)) for text in texts]

def load_data(data_dir='data'):
    """Load training and test data."""
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory '{data_dir}' does not exist. Creating it.")
        os.makedirs(data_dir, exist_ok=True)
    
    # Check if training file exists
    if not os.path.exists(train_path):
        logger.error(f"Training file not found at '{train_path}'")
        raise FileNotFoundError(f"Training file not found at '{train_path}'")
    
    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded training data: {train_df.shape}")
    
    # Check if test file exists
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        logger.info(f"Loaded test data: {test_df.shape}")
    else:
        logger.warning(f"Test file not found at '{test_path}'. Will use a split of training data.")
        test_df = None
    
    return train_df, test_df

def prepare_data(train_df, test_df=None, test_size=0.2, random_state=42, 
                 mlflow_logging=True, vectorizer_params=None):
    """
    Prepare data for model training.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame (optional)
        test_size: Size of validation split if test_df is None
        random_state: Random seed for reproducibility
        mlflow_logging: Whether to log to MLflow
        vectorizer_params: Dictionary of TfidfVectorizer parameters
        
    Returns:
        X_train_vectorized, X_val_vectorized, y_train, y_val, vectorizer, preprocessor
    """
    # Initialize preprocessor
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    
    # Log data distribution
    if mlflow_logging:
        dist_fig = plot_class_distribution(train_df)
        log_figure_to_mlflow(dist_fig, "figures/class_distribution.png")
    
    # If no test_df is provided, split the training data
    if test_df is None:
        if 'Question' not in train_df.columns:
            logger.error("Column 'Question' not found in training data")
            raise ValueError("Column 'Question' not found in training data")
            
        if 'label' not in train_df.columns:
            logger.error("Column 'label' not found in training data")
            raise ValueError("Column 'label' not found in training data")
            
        X = train_df['Question'].values
        y = train_df['label'].values
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Split data: train={X_train.shape[0]}, validation={X_val.shape[0]}")
    else:
        # Verify columns exist in both DataFrames
        if 'Question' not in train_df.columns or 'Question' not in test_df.columns:
            logger.error("Column 'Question' not found in data")
            raise ValueError("Column 'Question' not found in data")
            
        if 'label' not in train_df.columns:
            logger.error("Column 'label' not found in training data")
            raise ValueError("Column 'label' not found in training data")
            
        X_train = train_df['Question'].values
        y_train = train_df['label'].values
        X_val = test_df['Question'].values
        
        if 'label' in test_df.columns:
            y_val = test_df['label'].values
        else:
            y_val = None
            logger.warning("No 'label' column found in test data. Will proceed without test labels.")
        
        logger.info(f"Using separate test set: train={X_train.shape[0]}, test={X_val.shape[0]}")
    
    # Preprocess text
    logger.info("Preprocessing text data...")
    X_train_processed = preprocessor.preprocess(X_train)
    X_val_processed = preprocessor.preprocess(X_val)
    
    # Set default vectorizer parameters
    default_vectorizer_params = {
        'max_features': 10000,
        'ngram_range': (1, 2),
        'min_df': 5
    }
    
    # Update with custom parameters if provided
    if vectorizer_params:
        default_vectorizer_params.update(vectorizer_params)
        
    # Initialize and fit TFIDF vectorizer
    logger.info(f"Initializing TfidfVectorizer with parameters: {default_vectorizer_params}")
    vectorizer = TfidfVectorizer(**default_vectorizer_params)
    
    try:
        X_train_vectorized = vectorizer.fit_transform(X_train_processed)
        X_val_vectorized = vectorizer.transform(X_val_processed)
        
        logger.info(f"Vectorized features: {X_train_vectorized.shape[1]}")
    except Exception as e:
        logger.error(f"Error vectorizing text: {e}")
        raise
    
    # Log preprocessing parameters to MLflow
    if mlflow_logging:
        preprocessing_params = {
            'vectorizer': 'TfidfVectorizer',
            **default_vectorizer_params,
            'remove_stopwords': preprocessor.remove_stopwords,
            'lemmatize': preprocessor.lemmatize
        }
        mlflow.log_params(preprocessing_params)
    
    return X_train_vectorized, X_val_vectorized, y_train, y_val, vectorizer, preprocessor

def create_submission_file(test_df, y_pred, output_path='submission.csv'):
    """Create submission file with predictions."""
    submission_df = pd.DataFrame({
        'id': range(len(test_df)),
        'label': y_pred
    })
    
    try:
        submission_df.to_csv(output_path, index=False)
        logger.info(f"Submission file created at {output_path}")
        
        # Log submission file to MLflow
        mlflow.log_artifact(output_path, "submissions")
    except Exception as e:
        logger.error(f"Error creating submission file: {e}")
        raise
    
    return submission_df 