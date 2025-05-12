"""
Math Topic Classifier with MLflow Integration
---------------------------------------------

This service provides a web interface and API for classifying math problems into different topics.
It leverages MLflow for model tracking, management, and serving.

MLflow Integration Options:
--------------------------

1. Standalone Server with FastAPI (Default)
   - Runs a complete FastAPI server with custom UI and preprocessing
   - Uses MLflow for model loading and tracking
   - Command: `python serve_model.py`

2. MLflow's Native Model Server
   - Uses MLflow's built-in model server with a REST API
   - Provides only the /invocations endpoint without UI
   - Command: `python serve_model.py --use-mlflow-server`
   - Direct API access: `curl -X POST -H "Content-Type:application/json" -d '{"inputs":["Solve for x: x^2 = 4"]}' http://localhost:5000/invocations`

3. Hybrid Mode (UI connecting to MLflow Server)
   - Run MLflow server: `mlflow models serve -m models:/MODEL_NAME/STAGE`
   - Run this server: `python serve_model.py`
   - In the UI Settings tab, select "Use External MLflow Server"

Additional Options:
-----------------
* Change models at runtime via the API or UI
* Perform model evaluation with true labels
* Change model stage/version with `--change-stage`

See README for more details.
"""

import sys
import os
import mlflow
import pandas as pd
import logging
import json
import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import uvicorn
from contextlib import asynccontextmanager
import requests

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Now we can import from data_processing
from src.data_processing import TextPreprocessor, download_nltk_resources

# Constants
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "math-topic-classifier-embeddings"
MODEL_STAGE = "Production"  # or "Staging"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Create FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model when the FastAPI app starts and clean up when it shuts down"""
    # Initialize model on startup
    logger.info("Initializing model on startup...")
    init_model()
    yield
    # Cleanup on shutdown (if needed)
    logger.info("Shutting down...")

app = FastAPI(title="Math Topic Classifier API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
topic_mapping = None
preprocessor = None
vectorizer = None
sklearn_model = None
sentence_transformer = None
embedding_model_name = DEFAULT_EMBEDDING_MODEL
feature_eng = None
is_deberta_model = False
is_embedding_model = False
is_traditional_model = False
USE_MLFLOW_SERVER = False  # Whether to use MLflow server for predictions
MLFLOW_SERVER_URL = "http://localhost:5000"  # Default MLflow server URL

# Define request and response models
class PredictionRequest(BaseModel):
    questions: Union[List[str], str]

class ModelSwitchRequest(BaseModel):
    model_name: str

class Prediction(BaseModel):
    question: str
    topic_id: int
    topic_name: str

class PredictionResponse(BaseModel):
    predictions: List[Prediction]

class ModelSwitchResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    model_name: Optional[str] = None
    
class MLflowServerToggleRequest(BaseModel):
    enabled: bool
    server_url: Optional[str] = None
    
class MLflowServerToggleResponse(BaseModel):
    success: bool
    message: str
    enabled: bool
    server_url: str

class MLflowServerQuery(BaseModel):
    questions: List[str]
    server_url: Optional[str] = None
    model_type: Optional[str] = None  # To indicate model type: "deberta", "embedding", "traditional"
    model_name: Optional[str] = None  # For model switching when using MLflow server

# From train_embeddings.py to ensure compatibility
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

def engineer_features(questions):
    """Extract additional features from the math problems."""
    logger.info("Engineering features...")
    
    from sklearn.preprocessing import StandardScaler
    
    # Create a DataFrame
    df = pd.DataFrame({"Question": questions})
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

# Helper function to find vectorizer in MLflow artifacts
def find_vectorizer_in_mlflow(run_id):
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
    logger.warning(f"Could not find embedding model name, using default: {DEFAULT_EMBEDDING_MODEL}")
    return DEFAULT_EMBEDDING_MODEL

# Initialize model and topic mapping
def init_model():
    """Initialize model and preprocessor using MLflow's capabilities."""
    global model, topic_mapping, preprocessor, vectorizer, sklearn_model, sentence_transformer
    global embedding_model_name, feature_eng, is_deberta_model, is_embedding_model, is_traditional_model
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to {MLFLOW_TRACKING_URI}")
        
        # Load model from registry using MLflow's built-in functionality
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        logger.info(f"Loading model from {model_uri}...")
        
        try:
            # Use MLflow's pyfunc model loader - this is the key change to leverage MLflow's serving capabilities
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Model loaded successfully with MLflow pyfunc")
        except Exception as e:
            logger.error(f"Error loading model from registry: {e}")
            logger.warning("Will try to find model in logged runs...")
            
            # Try to find the model from logged runs
            client = mlflow.tracking.MlflowClient()
            runs = client.search_runs(
                experiment_ids=[mlflow.get_experiment_by_name("math-problem-classification").experiment_id],
                filter_string=f"tags.mlflow.runName LIKE '%{MODEL_NAME}%'",
                max_results=1,
                order_by=["start_time DESC"]
            )
            
            if runs and runs[0]:
                run_id = runs[0].info.run_id
                logger.info(f"Found recent run: {run_id}")
                model_uri = f"runs:/{run_id}/models"
                try:
                    model = mlflow.pyfunc.load_model(model_uri)
                    logger.info("Model loaded from run successfully")
                except Exception as inner_e:
                    logger.error(f"Error loading model from run: {inner_e}")
                    raise RuntimeError("Failed to load model from MLflow") from inner_e
            else:
                raise RuntimeError("Could not find any relevant model runs") from e
        
        # Get underlying model
        if hasattr(model, '_model_impl'):
            # The newer MLflow flavor loading approach
            logger.info("Model loaded with newer MLflow flavor interface")
            impl = model._model_impl
            
            # Check if this is a PyFunc model with a custom Python model
            if hasattr(impl, 'python_model'):
                logger.info("Found Python model implementation")
                # This is likely a custom model
                python_model = impl.python_model
                
                # Check for DeBERTa model
                if hasattr(python_model, 'model') and "deberta" in str(python_model.model).lower():
                    logger.info("Detected DeBERTa model")
                    sklearn_model = python_model.model
                    is_deberta_model = True
                    
                    # Try to extract feature engineering preprocessor
                    if hasattr(python_model, 'feature_eng'):
                        feature_eng = python_model.feature_eng
                        logger.info("Extracted feature engineering preprocessor from DeBERTa model")
                        
                elif hasattr(python_model, 'sklearn_model'):
                    # Regular sklearn model within PyFunc wrapper
                    logger.info("Found sklearn model within Python model")
                    sklearn_model = python_model.sklearn_model
                    is_traditional_model = True
                else:
                    # Use the Python model directly
                    sklearn_model = python_model
            else:
                # Might be a direct sklearn model
                sklearn_model = impl
                is_traditional_model = True
        elif hasattr(model, 'python_function'):
            # The older approach for MLflow <= 1.x
            logger.info("Model loaded with older MLflow interface")
            sklearn_model = model.python_function.loaded_model()
            is_traditional_model = True
        else:
            # Direct pythonic approach
            logger.info("Model loaded with direct pythonic approach")
            sklearn_model = model
            is_traditional_model = True
            
        # Log model information
        logger.info(f"Model type: {type(sklearn_model).__name__}")
        
        # Download NLTK resources for text preprocessing
        logger.info("Downloading NLTK resources...")
        download_nltk_resources()
        
        # Get run ID from model URI
        run_id = None
        if hasattr(model, 'metadata') and hasattr(model.metadata, 'run_id'):
            if "runs:/" in model.metadata.run_id:
                run_id = model.metadata.run_id.split("/")[1]
            else:
                run_id = model.metadata.run_id
            logger.info(f"Model run ID: {run_id}")

        # Check model type more precisely
        model_name_lower = MODEL_NAME.lower()
        if "deberta" in model_name_lower or "transformer" in model_name_lower:
            logger.info("Detected transformer model from name")
            is_deberta_model = True
        elif "embedding" in model_name_lower:
            logger.info("Detected embedding model from name")
            is_embedding_model = True
            
        # Initialize preprocessor
        preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
        logger.info("Text preprocessor initialized")
        
        # Try to load the vectorizer from MLflow artifacts or sklearn_model pipeline
        if hasattr(sklearn_model, 'named_steps') and 'vectorizer' in sklearn_model.named_steps:
            # The vectorizer is part of the pipeline
            logger.info("Vectorizer found in model pipeline")
            vectorizer = sklearn_model.named_steps['vectorizer']
        else:
            # Try to find the vectorizer in MLflow artifacts
            logger.info("Searching for vectorizer in MLflow artifacts...")
            vectorizer = find_vectorizer_in_mlflow(run_id)
            
            if vectorizer is None:
                logger.warning("Vectorizer not found in model artifacts. The model may be using embeddings or a different approach.")
        
        # Load embedding model name if model is using embeddings
        if (hasattr(sklearn_model, 'embedding_model') or 
            "embedding" in MODEL_NAME.lower() or
            is_embedding_model):
            
            logger.info("Model appears to use embeddings. Looking for embedding model info...")
            embedding_model_name = get_embedding_model_name(run_id)
            is_embedding_model = True
            
            # Initialize sentence transformer if needed
            try:
                # Make sure we have an embedding model name
                if embedding_model_name is None or embedding_model_name == DEFAULT_EMBEDDING_MODEL:
                    # Check model name to infer embedding model
                    if "minilm" in MODEL_NAME.lower():
                        # Use MiniLM model
                        embedding_model_name = "all-MiniLM-L6-v2"
                    elif "mpnet" in MODEL_NAME.lower():
                        # Use MPNet model
                        embedding_model_name = "all-mpnet-base-v2"
                    elif "bert" in MODEL_NAME.lower() and not "deberta" in MODEL_NAME.lower():
                        # Use BERT model
                        embedding_model_name = "bert-base-uncased"
                    else:
                        # Default is MiniLM
                        embedding_model_name = DEFAULT_EMBEDDING_MODEL
                          
                logger.info(f"Loading SentenceTransformer with model: {embedding_model_name}")
                from sentence_transformers import SentenceTransformer
                sentence_transformer = SentenceTransformer(embedding_model_name)
                logger.info("SentenceTransformer loaded successfully")
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer: {e}")
                logger.warning("Will proceed without embeddings support")
                
        # Load topic mapping
        try:
            from src.utils import TOPIC_MAPPING
            topic_mapping = TOPIC_MAPPING
            logger.info(f"Loaded topic mapping with {len(topic_mapping)} topics")
        except:
            logger.warning("Could not import TOPIC_MAPPING from utils. Using default mapping.")
            # Define a default mapping
            topic_mapping = {
                0: "Algebra",
                1: "Geometry and Trigonometry",
                2: "Calculus and Analysis",
                3: "Probability and Statistics",
                4: "Number Theory",
                5: "Combinatorics and Discrete Math",
                6: "Linear Algebra",
                7: "Abstract Algebra and Topology"
            }
        
        logger.info("Model initialization complete")
        logger.info(f"Model type summary: DeBERTa: {is_deberta_model}, Embedding: {is_embedding_model}, Traditional: {is_traditional_model}")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return False

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global model, sklearn_model, sentence_transformer, embedding_model_name
    global is_deberta_model, is_embedding_model, is_traditional_model, feature_eng
    global USE_MLFLOW_SERVER, MLFLOW_SERVER_URL  # These would be set via the UI
    
    try:
        # Get questions from request
        questions = request.questions
        if isinstance(questions, str):
            questions = [questions]
        
        # Check if we're in MLflow server mode (set via the web UI)
        if USE_MLFLOW_SERVER:
            logger.info(f"Using MLflow server at {MLFLOW_SERVER_URL} for prediction")
            # Determine model type for optimal input format
            model_type = None
            if is_deberta_model:
                model_type = "deberta"
                # Query MLflow server with DeBERTa optimized format
                predictions = query_mlflow_server(
                    questions=questions,
                    server_url=MLFLOW_SERVER_URL,
                    model_type=model_type
                )
            elif is_embedding_model:
                model_type = "embedding"
                # For embedding models, we may need more careful preprocessing
                try:
                    # Try the optimized approach first
                    predictions = query_mlflow_server(
                        questions=questions,
                        server_url=MLFLOW_SERVER_URL,
                        model_type=model_type
                    )
                except Exception as e:
                    logger.warning(f"Optimized embedding query failed: {e}, using full preprocessing")
                    # Fall back to the full preprocessing approach
                    predictions = preprocessing_query_mlflow_server(
                        questions=questions,
                        server_url=MLFLOW_SERVER_URL
                    )
            else:
                model_type = "traditional"
                # For traditional models
                predictions = query_mlflow_server(
                    questions=questions,
                    server_url=MLFLOW_SERVER_URL,
                    model_type=model_type
                )
            
            # Format the predictions as a response
            formatted_predictions = []
            for i, pred in enumerate(predictions):
                topic_id = int(pred)
                topic_name = topic_mapping.get(topic_id, f"Unknown Topic {topic_id}")
                
                formatted_predictions.append(
                    Prediction(
                        question=questions[i],
                        topic_id=topic_id,
                        topic_name=topic_name
                    )
                )
            
            return PredictionResponse(predictions=formatted_predictions)
        
        # Create DataFrame for prediction
        logger.info(f"Processing {len(questions)} questions")
        
        # Different processing paths based on model type
        if is_deberta_model:
            logger.info("Using DeBERTa model prediction path")
            
            # DeBERTa models can usually predict directly on raw text
            try:
                # First try direct prediction which works for most DeBERTa models
                predictions = model.predict(questions)
                logger.info("Direct prediction with DeBERTa model succeeded")
            except Exception as e:
                logger.warning(f"Direct prediction failed: {e}. Trying batch prediction...")
                
                # Some models might need batching for memory efficiency
                all_preds = []
                batch_size = 16  # Small batch size for memory efficiency
                
                for i in range(0, len(questions), batch_size):
                    batch = questions[i:i+batch_size]
                    try:
                        batch_preds = model.predict(batch)
                        all_preds.extend(batch_preds)
                    except Exception as batch_e:
                        logger.error(f"Batch prediction failed: {batch_e}")
                        raise RuntimeError(f"Both direct and batch prediction failed with DeBERTa model") from batch_e
                
                predictions = all_preds
                logger.info("Batch prediction with DeBERTa model succeeded")
                
        elif is_embedding_model:
            logger.info("Using embedding model prediction path")
            
            # Step 1: Clean text for embeddings 
            cleaned_text = [clean_text(q) for q in questions]
            logger.info("Cleaned text for embeddings")
            
            # Step 2: Preprocess text for TF-IDF
            if preprocessor is None:
                raise HTTPException(status_code=500, detail="Preprocessor not found. Cannot make predictions.")
                
            processed_text = preprocessor.preprocess(questions)
            logger.info(f"Preprocessed text successfully: {len(processed_text)} questions")
            
            # Step 3: Generate TF-IDF features
            if vectorizer is None:
                logger.error("Vectorizer not found. Cannot make predictions.")
                raise RuntimeError("Vectorizer not found. Cannot make predictions.")
            
            # CRITICAL: Use processed_text for TF-IDF, not cleaned_text
            # This must match how the model was trained and how local prediction works
            X_tfidf = vectorizer.transform(processed_text)
            logger.info(f"Generated TF-IDF features with shape: {X_tfidf.shape}")
            
            # Step 4: Generate sentence embeddings from cleaned_text
            if sentence_transformer is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    sentence_transformer = SentenceTransformer(embedding_model_name)
                    logger.info(f"Loaded sentence transformer model on-demand: {embedding_model_name}")
                except Exception as e:
                    logger.error(f"Failed to load sentence transformer: {e}")
                    raise HTTPException(status_code=500, detail="Cannot generate embeddings for prediction.")
            
            # Generate embeddings from cleaned_text, not processed_text
            embeddings = sentence_transformer.encode(cleaned_text)
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            
            # Step 5: Generate engineered features from original questions
            features_df = engineer_features(questions)
            logger.info(f"Generated engineered features with shape: {features_df.shape}")
            
            # Step 6: Convert to sparse matrices for combining
            
            embeddings_sparse = csr_matrix(embeddings)
            features_sparse = csr_matrix(features_df.values)
            
            # Combine all features in the same order as training
            X_combined = hstack([X_tfidf, embeddings_sparse, features_sparse]).tocsr()
            logger.info(f"Combined all features with shape: {X_combined.shape}")
            
            # Make prediction with sklearn model
            predictions = sklearn_model.predict(X_combined)
            logger.info("Made predictions with embedding model")
            
        else:
            logger.info("Using traditional model prediction path")
            
            # For traditional models, we just need to preprocess and vectorize
            processed_text = preprocessor.preprocess(questions)
            
            if vectorizer is not None:
                # Transform with vectorizer if available
                X = vectorizer.transform(processed_text)
                logger.info(f"Transformed text with vectorizer to shape: {X.shape}")
            else:
                # Some models might be able to process raw text
                X = processed_text
                logger.info("No vectorizer found, using preprocessed text directly")
            
            # Try prediction
            try:
                predictions = sklearn_model.predict(X)
                logger.info("Made predictions with traditional model")
            except Exception as e:
                logger.error(f"Error in traditional model prediction: {e}")
                
                # Fall back to the main model
                try:
                    predictions = model.predict(questions)
                    logger.info("Made predictions with fallback to main model")
                except Exception as fallback_e:
                    logger.error(f"Fallback prediction failed: {fallback_e}")
                    raise RuntimeError("All prediction methods failed") from fallback_e
        
        logger.info(f"Made predictions successfully: {predictions}")
        
        # Format response
        results = []
        for q, pred in zip(questions, predictions):
            pred_int = int(pred)
            topic_name = topic_mapping.get(pred_int, f"Unknown Topic {pred_int}")
            results.append(Prediction(
                question=q,
                topic_id=pred_int,
                topic_name=topic_name
            ))
        
        return PredictionResponse(predictions=results)
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(status_code=500, detail=str(e))

# Add a new endpoint to switch the model
@app.post("/switch_model", response_model=ModelSwitchResponse)
async def switch_model(request: ModelSwitchRequest):
    global MODEL_NAME, model, topic_mapping, preprocessor, vectorizer, sklearn_model, sentence_transformer
    global embedding_model_name, feature_eng, is_deberta_model, is_embedding_model, is_traditional_model
    
    try:
        # Store the current model name to report changes
        old_model_name = MODEL_NAME
        
        # Update the model name
        MODEL_NAME = request.model_name
        logger.info(f"Switching model from {old_model_name} to {MODEL_NAME}")
        
        # Reset model-specific flags
        is_deberta_model = False
        is_embedding_model = False
        is_traditional_model = False
        
        # Reinitialize the model with the new name
        if init_model():
            return ModelSwitchResponse(
                success=True, 
                message=f"Successfully switched from {old_model_name} to {MODEL_NAME}",
                model_name=MODEL_NAME
            )
        else:
            # If initialization fails, revert to the old model
            MODEL_NAME = old_model_name
            init_model()  # Attempt to reinitialize with the old model
            return ModelSwitchResponse(
                success=False, 
                error=f"Failed to initialize model '{request.model_name}'. Reverted to {old_model_name}."
            )
    
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Home page with usage instructions
@app.get("/", response_class=HTMLResponse)
async def home():
    # Add timestamp to verify latest version
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return f"""
    <html>
    <head>
        <title>Math Topic Classifier API</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            :root {{
                --primary-color: #4361ee;
                --secondary-color: #3f37c9;
                --success-color: #4caf50;
                --light-gray: #f8f9fa;
                --dark-gray: #495057;
                --border-color: #dee2e6;
                --border-radius: 8px;
                --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            
            body {{
                font-family: 'Roboto', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: var(--light-gray);
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .container {{
                background-color: white;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                padding: 30px;
                margin-bottom: 30px;
            }}
            
            .header {{
                border-bottom: 2px solid var(--primary-color);
                padding-bottom: 15px;
                margin-bottom: 25px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }}
            
            .header h1 {{
                color: var(--primary-color);
                font-weight: 700;
                margin: 0;
            }}
            
            .header .logo {{
                font-size: 24px;
                margin-right: 15px;
                color: var(--primary-color);
            }}
            
            h1, h2, h3 {{
                color: var(--dark-gray);
                margin-bottom: 15px;
            }}
            
            p {{
                margin-bottom: 15px;
            }}
            
            .card {{
                background-color: white;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                padding: 20px;
                margin-bottom: 20px;
            }}
            
            .code-block {{
                background-color: var(--light-gray);
                padding: 15px;
                border-radius: var(--border-radius);
                border-left: 4px solid var(--primary-color);
                font-family: monospace;
                white-space: pre-wrap;
                overflow-x: auto;
                margin: 15px 0;
                font-size: 14px;
            }}
            
            .form-container {{
                margin-top: 20px;
                background-color: white;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                padding: 25px;
            }}
            
            .form-group {{
                margin-bottom: 20px;
            }}
            
            label {{
                display: block;
                margin-bottom: 8px;
                font-weight: 500;
                color: var(--dark-gray);
            }}
            
            textarea, input[type="text"] {{
                width: 100%;
                padding: 12px;
                border: 1px solid var(--border-color);
                border-radius: var(--border-radius);
                font-family: 'Roboto', sans-serif;
                font-size: 14px;
                transition: border-color 0.3s;
            }}
            
            textarea {{
                height: 150px;
                resize: vertical;
            }}
            
            textarea:focus, input[type="text"]:focus {{
                outline: none;
                border-color: var(--primary-color);
                box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.2);
            }}
            
            button {{
                padding: 12px 20px;
                background-color: var(--primary-color);
                color: white;
                border: none;
                border-radius: var(--border-radius);
                cursor: pointer;
                font-weight: 500;
                transition: background-color 0.3s, transform 0.2s;
                display: inline-flex;
                align-items: center;
                justify-content: center;
            }}
            
            button:hover {{
                background-color: var(--secondary-color);
                transform: translateY(-2px);
            }}
            
            button i {{
                margin-right: 8px;
            }}
            
            #clear-btn {{
                background-color: #6c757d;
                margin-left: 10px;
            }}
            
            #clear-btn:hover {{
                background-color: #5a6268;
            }}
            
            #example-btn {{
                background-color: #17a2b8;
                margin-left: 10px;
            }}
            
            #example-btn:hover {{
                background-color: #138496;
            }}
            
            .result {{
                margin-top: 25px;
                padding: 20px;
                border-radius: var(--border-radius);
                display: none;
                background-color: white;
                box-shadow: var(--box-shadow);
            }}
            
            .result h3 {{
                margin-top: 0;
                color: var(--primary-color);
                display: flex;
                align-items: center;
            }}
            
            .result h3 i {{
                margin-right: 10px;
            }}
            
            .result-card {{
                background-color: var(--light-gray);
                border-radius: var(--border-radius);
                padding: 15px;
                margin-top: 15px;
                border-left: 4px solid var(--primary-color);
            }}
            
            .topic-name {{
                font-weight: bold;
                color: var(--primary-color);
            }}
            
            .accuracy-panel {{
                background-color: #e6f7ff;
                border-radius: var(--border-radius);
                padding: 15px;
                margin-bottom: 20px;
                border-left: 4px solid #1890ff;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            
            .accuracy-score {{
                font-size: 18px;
                margin-bottom: 5px;
            }}
            
            .accuracy-value {{
                font-weight: bold;
                font-size: 22px;
                color: #1890ff;
            }}
            
            .accuracy-details {{
                font-size: 14px;
                color: #555;
            }}
            
            .eval-info {{
                background-color: #fffbe6;
                padding: 8px;
                border-radius: var(--border-radius);
                font-size: 13px;
                margin-bottom: 10px;
            }}
            
            .eval-info i {{
                color: #faad14;
                margin-right: 5px;
            }}
            
            .form-control {{
                width: 100%;
                padding: 10px;
                border: 1px solid var(--border-color);
                border-radius: var(--border-radius);
                font-family: 'Roboto', sans-serif;
                font-size: 14px;
                transition: border-color 0.3s;
            }}
            
            .topic-reference {{
                color: #666;
                line-height: 1.4;
            }}
            
            .result-card.correct {{
                border-left: 4px solid #52c41a;
            }}
            
            .result-card.incorrect {{
                border-left: 4px solid #f5222d;
            }}
            
            .prediction-result {{
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .prediction-badge {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 500;
                color: white;
            }}
            
            .prediction-badge.correct {{
                background-color: #52c41a;
            }}
            
            .prediction-badge.incorrect {{
                background-color: #f5222d;
            }}
            
            .true-label {{
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px dashed #ddd;
                font-size: 14px;
            }}
            
            .timestamp {{
                color: #6c757d;
                font-size: 12px;
                text-align: center;
                margin-top: 30px;
            }}
            
            .tabs {{
                display: flex;
                margin-bottom: 20px;
                border-bottom: 1px solid var(--border-color);
            }}
            
            .tab {{
                padding: 10px 20px;
                cursor: pointer;
                border-bottom: 3px solid transparent;
                transition: all 0.3s;
                font-weight: 500;
            }}
            
            .tab.active {{
                border-bottom: 3px solid var(--primary-color);
                color: var(--primary-color);
            }}
            
            .tab-content {{
                display: none;
            }}
            
            .tab-content.active {{
                display: block;
            }}
            
            .loader {{
                display: none;
                text-align: center;
                margin: 20px 0;
            }}
            
            .spinner {{
                border: 4px solid rgba(0, 0, 0, 0.1);
                border-radius: 50%;
                border-top: 4px solid var(--primary-color);
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            .api-info {{
                display: flex;
                align-items: center;
                padding: 10px;
                background-color: #d1ecf1;
                border-radius: var(--border-radius);
                margin-bottom: 20px;
            }}
            
            .api-info i {{
                font-size: 20px;
                color: #0c5460;
                margin-right: 10px;
            }}
            
            .btn-row {{
                display: flex;
                margin-top: 15px;
            }}
            
            .topic-list {{
                columns: 2;
                margin-bottom: 20px;
            }}
            
            .topic-item {{
                padding: 5px 0;
            }}
            
            .model-selector-form {{
                background-color: var(--light-gray);
                padding: 15px;
                border-radius: var(--border-radius);
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                flex-wrap: wrap;
            }}
            
            .model-selector-form label {{
                margin-right: 10px;
                margin-bottom: 0;
                font-weight: 500;
            }}
            
            .model-selector-form input {{
                flex: 1;
                min-width: 200px;
                margin-right: 10px;
                margin-bottom: 0;
            }}
            
            .model-selector-form button {{
                background-color: #6f42c1;
            }}
            
            .model-selector-form button:hover {{
                background-color: #5a32a3;
            }}
            
            .status-badge {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 500;
                color: white;
                background-color: var(--primary-color);
            }}

            #model-status {{
                display: none;
                margin-top: 10px;
                padding: 10px;
                border-radius: var(--border-radius);
                width: 100%;
            }}

            #model-status.success {{
                background-color: #d4edda;
                color: #155724;
                border-left: 4px solid #28a745;
            }}

            #model-status.error {{
                background-color: #f8d7da;
                color: #721c24;
                border-left: 4px solid #dc3545;
            }}
            
            @media (max-width: 768px) {{
                .topic-list {{
                    columns: 1;
                }}
                
                .btn-row {{
                    flex-wrap: wrap;
                }}
                
                button {{
                    margin-bottom: 10px;
                }}

                .model-selector-form {{
                    flex-direction: column;
                    align-items: stretch;
                }}
                
                .model-selector-form label, .model-selector-form input, .model-selector-form button {{
                    margin-bottom: 10px;
                    margin-right: 0;
                    width: 100%;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-brain logo"></i> Math Topic Classifier</h1>
                <div>
                    <span class="status-badge" id="current-model">Model: {MODEL_NAME}</span>
                    <span>API v1.0</span>
                </div>
            </div>
            
            <div class="tabs">
                <div class="tab active" data-tab="classifier">Topic Classifier</div>
                <div class="tab" data-tab="api">API Documentation</div>
                <div class="tab" data-tab="topics">Topic List</div>
                <div class="tab" data-tab="settings">Settings</div>
            </div>
            
            <div class="tab-content active" id="classifier-tab">
                <div class="api-info">
                    <i class="fas fa-info-circle"></i>
                    <p>This tool uses machine learning to classify math problems into different topic areas.</p>
                </div>
                
                <div class="form-container">
                    <h2><i class="fas fa-pencil-alt"></i> Test the Classifier</h2>
                    <form id="test-form">
                        <div class="form-group">
                            <label for="questions">Enter one or more math problems (one per line):</label>
                            <textarea id="questions" placeholder="Example: Solve the quadratic equation x^2 - 5x + 6 = 0."></textarea>
                        </div>
                        
                        <div class="form-group">
                            <label for="enable-evaluation">
                                <input type="checkbox" id="enable-evaluation"> Enable Evaluation Mode
                            </label>
                            <div id="evaluation-options" style="display: none; margin-top: 10px;">
                                <p class="eval-info"><i class="fas fa-info-circle"></i> Add true topic IDs to evaluate model accuracy</p>
                                <label for="true-labels">True Topic IDs (comma-separated, matching order of questions):</label>
                                <input type="text" id="true-labels" placeholder="e.g., 0,2,3,1" class="form-control">
                                <div class="topic-reference" style="margin-top: 8px; font-size: 12px;">
                                    <span><strong>Topics:</strong> 0:Algebra, 1:Geometry, 2:Calculus, 3:Statistics, 4:Number Theory, 5:Combinatorics, 6:Linear Algebra, 7:Abstract Algebra</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="btn-row">
                            <button type="submit"><i class="fas fa-cogs"></i> Classify</button>
                            <button type="button" id="example-btn"><i class="fas fa-lightbulb"></i> Load Examples</button>
                            <button type="button" id="clear-btn"><i class="fas fa-trash"></i> Clear</button>
                        </div>
                    </form>
                    
                    <div class="loader" id="loader">
                        <div class="spinner"></div>
                        <p>Classifying...</p>
                    </div>
                    
                    <div id="result" class="result">
                        <h3><i class="fas fa-chart-pie"></i> Classification Results</h3>
                        <div id="accuracy-results" style="display: none;" class="accuracy-panel">
                            <div class="accuracy-score">
                                <span class="accuracy-label">Accuracy:</span>
                                <span id="accuracy-percentage" class="accuracy-value">0%</span>
                            </div>
                            <div class="accuracy-details">
                                <span id="correct-count">0</span> correct out of <span id="total-count">0</span> questions
                            </div>
                        </div>
                        <div id="result-cards"></div>
                        <pre id="result-content" style="display: none;"></pre>
                    </div>
                </div>
            </div>
            
            <div class="tab-content" id="api-tab">
                <h2><i class="fas fa-code"></i> API Documentation</h2>
                <p>Send POST requests to <code>/predict</code> with JSON data containing a 'questions' field.</p>
                
                <h3>Request Format</h3>
                <div class="code-block">{{
    "questions": [
        "Solve the quadratic equation x^2 - 5x + 6 = 0.",
        "Find the derivative of f(x) = x^3 - 2x + 1."
    ]
}}</div>
                
                <h3>Response Format</h3>
                <div class="code-block">{{
    "predictions": [
        {{
            "question": "Solve the quadratic equation x^2 - 5x + 6 = 0.",
            "topic_id": 0,
            "topic_name": "Algebra"
        }},
        {{
            "question": "Find the derivative of f(x) = x^3 - 2x + 1.",
            "topic_id": 2,
            "topic_name": "Calculus and Analysis"
        }}
    ]
}}</div>

                <h3>Python Example</h3>
                <div class="code-block">import requests
import json

url = "http://localhost:5001/predict"
data = {{
    "questions": [
        "Solve the quadratic equation x^2 - 5x + 6 = 0."
    ]
}}

response = requests.post(url, json=data)
results = response.json()
print(json.dumps(results, indent=2))</div>

                <h3>Switch Model Endpoint</h3>
                <p>Send POST requests to <code>/switch_model</code> to change the active model:</p>
                <div class="code-block">{{
    "model_name": "math-topic-classifier-embeddings"
}}</div>
            </div>
            
            <div class="tab-content" id="topics-tab">
                <h2><i class="fas fa-list"></i> Available Topics</h2>
                <p>The classifier can identify the following mathematical topics:</p>
                
                <div class="topic-list">
                    <div class="topic-item"><strong>0:</strong> Algebra</div>
                    <div class="topic-item"><strong>1:</strong> Geometry and Trigonometry</div>
                    <div class="topic-item"><strong>2:</strong> Calculus and Analysis</div>
                    <div class="topic-item"><strong>3:</strong> Probability and Statistics</div>
                    <div class="topic-item"><strong>4:</strong> Number Theory</div>
                    <div class="topic-item"><strong>5:</strong> Combinatorics and Discrete Math</div>
                    <div class="topic-item"><strong>6:</strong> Linear Algebra</div>
                    <div class="topic-item"><strong>7:</strong> Abstract Algebra and Topology</div>
                </div>
                
                <div class="card">
                    <h3><i class="fas fa-info-circle"></i> Model Information</h3>
                    <p>This classifier uses advanced machine learning techniques to categorize mathematical problems based on their content. It analyzes the text, mathematical expressions, and keywords to determine the most likely mathematical domain.</p>
                </div>
            </div>
            
            <div class="tab-content" id="settings-tab">
                <h2><i class="fas fa-cog"></i> Model Settings</h2>
                <p>You can switch to a different model by entering its name below:</p>
                
                <div class="model-selector-form">
                    <label for="model-name">Model Name:</label>
                    <input type="text" id="model-name" placeholder="e.g., math-topic-classifier-embeddings" value="{MODEL_NAME}">
                    <button type="button" id="switch-model-btn"><i class="fas fa-sync-alt"></i> Switch Model</button>
                </div>
                
                <div id="model-status"></div>
                
                <h2><i class="fas fa-server"></i> MLflow Server Connection</h2>
                <p>You can connect to an external MLflow model server:</p>
                
                <div class="model-selector-form">
                    <label for="mlflow-server-url">MLflow Server URL:</label>
                    <input type="text" id="mlflow-server-url" placeholder="e.g., http://localhost:5000" value="http://localhost:5000">
                    <button type="button" id="test-mlflow-connection-btn"><i class="fas fa-plug"></i> Test Connection</button>
                </div>
                
                <div id="mlflow-connection-status" style="display: none; margin-top: 10px; padding: 10px; border-radius: var(--border-radius);"></div>
                
                <div class="form-group">
                    <label for="use-mlflow-server">
                        <input type="checkbox" id="use-mlflow-server"> Use External MLflow Server for Predictions
                    </label>
                    <p class="eval-info" style="margin-top: 8px;">
                        <i class="fas fa-info-circle"></i> 
                        When enabled, prediction requests will be sent to the specified MLflow server
                        instead of using the local model.
                    </p>
                </div>
                
                <div class="card">
                    <h3><i class="fas fa-lightbulb"></i> Available Models</h3>
                    <p>Here are some models you might want to try:</p>
                    <ul>
                        <li><strong>math-topic-classifier-embeddings</strong> - Uses sentence embeddings for classification</li>
                        <li><strong>math-topic-classifier</strong> - Uses traditional ML methods</li>
                        <li><strong>math-topic-classifier-enhanced-deberta</strong> - Uses DeBERTa transformer model</li>
                    </ul>
                    <p>Note: The model must be registered in MLflow with the specified name.</p>
                    <p>To change model stages, use the command line interface with the --change-stage argument.</p>
                </div>
            </div>
            
            <p class="timestamp">Last updated: {timestamp}</p>
        </div>
        
        <script>
            // Tab functionality
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.addEventListener('click', () => {{
                    // Remove active class from all tabs
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    
                    // Add active class to clicked tab
                    tab.classList.add('active');
                    document.getElementById(`${{tab.dataset.tab}}-tab`).classList.add('active');
                }});
            }});
            
            // Example questions
            const exampleQuestions = [
                "Solve the quadratic equation x^2 - 5x + 6 = 0.",
                "Find the derivative of f(x) = x^3 - 2x + 1.",
                "If P(A) = 0.3 and P(B) = 0.5 and P(A∩B) = 0.2, find P(A|B).",
                "Find the area of a circle with radius 5 units.",
                "Prove that the sum of the first n positive integers is n(n+1)/2."
            ];
            
            // Example true labels (for evaluation mode example)
            const exampleLabels = "0,2,3,1,4";
            
            // Toggle evaluation mode
            document.getElementById('enable-evaluation').addEventListener('change', function() {{
                const evaluationOptions = document.getElementById('evaluation-options');
                evaluationOptions.style.display = this.checked ? 'block' : 'none';
            }});
            
            // Load examples button
            document.getElementById('example-btn').addEventListener('click', function() {{
                document.getElementById('questions').value = exampleQuestions.join('\\n');
                document.getElementById('true-labels').value = exampleLabels;
                
                // Auto-enable evaluation mode if examples include labels
                const evaluationCheckbox = document.getElementById('enable-evaluation');
                if (!evaluationCheckbox.checked) {{
                    evaluationCheckbox.checked = true;
                    document.getElementById('evaluation-options').style.display = 'block';
                }}
            }});
            
            // Clear button
            document.getElementById('clear-btn').addEventListener('click', function() {{
                document.getElementById('questions').value = '';
                document.getElementById('true-labels').value = '';
                document.getElementById('result').style.display = 'none';
                document.getElementById('accuracy-results').style.display = 'none';
            }});
            
            // Calculate accuracy
            function calculateAccuracy(predictions, trueLabels) {{
                let correct = 0;
                
                for (let i = 0; i < predictions.length; i++) {{
                    if (i < trueLabels.length && predictions[i].topic_id === trueLabels[i]) {{
                        correct++;
                    }}
                }}
                
                return {{
                    correct: correct,
                    total: Math.min(predictions.length, trueLabels.length),
                    percentage: Math.min(predictions.length, trueLabels.length) > 0 
                        ? Math.round((correct / Math.min(predictions.length, trueLabels.length)) * 100) 
                        : 0
                }};
            }}
            
            // Form submission
            document.getElementById('test-form').addEventListener('submit', function(e) {{
                e.preventDefault();
                
                // Get questions
                const questionsText = document.getElementById('questions').value;
                const questions = questionsText.split('\\n').filter(q => q.trim() !== '');
                
                if (questions.length === 0) {{
                    alert('Please enter at least one math problem.');
                    return;
                }}
                
                // Get true labels if evaluation is enabled
                const isEvaluationEnabled = document.getElementById('enable-evaluation').checked;
                let trueLabels = [];
                
                if (isEvaluationEnabled) {{
                    const trueLabelsText = document.getElementById('true-labels').value.trim();
                    if (trueLabelsText) {{
                        trueLabels = trueLabelsText.split(',').map(label => parseInt(label.trim()));
                    }}
                }}
                
                // Show loader
                document.getElementById('loader').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                document.getElementById('accuracy-results').style.display = 'none';
                
                // Create payload
                const payload = {{
                    questions: questions
                }};
                
                // Check if using MLflow server
                const useMLflowServer = document.getElementById('use-mlflow-server').checked;
                let endpoint = '/predict';
                
                if (useMLflowServer) {{
                    endpoint = '/query_mlflow';
                    const serverUrl = document.getElementById('mlflow-server-url').value.trim();
                    if (serverUrl) {{
                        payload.server_url = serverUrl;
                    }}
                    
                    // Determine model type for optimal input format
                    const currentModelName = document.getElementById('current-model').textContent.replace('Model: ', '');
                    if (currentModelName.toLowerCase().includes('deberta') || 
                        currentModelName.toLowerCase().includes('transformer')) {{
                        payload.model_type = 'deberta';
                    }} else if (currentModelName.toLowerCase().includes('embedding')) {{
                        payload.model_type = 'embedding';
                    }} else {{
                        payload.model_type = 'traditional';
                    }}
                }}
                
                // Make request
                fetch(endpoint, {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify(payload)
                }})
                .then(response => {{
                    if (!response.ok) {{
                        throw new Error('HTTP error! Status: ' + response.status);
                    }}
                    return response.json();
                }})
                .then(data => {{
                    // Hide loader
                    document.getElementById('loader').style.display = 'none';
                    
                    // Show results
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('result-content').textContent = JSON.stringify(data, null, 2);
                    
                    // Clear previous results
                    const resultCards = document.getElementById('result-cards');
                    resultCards.innerHTML = '';
                    
                    // Create a card for each prediction
                    if (data.predictions && data.predictions.length > 0) {{
                        // Calculate accuracy if evaluation is enabled
                        if (isEvaluationEnabled && trueLabels.length > 0) {{
                            const accuracy = calculateAccuracy(data.predictions, trueLabels);
                            
                            // Display accuracy metrics
                            document.getElementById('accuracy-percentage').textContent = accuracy.percentage + '%';
                            document.getElementById('correct-count').textContent = accuracy.correct;
                            document.getElementById('total-count').textContent = accuracy.total;
                            document.getElementById('accuracy-results').style.display = 'block';
                        }}
                        
                        data.predictions.forEach((pred, index) => {{
                            const card = document.createElement('div');
                            const isCorrect = isEvaluationEnabled && trueLabels.length > index && pred.topic_id === trueLabels[index];
                            const hasEvaluation = isEvaluationEnabled && trueLabels.length > index;
                            
                            // Add appropriate classes based on correctness
                            card.className = 'result-card';
                            if (hasEvaluation) {{
                                card.className += isCorrect ? ' correct' : ' incorrect';
                            }}
                            
                            // Create card content
                            let cardContent = 
                                '<p><strong>Question:</strong> ' + pred.question + '</p>' +
                                '<div class="prediction-result">' +
                                '<p><strong>Predicted Topic:</strong> <span class="topic-name">' + pred.topic_name + '</span> (ID: ' + pred.topic_id + ')</p>';
                            
                            // Add evaluation badge if in evaluation mode
                            if (hasEvaluation) {{
                                const badgeClass = isCorrect ? 'correct' : 'incorrect';
                                const badgeText = isCorrect ? 'Correct' : 'Incorrect';
                                cardContent += '<span class="prediction-badge ' + badgeClass + '">' + badgeText + '</span>';
                            }}
                            
                            cardContent += '</div>';
                            
                            // Add true label if available
                            if (hasEvaluation) {{
                                const trueLabelId = trueLabels[index];
                                const topicNames = {{
                                    0: "Algebra",
                                    1: "Geometry and Trigonometry",
                                    2: "Calculus and Analysis",
                                    3: "Probability and Statistics",
                                    4: "Number Theory",
                                    5: "Combinatorics and Discrete Math",
                                    6: "Linear Algebra",
                                    7: "Abstract Algebra and Topology"
                                }};
                                const topicName = topicNames[trueLabelId] || ('Topic ' + trueLabelId);
                                cardContent += 
                                    '<div class="true-label">' +
                                    '<strong>True Topic:</strong> <span class="topic-name">' + topicName + '</span> (ID: ' + trueLabelId + ')' +
                                    '</div>';
                            }}
                            
                            card.innerHTML = cardContent;
                            resultCards.appendChild(card);
                        }});
                    }} else if (data.error) {{
                        const card = document.createElement('div');
                        card.className = 'result-card';
                        card.innerHTML = '<p><strong>Error:</strong> ' + data.error + '</p>';
                        resultCards.appendChild(card);
                    }}
                }})
                .catch(error => {{
                    // Hide loader
                    document.getElementById('loader').style.display = 'none';
                    
                    // Show error
                    document.getElementById('result').style.display = 'block';
                    
                    const resultCards = document.getElementById('result-cards');
                    resultCards.innerHTML = 
                        '<div class="result-card">' +
                        '<p><strong>Error:</strong> ' + (error.message || 'An unknown error occurred') + '</p>' +
                        '</div>';
                }});
            }});
            
            // Test MLflow Connection button
            document.getElementById('test-mlflow-connection-btn').addEventListener('click', function() {{
                const serverUrl = document.getElementById('mlflow-server-url').value.trim();
                
                if (!serverUrl) {{
                    alert('Please enter a MLflow server URL.');
                    return;
                }}
                
                // Show status as pending
                const connectionStatus = document.getElementById('mlflow-connection-status');
                connectionStatus.style.display = 'block';
                connectionStatus.className = '';
                connectionStatus.style.backgroundColor = '#e6f7ff';
                connectionStatus.style.borderLeft = '4px solid #1890ff';
                connectionStatus.innerHTML = 
                    '<div class="spinner" style="display: inline-block; width: 20px; height: 20px; margin-right: 10px;"></div>' +
                    '<span>Testing connection to MLflow server at ' + serverUrl + '...</span>';
                
                // Test with a simple question
                const payload = {{
                    questions: ["Test connection"],
                    server_url: serverUrl
                }};
                
                fetch('/query_mlflow', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify(payload)
                }})
                .then(response => {{
                    if (!response.ok) {{
                        return response.json().then(errorData => {{
                            throw new Error(errorData.detail || 'HTTP error! Status: ' + response.status);
                        }});
                    }}
                    return response.json();
                }})
                .then(data => {{
                    // Connection successful
                    connectionStatus.style.backgroundColor = '#d4edda';
                    connectionStatus.style.borderLeft = '4px solid #28a745';
                    connectionStatus.innerHTML = 
                        '<i class="fas fa-check-circle"></i> Successfully connected to MLflow server at ' + serverUrl + 
                        '<div style="margin-top:8px;font-size:12px;">Connection established. The server is using format: ' + 
                        (data.format || 'unknown') + '</div>';
                    
                    // Auto-enable the "Use MLflow Server" option
                    document.getElementById('use-mlflow-server').checked = true;
                }})
                .catch(error => {{
                    // Connection failed - show more details about the error
                    connectionStatus.style.backgroundColor = '#f8d7da';
                    connectionStatus.style.borderLeft = '4px solid #dc3545';
                    connectionStatus.innerHTML = '<i class="fas fa-exclamation-circle"></i> Error connecting to MLflow server: ' + 
                        (error.message || 'Unknown error') + 
                        '<div style="margin-top:8px;font-size:12px;">Make sure the MLflow server is running and supports the /invocations endpoint.</div>';
                }});
            }});
            
            // Switch model button
            document.getElementById('switch-model-btn').addEventListener('click', function() {{
                const modelName = document.getElementById('model-name').value.trim();
                
                if (!modelName) {{
                    alert('Please enter a model name.');
                    return;
                }}
                
                // Show loader in the model status area
                const modelStatus = document.getElementById('model-status');
                modelStatus.style.display = 'block';
                modelStatus.className = '';
                modelStatus.innerHTML = 
                     '<div class="spinner" style="display: inline-block; width: 20px; height: 20px; margin-right: 10px;"></div>' +
                     '<span>Switching to model: ' + modelName + '...</span>';
                
                // Determine endpoint based on whether MLflow server mode is enabled
                const useMLflowServer = document.getElementById('use-mlflow-server').checked;
                const endpoint = useMLflowServer ? '/switch_mlflow_model' : '/switch_model';
                
                // Create payload
                const payload = {{
                    model_name: modelName
                }};
                
                // Make request
                fetch(endpoint, {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify(payload)
                }})
                .then(response => response.json())
                .then(data => {{
                    if (data.success) {{
                        // Update the model name display
                        document.getElementById('current-model').textContent = 'Model: ' + data.model_name;
                        
                        // Show success message
                        modelStatus.className = 'success';
                        modelStatus.innerHTML = '<i class="fas fa-check-circle"></i> ' + data.message;
                    }} else {{
                        // Show error message
                        modelStatus.className = 'error';
                        modelStatus.innerHTML = '<i class="fas fa-exclamation-circle"></i> ' + data.error;
                    }}
                }})
                .catch(error => {{
                    // Show error message
                    modelStatus.className = 'error';
                    modelStatus.innerHTML = '<i class="fas fa-exclamation-circle"></i> Error: ' + (error.message || 'An unknown error occurred');
                }});
            }});

            // MLflow Server Toggle
            document.getElementById('use-mlflow-server').addEventListener('change', function() {{
                const isEnabled = this.checked;
                const serverUrl = document.getElementById('mlflow-server-url').value.trim();
                
                if (!serverUrl && isEnabled) {{
                    alert('Please enter a MLflow server URL.');
                    this.checked = false;
                    return;
                }}
                
                // Update the MLflow server settings
                const payload = {{
                    enabled: isEnabled,
                    server_url: serverUrl
                }};
                
                fetch('/toggle_mlflow_server', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify(payload)
                }})
                .then(response => response.json())
                .then(data => {{
                    if (data.success) {{
                        const connectionStatus = document.getElementById('mlflow-connection-status');
                        connectionStatus.style.display = 'block';
                        connectionStatus.style.backgroundColor = '#d4edda';
                        connectionStatus.style.borderLeft = '4px solid #28a745';
                        connectionStatus.innerHTML = 
                            '<i class="fas fa-' + (isEnabled ? 'check' : 'info') + '-circle"></i> ' + 
                            (isEnabled ? 
                            'MLflow server mode enabled. Predictions will be sent to ' + data.server_url :
                            'MLflow server mode disabled. Using local model for predictions.');
                    }} else {{
                        // Reset checkbox if it failed
                        this.checked = !isEnabled;
                        alert('Failed to update MLflow server settings.');
                    }}
                }})
                .catch(error => {{
                    // Reset checkbox if it failed
                    this.checked = !isEnabled;
                    alert('Error updating MLflow server settings: ' + error.message);
                }});
            }});
        </script>
    </body>
    </html>
    """

# Add a test route that supports GET for browser testing
@app.get("/test", response_class=HTMLResponse)
async def test_form():
    return await home()

# Function to change model stage from command line
def change_model_stage(model_name, model_version, new_stage):
    """
    Change the stage of a model version in MLflow.
    
    Args:
        model_name (str): The name of the registered model
        model_version (str): The version of the model to change
        new_stage (str): The new stage ('None', 'Staging', 'Production', 'Archived')
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Validate stage value
    valid_stages = ["None", "Staging", "Production", "Archived"]
    if new_stage not in valid_stages:
        logger.error(f"Invalid stage: {new_stage}. Must be one of: {', '.join(valid_stages)}")
        return False
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to {MLFLOW_TRACKING_URI}")
        
        # Use MLflow client to change the model stage
        client = mlflow.tracking.MlflowClient()
        
        # Get current stage to report change
        model_details = client.get_model_version(name=model_name, version=model_version)
        old_stage = model_details.current_stage
        
        # Update the model version's stage
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=new_stage
        )
        
        logger.info(f"Successfully changed model '{model_name}' version {model_version} stage from {old_stage} to {new_stage}")
        return True
        
    except Exception as e:
        logger.error(f"Error changing model stage: {e}")
        import traceback
        traceback.print_exc()
        return False

def query_mlflow_server(questions: List[str], server_url: str = None, model_type: str = None, model_name: str = None) -> List[int]:
    """
    Query an MLflow model server directly using the /invocations endpoint.
    
    Args:
        questions: List of questions to classify
        server_url: URL of the MLflow model server, defaults to localhost:5000
        model_type: Type of model ("deberta", "embedding", "traditional") for optimal input format
        model_name: Optional model name for model switching when using MLflow server
        
    Returns:
        List of topic IDs
    """
    global MODEL_NAME, vectorizer, sentence_transformer, embedding_model_name, feature_eng
    global is_deberta_model, is_embedding_model, is_traditional_model

    if server_url is None:
        server_url = "http://localhost:5000"
    
    # MLflow model server expects Content-Type application/json
    headers = {
        "Content-Type": "application/json",
    }
    
    # Check if this is a model switching request
    if model_name is not None and model_name != MODEL_NAME:
        # Update the model name locally to match what we're requesting from the server
        old_model_name = MODEL_NAME
        MODEL_NAME = model_name
        logger.info(f"Switching to model {MODEL_NAME} on MLflow server at {server_url}")
        
        # Reset model-specific flags
        is_deberta_model = False
        is_embedding_model = False
        is_traditional_model = False
        
        # Set the flag based on model_type
        if model_type == "deberta":
            is_deberta_model = True
        elif model_type == "embedding":
            is_embedding_model = True
        elif model_type == "traditional":
            is_traditional_model = True
        
        # Call init_model to refresh vectorizer, etc. based on the new model name
        # We only use this to refresh local artifacts, not the model itself
        try:
            init_model()
        except Exception as e:
            logger.warning(f"Failed to initialize local artifacts for {MODEL_NAME}: {e}")
    
    # Determine payload format based on model type
    if model_type is None:
        # Infer from global flags
        if is_deberta_model:
            model_type = "deberta"
        elif is_embedding_model:
            model_type = "embedding"
        else:
            model_type = "traditional"
    
    # Call MLflow server's invocations endpoint
    invocations_url = f"{server_url}/invocations"
    logger.info(f"Querying MLflow server at {invocations_url} with model type: {model_type}")
    
    try:
        response = None
        
        # Choose the appropriate payload format based on model type
        if model_type == "deberta":
            # DeBERTa models usually expect simple string inputs
            logger.info("Using DeBERTa input format: direct text input")
            # For DeBERTa, we can just send the raw text
            payload = {"inputs": questions}
            
            response = requests.post(
                invocations_url, 
                json=payload,
                headers=headers
            )
            
            # If that fails, try dataframe format
            if response.status_code == 400:
                logger.warning("Direct input format failed, trying dataframe format")
                import pandas as pd
                df = pd.DataFrame({"Question": questions})
                payload = df.to_dict(orient="split")
                
                response = requests.post(
                    invocations_url, 
                    json=payload,
                    headers=headers
                )
                
        elif model_type == "embedding":
            # For embedding models, we need to preprocess the text and prepare all the features
            logger.info("Using embedding-specific preprocessing for MLflow server")
            
            try:
                # Check if we have all necessary artifacts
                if vectorizer is None or preprocessor is None:
                    logger.warning("Missing vectorizer or preprocessor. Trying to initialize.")
                    init_model()
                    if vectorizer is None or preprocessor is None:
                        logger.error("Missing required artifacts even after initialization.")
                        return preprocessing_query_mlflow_server(questions, server_url)
                
                # *** IMPORTANT: This must match EXACTLY how the local embedding model works ***
                # Step 1: Clean text for embeddings
                cleaned_text = [clean_text(q) for q in questions]
                logger.info("Cleaned text for embeddings")
                
                # Step 2: Preprocess text for TF-IDF
                processed_text = preprocessor.preprocess(questions)
                logger.info(f"Preprocessed text successfully: {len(processed_text)} questions")
                
                # Step 3: Generate TF-IDF features
                # CRITICAL: Use processed_text for TF-IDF, not cleaned_text
                X_tfidf = vectorizer.transform(processed_text)
                logger.info(f"Generated TF-IDF features with shape: {X_tfidf.shape}")
                
                # Step 4: Generate sentence embeddings - if available locally, send them
                # Otherwise, send the cleaned text for server-side embedding
                embeddings = None
                if sentence_transformer is not None:
                    # Generate embeddings locally to ensure consistency
                    logger.info("Generating sentence embeddings locally for consistency")
                    embeddings = sentence_transformer.encode(cleaned_text)
                    logger.info(f"Generated embeddings with shape: {embeddings.shape}")
                
                # Step 5: Generate engineered features
                features_df = engineer_features(questions)
                logger.info(f"Generated engineered features with shape: {features_df.shape}")
                
                # Step 6: Convert to sparse matrices for combining
                from scipy.sparse import hstack, csr_matrix
                
                # If we have embeddings, include them. Otherwise, let the server generate them
                if embeddings is not None:
                    # Convert to matrices
                    embeddings_sparse = csr_matrix(embeddings)
                    features_sparse = csr_matrix(features_df.values)
                    
                    # Combine all features as in train_embeddings.py AND in the local predict function
                    X_combined = hstack([X_tfidf, embeddings_sparse, features_sparse]).tocsr()
                    logger.info(f"Combined all features locally with shape: {X_combined.shape}")
                    
                    # Use dense format directly since sparse format fails
                    logger.info("Using dense format for MLflow server request (skipping sparse format)")
                    payload = {"inputs": X_combined.toarray().tolist()}
                else:
                    # Use dense format for separate components
                    logger.info("Using dense format for separate components")
                    payload = {
                        "inputs": {
                            "tfidf_features": X_tfidf.toarray().tolist(),
                            "engineered_features": features_df.values.tolist(),
                            "cleaned_questions": cleaned_text,
                            "model_type": "embedding"
                        }
                    }
                
                # Make the request
                response = requests.post(
                    invocations_url,
                    json=payload,
                    headers=headers
                )
            except Exception as e:
                logger.error(f"Error preparing embedding model input: {e}")
                # Fall back to the preprocessing function
                return preprocessing_query_mlflow_server(questions, server_url)
        else:  # traditional model
            # Try common MLflow payload formats in sequence
            logger.info("Using traditional model input format")
            import pandas as pd
            response = None
            
            # First, check if we need to preprocess
            if vectorizer is not None and preprocessor is not None:
                # Preprocess and vectorize locally
                processed_questions = preprocessor.preprocess(questions)
                X = vectorizer.transform(processed_questions)
                
                # Check if this model uses embedding features (large feature size usually indicates embeddings)
                has_embedding_features = X.shape[1] > 1000
                
                if has_embedding_features:
                    # Use dense format directly for models with embedding features to avoid sparse format failures
                    logger.info(f"Using dense format directly for model with large feature size ({X.shape[1]})")
                    payload = {"inputs": X.toarray().tolist()}
                else:
                    # Create a sparse matrix format for efficient transfer
                    payload = {
                        "sparse_input": {
                            "data": X.data.tolist(),
                            "indices": X.indices.tolist(),
                            "indptr": X.indptr.tolist(),
                            "shape": X.shape
                        }
                    }
                
                # Make the request with vectorized data
                response = requests.post(
                    invocations_url,
                    json=payload,
                    headers=headers
                )
                
                # If sparse format fails, try dense format
                if not has_embedding_features and response.status_code == 400:
                    logger.warning("Sparse format failed, trying dense format")
                    payload = {"inputs": X.toarray().tolist()}
                    
                    response = requests.post(
                        invocations_url,
                        json=payload,
                        headers=headers
                    )
            
            # If local preprocessing fails or no vectorizer/preprocessor, try standard formats
            if response is None or response.status_code == 400:
                logger.warning("Local preprocessing failed or not available, trying standard formats")
                # Convert to DataFrame, then to dict without column names
                df = pd.DataFrame(data=[[q] for q in questions])
                payload = df.to_dict(orient="split")
                # Remove column names to avoid scikit-learn warning
                if "columns" in payload:
                    del payload["columns"]
                
                response = requests.post(
                    invocations_url, 
                    json=payload,
                    headers=headers
                )
                
                # 2. Try standard dataframe_split format
                if response.status_code == 400:
                    logger.warning("First payload format failed, trying dataframe_split with feature names")
                    payload = {
                        "dataframe_split": {
                            "columns": ["text"],
                            "data": [[q] for q in questions]
                        }
                    }
                    response = requests.post(
                        invocations_url, 
                        json=payload,
                        headers=headers
                    )
                
                # 3. Try simple list format
                if response.status_code == 400:
                    logger.warning("Second payload format failed, trying inputs format")
                    payload = {"inputs": questions}
                    response = requests.post(
                        invocations_url, 
                        json=payload,
                        headers=headers
                    )
                
                # 4. Try instances format
                if response.status_code == 400:
                    logger.warning("Third payload format failed, trying instances format")
                    payload = {"instances": questions}
                    response = requests.post(
                        invocations_url, 
                        json=payload,
                        headers=headers
                    )
        
        # Check final response
        if response.status_code != 200:
            logger.error(f"MLflow server returned status code {response.status_code}: {response.text}")
            raise RuntimeError(f"Failed to get predictions from MLflow server: {response.text}")
        
        # Parse response (MLflow returns raw predictions)
        predictions = response.json()
        logger.info(f"Received predictions from MLflow server: {predictions}")
        
        # Handle different response formats
        if isinstance(predictions, dict):
            if "predictions" in predictions:
                predictions = predictions["predictions"]
            elif "result" in predictions:
                predictions = predictions["result"]
                
        # Convert to integers if needed
        result_predictions = []
        if isinstance(predictions, list):
            for pred in predictions:
                if isinstance(pred, list):
                    # Handle nested lists
                    result_predictions.append(int(pred[0]) if pred else 0)
                elif isinstance(pred, dict) and "prediction" in pred:
                    # Handle dictionary format
                    result_predictions.append(int(pred["prediction"]))
                else:
                    # Handle direct values
                    result_predictions.append(int(pred))
        
        return result_predictions
    except Exception as e:
        logger.error(f"Error querying MLflow server: {e}")
        raise

# Add a new endpoint to query MLflow server directly
@app.post("/query_mlflow")
async def query_mlflow_endpoint(request: MLflowServerQuery):
    """
    Query an MLflow model server and return formatted predictions
    """
    try:
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="No questions provided for classification")
            
        logger.info(f"Querying MLflow server with {len(request.questions)} questions")
        
        # Handle model switching if requested
        model_name = request.model_name
        model_type = request.model_type
        
        # Special case for testing connection - we'll use the simple format
        if len(request.questions) == 1 and request.questions[0].lower().strip() == "test connection":
            logger.info("Test connection detected, using simple query format")
            try:
                predictions = query_mlflow_server(
                    questions=request.questions,
                    server_url=request.server_url,
                    model_type=model_type,
                    model_name=model_name
                )
                # If this works, great. If not, we'll try the preprocessing approach below
            except Exception as e:
                logger.warning(f"Simple query format failed for test connection: {e}")
                # Fall back to the preprocessing approach
                predictions = preprocessing_query_mlflow_server(
                    questions=request.questions,
                    server_url=request.server_url
                )
        elif model_type == "embedding":
            # For embedding models, we need to be careful about preprocessing
            logger.info("Using embedding model-specific preprocessing for MLflow query")
            try:
                # Try the regular query_mlflow_server with embedding model_type first
                predictions = query_mlflow_server(
                    questions=request.questions,
                    server_url=request.server_url,
                    model_type=model_type,
                    model_name=model_name
                )
            except Exception as e:
                logger.warning(f"Embedding-specific query failed: {e}, falling back to full preprocessing")
                # If that fails, use the comprehensive preprocessing approach
                predictions = preprocessing_query_mlflow_server(
                    questions=request.questions,
                    server_url=request.server_url
                )
        else:
            # For traditional or DeBERTa models, use the regular query function
            logger.info(f"Using standard query format for {model_type if model_type else 'unknown'} model type")
            predictions = query_mlflow_server(
                questions=request.questions,
                server_url=request.server_url,
                model_type=model_type,
                model_name=model_name
            )
        
        # Format the predictions as a response
        formatted_predictions = []
        for i, pred in enumerate(predictions):
            topic_id = int(pred)
            topic_name = topic_mapping.get(topic_id, f"Unknown Topic {topic_id}")
            
            formatted_predictions.append(
                Prediction(
                    question=request.questions[i],
                    topic_id=topic_id,
                    topic_name=topic_name
                )
            )
        
        return PredictionResponse(predictions=formatted_predictions)
        
    except Exception as e:
        logger.error(f"Error in query_mlflow_endpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Add a more comprehensive function that preprocesses data like the training pipeline
def preprocessing_query_mlflow_server(questions: List[str], server_url: str = None) -> List[int]:
    """
    Preprocess questions the same way as in local predict and train_embeddings.py, then query MLflow server.
    This function handles the complete preprocessing pipeline as done in predict method.
    
    Args:
        questions: List of questions to classify
        server_url: URL of the MLflow model server, defaults to localhost:5000
        
    Returns:
        List of topic IDs
    """
    global vectorizer, sentence_transformer, embedding_model_name
    
    if server_url is None:
        server_url = "http://localhost:5000"
    
    # MLflow invocations endpoint
    invocations_url = f"{server_url}/invocations"
    logger.info(f"Full preprocessing with local embedding generation before querying MLflow server at {invocations_url}")
    
    try:
        # Steps that exactly match the local predict method for embedding models
        
        # Step 1: Clean text for embeddings
        cleaned_text = [clean_text(q) for q in questions]
        logger.info("Cleaned text for embeddings")
        
        # Step 2: Preprocess text for TF-IDF
        if preprocessor is None:
            logger.warning("Preprocessor not found. Attempting to initialize model.")
            init_model()
            if preprocessor is None:
                raise RuntimeError("Preprocessor not found. Cannot make predictions.")
                
        processed_text = preprocessor.preprocess(questions)
        logger.info(f"Preprocessed text successfully: {len(processed_text)} questions")
        
        # Step 3: Generate TF-IDF features
        if vectorizer is None:
            logger.warning("Vectorizer not found. Attempting to initialize model.")
            init_model()
            if vectorizer is None:
                raise RuntimeError("Vectorizer not found. Cannot make predictions.")
        
        X_tfidf = vectorizer.transform(processed_text)
        logger.info(f"Generated TF-IDF features with shape: {X_tfidf.shape}")
        
        # Step 4: Generate sentence embeddings
        if sentence_transformer is None:
            try:
                # Make sure we have an embedding model name
                if embedding_model_name is None or embedding_model_name == DEFAULT_EMBEDDING_MODEL:
                    # Check model name to infer embedding model
                    if "minilm" in MODEL_NAME.lower():
                        # Use MiniLM model
                        embedding_model_name = "all-MiniLM-L6-v2"
                    elif "mpnet" in MODEL_NAME.lower():
                        # Use MPNet model
                        embedding_model_name = "all-mpnet-base-v2"
                    elif "bert" in MODEL_NAME.lower() and not "deberta" in MODEL_NAME.lower():
                        # Use BERT model
                        embedding_model_name = "bert-base-uncased"
                    else:
                        # Default is MiniLM
                        embedding_model_name = DEFAULT_EMBEDDING_MODEL
                          
                logger.info(f"Loading SentenceTransformer with model: {embedding_model_name}")
                from sentence_transformers import SentenceTransformer
                sentence_transformer = SentenceTransformer(embedding_model_name)
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer: {e}")
                raise RuntimeError(f"Cannot generate embeddings for prediction: {e}")
        
        # Generate the embeddings from cleaned text (not processed text)
        logger.info("Generating sentence embeddings")
        embeddings = sentence_transformer.encode(cleaned_text)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Step 5: Generate engineered features
        logger.info("Engineering additional features")
        # Engineer features from the original questions, not the cleaned ones
        features_df = engineer_features(questions)
        logger.info(f"Generated engineered features with shape: {features_df.shape}")
        
        # Step 6: Combine all features EXACTLY as in local predict method
        logger.info("Combining all features")
        from scipy.sparse import hstack, csr_matrix
        
        # Convert to sparse matrices
        embeddings_sparse = csr_matrix(embeddings)
        features_sparse = csr_matrix(features_df.values)
        
        # Combine all features - must be in the exact same order as training
        X_combined = hstack([X_tfidf, embeddings_sparse, features_sparse]).tocsr()
        logger.info(f"Combined all features with shape: {X_combined.shape}")
        
        # Use dense format directly since sparse format fails
        logger.info("Using dense format for MLflow server request (skipping sparse format)")
        payload = {"inputs": X_combined.toarray().tolist()}
        
        # Make request to MLflow server
        logger.info("Sending request to MLflow server with fully preprocessed features")
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(
            invocations_url,
            json=payload,
            headers=headers
        )
        
        # Check response
        if response.status_code != 200:
            logger.error(f"MLflow server returned status code {response.status_code}: {response.text}")
            raise RuntimeError(f"Failed to get predictions from MLflow server: {response.text}")
        
        # Parse response
        predictions = response.json()
        logger.info(f"Received predictions from MLflow server: {predictions}")
        
        # Handle different response formats
        if isinstance(predictions, dict):
            if "predictions" in predictions:
                predictions = predictions["predictions"]
            elif "result" in predictions:
                predictions = predictions["result"]
                
        # Convert to integers
        result_predictions = []
        if isinstance(predictions, list):
            for pred in predictions:
                if isinstance(pred, list):
                    # Handle nested lists
                    result_predictions.append(int(pred[0]) if pred else 0)
                elif isinstance(pred, dict) and "prediction" in pred:
                    # Handle dictionary format
                    result_predictions.append(int(pred["prediction"]))
                else:
                    # Handle direct values
                    result_predictions.append(int(pred))
        
        return result_predictions
    except Exception as e:
        logger.error(f"Error in preprocessing and querying MLflow server: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

@app.post("/toggle_mlflow_server", response_model=MLflowServerToggleResponse)
async def toggle_mlflow_server(request: MLflowServerToggleRequest):
    """Toggle whether to use MLflow server for predictions"""
    global USE_MLFLOW_SERVER, MLFLOW_SERVER_URL
    
    try:
        # Update settings
        USE_MLFLOW_SERVER = request.enabled
        
        # Update server URL if provided
        if request.server_url:
            MLFLOW_SERVER_URL = request.server_url
        
        action = "enabled" if USE_MLFLOW_SERVER else "disabled"
        logger.info(f"MLflow server mode {action}. Server URL: {MLFLOW_SERVER_URL}")
        
        return MLflowServerToggleResponse(
            success=True,
            message=f"MLflow server mode {action}",
            enabled=USE_MLFLOW_SERVER,
            server_url=MLFLOW_SERVER_URL
        )
        
    except Exception as e:
        logger.error(f"Error toggling MLflow server mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/switch_mlflow_model", response_model=ModelSwitchResponse)
async def switch_mlflow_model(request: ModelSwitchRequest):
    """Switch model when using MLflow server"""
    global USE_MLFLOW_SERVER, MLFLOW_SERVER_URL, MODEL_NAME
    global is_deberta_model, is_embedding_model, is_traditional_model
    global sentence_transformer, embedding_model_name, vectorizer, preprocessor
    
    try:
        if not USE_MLFLOW_SERVER:
            return ModelSwitchResponse(
                success=False,
                error="MLflow server mode is not enabled. Use /switch_model for local models.",
                model_name=MODEL_NAME
            )
            
        # Store old model name for response
        old_model_name = MODEL_NAME
        
        # Update the model name
        MODEL_NAME = request.model_name
        
        # Reset all model type flags
        is_deberta_model = False
        is_embedding_model = False
        is_traditional_model = False
        
        # Set the appropriate flag based on the model name
        model_name_lower = MODEL_NAME.lower()
        if "deberta" in model_name_lower or "transformer" in model_name_lower:
            is_deberta_model = True
            logger.info(f"Set model type to DeBERTa/Transformer for MLflow model: {MODEL_NAME}")
        elif "embedding" in model_name_lower or "sentence" in model_name_lower or "minilm" in model_name_lower or "mpnet" in model_name_lower:
            is_embedding_model = True
            logger.info(f"Set model type to Embedding for MLflow model: {MODEL_NAME}")
            
            # For embedding models, attempt to load the local artifacts to match the model
            try:
                # Reset embedding-specific resources to ensure they're properly initialized for the new model
                sentence_transformer = None
                
                # If we're switching to a different embedding model, update what kind of transformer to use
                if "minilm" in model_name_lower:
                    embedding_model_name = "all-MiniLM-L6-v2"
                elif "mpnet" in model_name_lower:
                    embedding_model_name = "all-mpnet-base-v2"
                elif "bert" in model_name_lower and not "deberta" in model_name_lower:
                    embedding_model_name = "bert-base-uncased"
                
                # Try to initialize the model to get local artifacts (vectorizer, etc.)
                init_model()
                
                # Test with a sample question to verify we can process it
                test_question = ["This is a test for the embedding model."]
                test_result = preprocessing_query_mlflow_server(
                    questions=test_question,
                    server_url=MLFLOW_SERVER_URL
                )
                logger.info(f"Successfully tested embedding model with result: {test_result}")
            except Exception as e:
                logger.warning(f"Error initializing local artifacts for embedding model: {e}")
                # Continue even if this fails - the preprocessing_query_mlflow_server function has fallbacks
        else:
            is_traditional_model = True
            logger.info(f"Set model type to Traditional for MLflow model: {MODEL_NAME}")
            
            # For traditional models, ensure we have vectorizer and preprocessor
            try:
                init_model()
            except Exception as e:
                logger.warning(f"Could not initialize artifacts for traditional model: {e}")
        
        return ModelSwitchResponse(
            success=True,
            message=f"Switched to model {MODEL_NAME} in MLflow server mode",
            model_name=MODEL_NAME
        )
    except Exception as e:
        logger.error(f"Error switching MLflow model: {e}")
        return ModelSwitchResponse(
            success=False,
            error=str(e),
            model_name=MODEL_NAME
        )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Serve or manage math topic classifier models with MLflow and FastAPI')
    
    # Server configuration arguments
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=5001,
                       help='Port to bind the server to')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    
    # Model selection arguments
    parser.add_argument('--model', type=str, default=MODEL_NAME,
                       help='Name of the model to serve')
    parser.add_argument('--version', type=str,
                       help='Version of the model to serve (optional, uses stage if not specified)')
    parser.add_argument('--stage', type=str, default=MODEL_STAGE,
                       help='Stage of the model to serve (Production, Staging, None, etc.)')
    
    # Model stage management (doesn't start server, just changes stage)
    parser.add_argument('--change-stage', action='store_true',
                       help='Change model stage without starting the server')
    parser.add_argument('--target-stage', type=str,
                       help='Target stage for the model (when using --change-stage)')
    
    # MLflow serving integration
    parser.add_argument('--use-mlflow-server', action='store_true',
                       help='Use MLflow\'s built-in model server instead of FastAPI')
    
    args = parser.parse_args()
    
    # Handle stage change operation first (if requested)
    if args.change_stage:
        # Check required arguments
        if not args.model:
            logger.error("Model name (--model) is required when changing stage")
            sys.exit(1)
        if not args.version:
            logger.error("Model version (--version) is required when changing stage")
            sys.exit(1)
        if not args.target_stage:
            logger.error("Target stage (--target-stage) is required when changing stage")
            sys.exit(1)
            
        # Attempt to change stage
        logger.info(f"Attempting to change model '{args.model}' version {args.version} to stage '{args.target_stage}'")
        success = change_model_stage(args.model, args.version, args.target_stage)
        
        if success:
            logger.info("Model stage changed successfully")
            sys.exit(0)
        else:
            logger.error("Failed to change model stage")
            sys.exit(1)
    
    # Update model name and stage from command line args
    if args.model:
        MODEL_NAME = args.model
        logger.info(f"Using model name from command line: {MODEL_NAME}")
    
    if args.stage:
        MODEL_STAGE = args.stage
        logger.info(f"Using model stage from command line: {MODEL_STAGE}")
    
    # Set specific model version if provided
    if args.version:
        logger.info(f"Using specific model version: {args.version}")
        model_uri = f"models:/{MODEL_NAME}/{args.version}"
    else:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    
    logger.info(f"Model URI set to: {model_uri}")
    
    # If using MLflow's built-in server
    if args.use_mlflow_server:
        logger.info(f"Starting MLflow model server for {model_uri}")
        import subprocess
        
        # Use subprocess to call MLflow serve command
        cmd = [
            "mlflow", "models", "serve",
            "--model-uri", model_uri,
            "--port", str(args.port),
            "--host", args.host,
            "--no-conda"
        ]
        
        if args.debug:
            cmd.append("--enable-mlserver")
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        try:
            # Use subprocess instead of direct function call
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error starting MLflow model server: {e}")
            sys.exit(1)
    else:
        # Start the FastAPI server with Uvicorn
        logger.info(f"Starting FastAPI server on {args.host}:{args.port}")
        uvicorn.run(
            "serve_model:app",
            host=args.host,
            port=args.port,
            reload=args.debug,
            log_level="info" if not args.debug else "debug"
        ) 