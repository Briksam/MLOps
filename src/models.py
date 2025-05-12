import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

logger = logging.getLogger(__name__)

# Define available model types for validation
VALID_MODEL_TYPES = ['lr', 'rf', 'gb', 'svm', 'nb', 'ovr', 'ensemble', 'stacking']

def get_model(model_type='lr', **kwargs):
    """
    Get a model instance by type.
    
    Args:
        model_type (str): Model type ('lr', 'rf', 'gb', 'svm', 'nb', 'ovr')
        **kwargs: Additional arguments for the model
        
    Returns:
        model: A scikit-learn model instance
        
    Raises:
        ValueError: If model_type is not recognized
    """
    models = {
        'lr': LogisticRegression,
        'rf': RandomForestClassifier,
        'gb': GradientBoostingClassifier,
        'svm': LinearSVC,
        'nb': MultinomialNB,
        'ovr': OneVsRestClassifier
    }
    
    if model_type not in models:
        # Provide a helpful error message with available models
        available_models = list(models.keys())
        error_msg = f"Model type '{model_type}' not recognized. Available models: {', '.join(available_models)}"
        logger.error(error_msg)
        
        if model_type in ['ensemble', 'stacking']:
            error_msg += f". For '{model_type}', use get_ensemble_model() or get_stacking_model() instead."
            logger.error(error_msg)
            
        raise ValueError(error_msg)
    
    logger.info(f"Creating model of type: {model_type}")
    
    try:
        if model_type == 'ovr':
            # Special case for OneVsRest
            base_classifier = kwargs.pop('base_classifier', LogisticRegression())
            return models[model_type](base_classifier, **kwargs)
        
        return models[model_type](**kwargs)
    except Exception as e:
        logger.error(f"Error creating model of type {model_type}: {e}")
        raise

# Define default hyperparameters for each model type
DEFAULT_PARAMS = {
    'lr': {
        'C': 1.0,
        'max_iter': 1000,
        'penalty': 'l2',
        'solver': 'liblinear',
        'multi_class': 'ovr',
        'random_state': 42
    },
    'rf': {
        'n_estimators': 100,
        'max_depth': 30,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    },
    'gb': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42
    },
    'svm': {
        'C': 1.0,
        'tol': 1e-4,
        'dual': False,
        'random_state': 42,
        'max_iter': 1000
    },
    'nb': {
        'alpha': 1.0
    }
}

def validate_parameters(model_type, params):
    """
    Validate parameters for a specific model type.
    
    Args:
        model_type (str): Type of model
        params (dict): Parameters to validate
        
    Returns:
        dict: Valid parameters for the model
        
    Raises:
        ValueError: If invalid parameters are provided
    """
    # Check if model_type is valid
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"Invalid model type: {model_type}. Valid types: {', '.join(VALID_MODEL_TYPES)}")
    
    # For ensemble and stacking, just return params (validated elsewhere)
    if model_type in ['ensemble', 'stacking']:
        return params
    
    # Get model class to check valid parameters
    model_classes = {
        'lr': LogisticRegression,
        'rf': RandomForestClassifier,
        'gb': GradientBoostingClassifier,
        'svm': LinearSVC,
        'nb': MultinomialNB,
        'ovr': OneVsRestClassifier
    }
    
    # Create empty model to check valid parameters
    if model_type in model_classes:
        try:
            valid_params = {}
            model_class = model_classes[model_type]
            
            # Get valid parameters for this model class
            dummy_model = model_class()
            valid_param_names = dummy_model.get_params().keys()
            
            # Filter out invalid parameters
            for param, value in params.items():
                if param in valid_param_names:
                    valid_params[param] = value
                else:
                    logger.warning(f"Parameter '{param}' is not valid for model type '{model_type}' and will be ignored")
            
            return valid_params
        except Exception as e:
            logger.error(f"Error validating parameters for {model_type}: {e}")
            # If validation fails, just return the original params
            return params
    
    return params

def get_model_with_default_params(model_type='lr', override_params=None):
    """
    Get a model with default parameters for the given model type.
    
    Args:
        model_type (str): Model type ('lr', 'rf', 'gb', 'svm', 'nb')
        override_params (dict): Parameters to override default ones
        
    Returns:
        model: A scikit-learn model instance with default parameters
    """
    # Get default parameters for this model type
    params = DEFAULT_PARAMS.get(model_type, {}).copy()
    
    # Override with custom parameters if provided
    if override_params:
        params.update(override_params)
    
    # Validate parameters
    params = validate_parameters(model_type, params)
    
    # Special case for OneVsRest classifier
    if model_type == 'ovr':
        base_type = override_params.get('base_type', 'lr')
        base_params = DEFAULT_PARAMS.get(base_type, {}).copy()
        if 'base_params' in override_params:
            base_params.update(override_params['base_params'])
        
        base_classifier = get_model(base_type, **base_params)
        params.pop('base_type', None)
        params.pop('base_params', None)
        return get_model(model_type, base_classifier=base_classifier, **params)
    
    try:
        return get_model(model_type, **params)
    except ValueError as e:
        if model_type == 'ensemble':
            logger.info("Creating ensemble model with default configuration")
            return get_ensemble_model()
        elif model_type == 'stacking':
            logger.info("Creating stacking model with default configuration")
            return get_stacking_model()
        else:
            raise e

def create_text_classification_pipeline(preprocessor=None, vectorizer=None, model_type='lr', model_params=None):
    """
    Create a text classification pipeline.
    
    Args:
        preprocessor: Text preprocessor with a preprocess method
        vectorizer: A feature vectorizer (e.g., TfidfVectorizer)
        model_type (str): Type of model to use
        model_params (dict): Parameters for the model
        
    Returns:
        pipeline: A scikit-learn pipeline for text classification
        
    Raises:
        ValueError: If model creation fails
    """
    steps = []
    
    # Add vectorizer if provided
    if vectorizer is not None:
        steps.append(('vectorizer', vectorizer))
    
    try:
        # Get model with default or custom parameters
        model = get_model_with_default_params(model_type, model_params)
        steps.append(('classifier', model))
        
        # Create pipeline
        pipeline = Pipeline(steps)
        
        return pipeline
    except Exception as e:
        logger.error(f"Error creating pipeline with model type {model_type}: {e}")
        raise

# Collection of advanced models
def get_ensemble_model(models=None, voting='soft', weights=None):
    """
    Create a voting ensemble of multiple models.
    
    Args:
        models: List of (name, model) tuples for ensemble
        voting: Voting strategy ('hard' or 'soft')
        weights: Model weights for weighted voting
        
    Returns:
        VotingClassifier: Ensemble model
    """
    from sklearn.ensemble import VotingClassifier
    
    if models is None:
        # Default ensemble of different model types
        models = [
            ('lr', get_model_with_default_params('lr')),
            ('rf', get_model_with_default_params('rf')),
            ('gb', get_model_with_default_params('gb'))
        ]
    
    # Validate models format
    if not all(isinstance(m, tuple) and len(m) == 2 for m in models):
        logger.error("Invalid model format for ensemble. Each model should be a (name, model) tuple.")
        raise ValueError("Invalid model format for ensemble. Each model should be a (name, model) tuple.")
    
    # Validate voting parameter
    if voting not in ['hard', 'soft']:
        logger.warning(f"Invalid voting parameter '{voting}'. Must be 'hard' or 'soft'. Using 'soft' as default.")
        voting = 'soft'
    
    return VotingClassifier(
        estimators=models,
        voting=voting,
        weights=weights
    )

def get_stacking_model(base_models=None, meta_model=None):
    """
    Create a stacking ensemble model.
    
    Args:
        base_models: List of (name, model) tuples for base models
        meta_model: Meta-classifier for stacking
        
    Returns:
        StackingClassifier: Stacking ensemble model
    """
    from sklearn.ensemble import StackingClassifier
    
    if base_models is None:
        # Default base models
        base_models = [
            ('lr', get_model_with_default_params('lr')),
            ('rf', get_model_with_default_params('rf')),
            ('nb', get_model_with_default_params('nb'))
        ]
    
    if meta_model is None:
        meta_model = get_model_with_default_params('gb')
    
    # Validate base_models format
    if not all(isinstance(m, tuple) and len(m) == 2 for m in base_models):
        logger.error("Invalid model format for stacking. Each model should be a (name, model) tuple.")
        raise ValueError("Invalid model format for stacking. Each model should be a (name, model) tuple.")
    
    return StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    ) 