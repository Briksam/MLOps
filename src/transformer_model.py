import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AdamW, 
    get_scheduler,
    DebertaV2Config
)
from tqdm.auto import tqdm
import mlflow
import joblib


logger = logging.getLogger(__name__)

class MathQuestionDataset(Dataset):
    """Dataset for math question classification."""
    
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        """
        Initialize dataset.
        
        Args:
            texts: List of text samples
            labels: List of labels (optional)
            tokenizer: Transformer tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize input text
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert to appropriate format
        item = {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
        }
        
        # Add label if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

class DebertaClassifier(BaseEstimator, ClassifierMixin):
    """DeBERTa-based classifier with feature engineering integration for math question classification."""
    
    def __init__(
        self,
        model_name="microsoft/deberta-v3-xsmall",
        num_labels=8,
        max_length=128,
        batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        epochs=3,
        warmup_ratio=0.1,
        enable_fp16=False,
        random_state=42,
        device=None,
        feature_preprocessor=None,
        feature_integration_mode="hybrid",  # Can be "none", "hybrid", or "ensemble"
        feature_weight=0.3  # Weight given to engineered features in hybrid mode
    ):
        """
        Initialize the DeBERTa classifier with feature engineering integration.
        
        Args:
            model_name: Pretrained model name
            num_labels: Number of classes
            max_length: Maximum sequence length
            batch_size: Training batch size
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            epochs: Number of training epochs
            warmup_ratio: Ratio of warmup steps
            enable_fp16: Enable mixed precision training
            random_state: Random state for reproducibility
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            feature_preprocessor: Optional feature engineering preprocessor
            feature_integration_mode: How to integrate engineered features ("none", "hybrid", "ensemble")
            feature_weight: Weight given to engineered features (0-1) in hybrid mode
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.enable_fp16 = enable_fp16
        self.random_state = random_state
        self.feature_preprocessor = feature_preprocessor
        self.feature_integration_mode = feature_integration_mode
        self.feature_weight = feature_weight
        
        # Initialize a classifier for engineered features
        self.feature_classifier = None
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self._is_fitted = False
    
    def _init_model_tokenizer(self):
        """Initialize model and tokenizer."""
        logger.info(f"Initializing tokenizer and model with {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model
        model_config = DebertaV2Config.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            output_hidden_states=False
        )
        
        # Reduce model size with smaller attention heads and layers for speed
        if "xsmall" in self.model_name:
            # Ensure these parameters are respected for xsmall model
            logger.info("Using optimized configuration for CPU training")
            # No changes needed, just use the preset xsmall model
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=model_config
        )
        
        # Move model to device
        self.model.to(self.device)
    
    def fit(self, X, y):
        """
        Train the model.
        
        Args:
            X: Input features or text
            y: Target labels
        
        Returns:
            self: Trained model
        """
        # Initialize model and tokenizer
        if self.tokenizer is None or self.model is None:
            self._init_model_tokenizer()
        
        # Convert inputs to appropriate format
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Feature engineering integration
        if self.feature_preprocessor is not None and self.feature_integration_mode != "none":
            logger.info(f"Using feature engineering with integration mode: {self.feature_integration_mode}")
            # Fit and transform engineered features
            X_features_df = self.feature_preprocessor.fit_transform(X)
            
            # Train a classifier on engineered features
            from sklearn.ensemble import RandomForestClassifier
            self.feature_classifier = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            )
            self.feature_classifier.fit(X_features_df, y)
            logger.info(f"Trained feature classifier with {X_features_df.shape[1]} engineered features")
        
        # Prepare dataset for transformer
        train_dataset = MathQuestionDataset(
            texts=X,
            labels=y,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Create data loader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Set up optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Set up learning rate scheduler
        num_training_steps = self.epochs * len(train_dataloader)
        num_warmup_steps = int(self.warmup_ratio * num_training_steps)
        
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Set up mixed precision training if enabled and available
        if self.enable_fp16 and torch.cuda.is_available():
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            amp_enabled = True
        else:
            amp_enabled = False
        
        # Training loop
        self.model.train()
        logger.info(f"Starting training for {self.epochs} epochs")
        
        for epoch in range(self.epochs):
            total_loss = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with or without mixed precision
                if amp_enabled:
                    with autocast():
                        outputs = self.model(**batch)
                        loss = outputs.loss
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    # Standard forward and backward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Update learning rate
                lr_scheduler.step()
                
                # Update progress bar
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Log epoch loss
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1}/{self.epochs} - Average loss: {avg_loss:.4f}")
            
            # Log to MLflow if active run exists
            if mlflow.active_run():
                mlflow.log_metric(f"train_loss_epoch_{epoch+1}", avg_loss)
        
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions using a hybrid approach.
        
        Args:
            X: Input features or text
        
        Returns:
            y_pred: Predicted labels
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Get probabilities and take argmax
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using a hybrid approach.
        
        Args:
            X: Input features or text
        
        Returns:
            y_proba: Class probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Convert inputs to appropriate format
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Get transformer predictions
        transformer_proba = self._predict_transformer_proba(X)
        
        # If no feature engineering integration, return transformer predictions
        if self.feature_preprocessor is None or self.feature_integration_mode == "none" or self.feature_classifier is None:
            return transformer_proba
        
        # Get engineered feature predictions
        feature_proba = self._predict_feature_proba(X)
        
        # Combine predictions based on integration mode
        if self.feature_integration_mode == "hybrid":
            # Weighted average of transformer and feature probabilities
            combined_proba = (1 - self.feature_weight) * transformer_proba + self.feature_weight * feature_proba
            return combined_proba
        elif self.feature_integration_mode == "ensemble":
            # Product of probabilities (ensembling approach)
            combined_proba = transformer_proba * feature_proba
            # Normalize
            row_sums = combined_proba.sum(axis=1, keepdims=True)
            return combined_proba / row_sums
        else:
            # Default to transformer predictions if invalid mode
            return transformer_proba
    
    def _predict_transformer_proba(self, X):
        """Get probability predictions from the transformer model."""
        # Prepare dataset
        test_dataset = MathQuestionDataset(
            texts=X,
            labels=None,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Create data loader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False
        )
        
        # Prediction loop
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Transformer predictions"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                logits = outputs.logits
                
                # Convert logits to probabilities
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs)
        
        return np.array(all_probs)
    
    def _predict_feature_proba(self, X):
        """Get probability predictions from the feature-based classifier."""
        # Transform inputs using feature engineering
        X_features = self.feature_preprocessor.transform(X)
        
        # Get predictions from feature classifier
        return self.feature_classifier.predict_proba(X_features)
    
    def save(self, path):
        """
        Save model and tokenizer.
        
        Args:
            path: Directory path to save model
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(os.path.join(path, "model"))
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
        
        # Save feature preprocessor if available
        if self.feature_preprocessor is not None:
            feature_eng_path = os.path.join(path, "feature_eng.joblib")
            joblib.dump(self.feature_preprocessor, feature_eng_path)
            logger.info(f"Feature engineering preprocessor saved to {feature_eng_path}")
        
        # Save feature classifier if available
        if self.feature_classifier is not None:
            feature_clf_path = os.path.join(path, "feature_classifier.joblib")
            joblib.dump(self.feature_classifier, feature_clf_path)
            logger.info(f"Feature classifier saved to {feature_clf_path}")
        
        # Save hyperparameters
        params = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'warmup_ratio': self.warmup_ratio,
            'enable_fp16': self.enable_fp16,
            'random_state': self.random_state,
            'has_feature_preprocessor': self.feature_preprocessor is not None,
            'feature_integration_mode': self.feature_integration_mode,
            'feature_weight': self.feature_weight
        }
        
        with open(os.path.join(path, "params.json"), 'w') as f:
            import json
            json.dump(params, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """
        Load model and tokenizer.
        
        Args:
            path: Directory path to load model from
        
        Returns:
            model: Loaded model
        """
        # Load hyperparameters
        with open(os.path.join(path, "params.json"), 'r') as f:
            import json
            params = json.load(f)
        
        # Remove feature preprocessor flag from params
        has_feature_preprocessor = params.pop('has_feature_preprocessor', False)
        
        # Create instance
        instance = cls(**params)
        
        # Load tokenizer
        instance.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
        
        # Load model
        instance.model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(path, "model")
        )
        
        # Move model to device
        instance.model.to(instance.device)
        
        # Load feature preprocessor if it exists
        if has_feature_preprocessor:
            feature_eng_path = os.path.join(path, "feature_eng.joblib")
            if os.path.exists(feature_eng_path):
                instance.feature_preprocessor = joblib.load(feature_eng_path)
                logger.info(f"Loaded feature engineering preprocessor from {feature_eng_path}")
        
        # Load feature classifier if it exists
        feature_clf_path = os.path.join(path, "feature_classifier.joblib")
        if os.path.exists(feature_clf_path):
            instance.feature_classifier = joblib.load(feature_clf_path)
            logger.info(f"Loaded feature classifier from {feature_clf_path}")
        
        # Mark as fitted
        instance._is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        return instance


class FeatureEngineeringPreprocessor:
    """Advanced preprocessing with math-specific feature engineering."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        from sklearn.feature_extraction.text import CountVectorizer
        import re
        
        self.math_pattern_extractors = {
            'has_equation': re.compile(r'='),
            'has_variable': re.compile(r'[a-zA-Z][\(\)\s]*='),
            'has_function': re.compile(r'f\s*\('),
            'has_integral': re.compile(r'(integral|∫)'),
            'has_derivative': re.compile(r'(derivative|differentiate|d\/dx)'),
            'has_probability': re.compile(r'(probability|chance)'),
            'has_matrix': re.compile(r'(matrix|matrices)'),
            'has_triangle': re.compile(r'triangle'),
            'has_geometry': re.compile(r'(circle|square|rectangle|polygon)'),
            'has_inequality': re.compile(r'[<>≤≥]'),
            'has_fraction': re.compile(r'(fraction|\/)'),
            'has_exponent': re.compile(r'(\^|\*\*)'),
            'has_complex': re.compile(r'(complex\s+number|imaginary)'),
            'has_algebra': re.compile(r'(factor|expand|simplify)'),
            'has_statistics': re.compile(r'(mean|median|mode|variance|standard deviation)'),
            'has_calculus': re.compile(r'(limit|differentiate|integrate|series)'),
        }
        
        # Initialize keyword vectorizer for math topics
        self.keyword_vectorizer = CountVectorizer(
            vocabulary=[
                'algebra', 'calculus', 'geometry', 'probability',
                'statistics', 'linear algebra', 'discrete mathematics',
                'solve', 'compute', 'evaluate', 'find', 'determine',
                'prove', 'calculate', 'theorem', 'identity', 'formula'
            ],
            binary=True
        )
        
        self.is_fitted = False
    
    def fit(self, texts, y=None):
        """Fit the preprocessing pipeline."""
        # Fit keyword vectorizer
        self.keyword_vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform_text(self, text):
        """Transform a single text with feature engineering."""
        # Extract math-specific patterns
        features = {name: bool(pattern.search(text)) for name, pattern in self.math_pattern_extractors.items()}
        
        # Add text length features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Add character-level features
        features['number_ratio'] = sum(c.isdigit() for c in text) / max(len(text), 1)
        features['symbol_ratio'] = sum(not c.isalnum() and not c.isspace() for c in text) / max(len(text), 1)
        
        return features
    
    def transform(self, texts):
        """Transform a list of texts with feature engineering."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # Apply text transformation to each text
        transformed_features = [self.transform_text(text) for text in texts]
        
        # Convert to DataFrame
        import pandas as pd
        feature_df = pd.DataFrame(transformed_features)
        
        # Add keyword features
        keyword_features = self.keyword_vectorizer.transform(texts).toarray()
        keyword_df = pd.DataFrame(
            keyword_features,
            columns=[f'keyword_{w}' for w in self.keyword_vectorizer.get_feature_names_out()]
        )
        
        # Combine all features
        combined_df = pd.concat([feature_df, keyword_df], axis=1)
        
        return combined_df
    
    def fit_transform(self, texts, y=None):
        """Fit and transform texts."""
        return self.fit(texts).transform(texts)


def get_deberta_classifier(num_labels=8, **kwargs):
    """Get a DeBERTa classifier with default configuration."""
    return DebertaClassifier(num_labels=num_labels, **kwargs)


class HybridEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Hybrid ensemble combining transformer and traditional models."""
    
    def __init__(
        self,
        transformer_model=None,
        traditional_models=None,
        weights=None,
        device=None
    ):
        """
        Initialize the hybrid ensemble.
        
        Args:
            transformer_model: DeBERTa classifier
            traditional_models: List of (name, model) tuples for traditional models
            weights: List of weights for each model
            device: Device to use for transformer model
        """
        self.transformer_model = transformer_model or get_deberta_classifier(device=device)
        self.traditional_models = traditional_models or []
        self.weights = weights
        self.device = device
        self.feature_eng = FeatureEngineeringPreprocessor()
        self._is_fitted = False
        
        # Calculate weights if not provided
        if self.weights is None:
            # Assign higher weight to transformer model
            n_models = len(self.traditional_models) + 1
            self.weights = [2.0] + [1.0] * (n_models - 1)
            
            # Normalize weights
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
    
    def fit(self, X, y):
        """
        Train all models in the ensemble.
        
        Args:
            X: Input text
            y: Target labels
        
        Returns:
            self: Trained model
        """
        logger.info("Training hybrid ensemble")
        
        # Train transformer model
        logger.info("Training transformer model")
        self.transformer_model.fit(X, y)
        
        # Generate engineered features for traditional models
        logger.info("Generating engineered features")
        X_features = self.feature_eng.fit_transform(X)
        
        # Train traditional models
        for name, model in self.traditional_models:
            logger.info(f"Training {name} model")
            model.fit(X_features, y)
        
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions from the ensemble.
        
        Args:
            X: Input text
        
        Returns:
            y_pred: Predicted labels
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Get transformer predictions
        transformer_proba = self.transformer_model.predict_proba(X)
        
        # Initialize weighted probabilities with transformer predictions
        weighted_proba = self.weights[0] * transformer_proba
        
        # Generate engineered features for traditional models
        X_features = self.feature_eng.transform(X)
        
        # Add predictions from traditional models
        for i, (name, model) in enumerate(self.traditional_models):
            # Get probabilities from model
            model_proba = model.predict_proba(X_features)
            
            # Add to weighted probabilities
            weighted_proba += self.weights[i + 1] * model_proba
        
        # Get final predictions
        y_pred = np.argmax(weighted_proba, axis=1)
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Input text
        
        Returns:
            y_proba: Class probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Get transformer predictions
        transformer_proba = self.transformer_model.predict_proba(X)
        
        # Initialize weighted probabilities with transformer predictions
        weighted_proba = self.weights[0] * transformer_proba
        
        # Generate engineered features for traditional models
        X_features = self.feature_eng.transform(X)
        
        # Add predictions from traditional models
        for i, (name, model) in enumerate(self.traditional_models):
            # Get probabilities from model
            model_proba = model.predict_proba(X_features)
            
            # Add to weighted probabilities
            weighted_proba += self.weights[i + 1] * model_proba
        
        return weighted_proba
    
    def save(self, path):
        """
        Save all models in the ensemble.
        
        Args:
            path: Directory path to save model
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        # Create directory
        os.makedirs(path, exist_ok=True)
        
        # Save transformer model
        transformer_path = os.path.join(path, "transformer")
        self.transformer_model.save(transformer_path)
        
        # Save traditional models
        traditional_path = os.path.join(path, "traditional")
        os.makedirs(traditional_path, exist_ok=True)
        
        for i, (name, model) in enumerate(self.traditional_models):
            model_path = os.path.join(traditional_path, f"{name}.joblib")
            joblib.dump(model, model_path)
        
        # Save feature engineering preprocessor
        joblib.dump(self.feature_eng, os.path.join(path, "feature_eng.joblib"))
        
        # Save metadata
        metadata = {
            'model_names': [name for name, _ in self.traditional_models],
            'weights': self.weights
        }
        
        with open(os.path.join(path, "metadata.json"), 'w') as f:
            import json
            json.dump(metadata, f)
        
        logger.info(f"Ensemble saved to {path}")
    
    @classmethod
    def load(cls, path):
        """
        Load ensemble from disk.
        
        Args:
            path: Directory path to load model from
        
        Returns:
            model: Loaded model
        """
        # Load metadata
        with open(os.path.join(path, "metadata.json"), 'r') as f:
            import json
            metadata = json.load(f)
        
        # Load transformer model
        transformer_path = os.path.join(path, "transformer")
        transformer_model = DebertaClassifier.load(transformer_path)
        
        # Load traditional models
        traditional_path = os.path.join(path, "traditional")
        traditional_models = []
        
        for name in metadata['model_names']:
            model_path = os.path.join(traditional_path, f"{name}.joblib")
            model = joblib.load(model_path)
            traditional_models.append((name, model))
        
        # Create instance
        instance = cls(
            transformer_model=transformer_model,
            traditional_models=traditional_models,
            weights=metadata['weights']
        )
        
        # Load feature engineering preprocessor
        instance.feature_eng = joblib.load(os.path.join(path, "feature_eng.joblib"))
        
        # Mark as fitted
        instance._is_fitted = True
        
        logger.info(f"Ensemble loaded from {path}")
        return instance 