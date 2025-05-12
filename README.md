# Math Problem Classification System

A comprehensive machine learning system for classifying mathematical problems into 8 different topics using MLflow for model lifecycle management.

## Project Overview

This project implements a complete machine learning lifecycle for classifying math problems into the following categories:

0. Algebra
1. Geometry and Trigonometry
2. Calculus and Analysis
3. Probability and Statistics 
4. Number Theory
5. Combinatorics and Discrete Math
6. Linear Algebra
7. Abstract Algebra and Topology

The system features:

- **Experiment Tracking**: Track parameters, metrics, and artifacts for multiple model experiments
- **Model Training & Tuning**: Train and optimize various models with hyperparameter tuning
- **Advanced NLP Models**: Support for transformer-based models (DeBERTa) and sentence embeddings
- **Ensemble Learning**: Implement voting and stacking ensemble methods
- **Model Registry**: Version control and lifecycle management for models
- **Model Serving**: REST API with FastAPI and MLflow integration
- **Performance Monitoring**: Track drift and performance metrics over time
- **Interactive UI**: Web interface for model management and testing

## System Architecture

```
┌───────────────┐    ┌───────────────┐    ┌──────────────────┐
│  Data Sources │───▶│  Preprocessing │───▶│  Model Training  │
└───────────────┘    └───────────────┘    └──────────────────┘
                                                   │
┌───────────────┐    ┌───────────────┐             ▼
│  Monitoring   │◀───│  Model Serving │◀────┐  ┌──────────────┐
└───────────────┘    └───────────────┘     │  │ MLflow Model │
                            ▲              │  │  Registry    │
                            │              │  └──────────────┘
                      ┌─────────────┐      │         │
                      │    Users    │◀─────┘         │
                      └─────────────┘                ▼
                                             ┌──────────────────┐
                                             │ Hyperparameter   │
                                             │ Optimization     │
                                             └──────────────────┘
```

## Project Structure

```
project/
├── data/                    # Dataset files and predictions
│   ├── train.csv            # Training dataset
│   ├── test.csv             # Testing dataset
│   ├── monitoring*.csv      # Data for model monitoring
│   └── submission_*.csv     # Model predictions
├── src/                     # Source code
│   ├── data_processing.py   # Data preprocessing functions
│   ├── models.py            # Model definitions
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   ├── hyperopt_tuning.py   # Hyperparameter tuning
│   ├── ensemble_hyperopt.py # Ensemble optimization
│   ├── train_embeddings.py  # Sentence embeddings training
│   ├── transformer_model.py # DeBERTa transformer models
│   ├── train_enhanced.py    # Enhanced model training
│   ├── monitor.py           # Model monitoring
│   └── utils.py             # Utility functions
├── models/                  # Saved model artifacts
├── mlruns/                  # MLflow tracking data
├── monitoring/              # Monitoring data and visualizations
├── run_server.py            # MLflow server startup script
├── serve_model.py           # Model serving with FastAPI
├── sample_input.csv         # Example input data
├── requirements.txt         # Project dependencies

```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd project
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Create required directories (if they don't exist):
   ```powershell
   mkdir -p data models mlruns monitoring
   ```

4. Download NLTK resources (will be done automatically during first run):
   ```powershell
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## Quick Start Guide

### Starting MLflow Server

MLflow provides experiment tracking, model registry, and model serving capabilities:

```powershell
python run_server.py --port 5000
```

Access the MLflow UI at http://localhost:5000

### Training a Basic Model

Train a logistic regression model with default parameters:

```powershell
python src/train.py --model-type lr --register-model
```

Available model types: `lr` (Logistic Regression), `rf` (Random Forest), `gb` (Gradient Boosting), `svm` (Support Vector Machine), `nb` (Naive Bayes)

### Hyperparameter Tuning

Optimize model parameters automatically:

```powershell
python src/hyperopt_tuning.py --model-type rf --max-evals 20 --register-model
```

### Training Advanced Models

#### Sentence Embeddings Model

```powershell
python src/train_embeddings.py --embedding-model "all-MiniLM-L6-v2" --register-model
```

#### DeBERTa Transformer Model

```powershell
python src/train_enhanced.py --model-approach deberta --register-model
```

#### Hybrid Ensemble Model

```powershell
python src/train_enhanced.py --model-approach hybrid --traditional-models lr,rf --register-model
```

### Creating Ensembles

#### Voting Ensemble

```powershell
python src/ensemble_hyperopt.py --base-model-types lr,rf,gb --ensemble-type voting --register-model
```

#### Stacking Ensemble

```powershell
python src/ensemble_hyperopt.py --base-model-types lr,rf,gb,svm --ensemble-type stacking --register-model
```

## Serving Models

### Using FastAPI Server

Start the model server with a web interface:

```powershell
python serve_model.py --port 5001
python serve_model.py --model math-topic-classifier --version 1
python serve_model.py --model math-topic-classifier --stage Staging
python serve_model.py --change-stage --model math-topic-classifier --version 1 --target-stage Production
```

This provides:
- Interactive web UI at http://localhost:5001
- REST API at http://localhost:5001/predict
- Model switching and configuration options

### Using MLflow's Native Server

Deploy a registered model using MLflow's server:

```powershell
$Env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$Env:MLFLOW_REGISTRY_URI = "http://localhost:5000"
mlflow models serve -m "models:/math-topic-classifier/Production" -p 5001 --no-conda
```

### Hybrid Mode

Use the FastAPI UI with MLflow's server for predictions:
1. Start MLflow server with a model
2. Start FastAPI server: `python serve_model.py --use-mlflow-server`
3. Access the UI and enable "Use External MLflow Server" in settings

## Model Monitoring

Track model performance and drift:

```powershell
python src/monitor.py --model-name math-topic-classifier --stage Production --data-path data/monitoring1.csv
```

Monitoring visualizations are saved to the `monitoring/` directory:
- Confusion matrices
- Performance trends
- Data drift analysis
- Class distribution changes

## API Usage

### REST API Endpoints

#### Prediction API

```
POST /predict
Content-Type: application/json

{
  "questions": ["Solve for x in the equation 2x + 5 = 13"]
}
```

Response:
```json
{
  "predictions": [
    {
      "question": "Solve for x in the equation 2x + 5 = 13",
      "topic_id": 0,
      "topic_name": "Algebra"
    }
  ]
}
```

#### Model Switching

```
POST /switch_model
Content-Type: application/json

{
  "model_name": "math-topic-classifier-embeddings"
}
```

#### MLflow Server Toggle

```
POST /toggle_mlflow_server
Content-Type: application/json

{
  "enabled": true,
  "server_url": "http://localhost:5000"
}
```

## Model Types and Performance

The system supports multiple model types with varying levels of performance:

| Model Type | Description | Avg. Accuracy | Training Speed | Inference Speed |
|------------|-------------|---------------|---------------|----------------|
| Logistic Regression | Fast baseline model | ~80% | Very Fast | Very Fast |
| Random Forest | Ensemble of decision trees | ~85% | Fast | Fast |
| Gradient Boosting | Boosted tree ensemble | ~87% | Medium | Fast |
| SVM | Support vector machine | ~83% | Medium | Fast |
| Sentence Embeddings | Semantic text embeddings | ~90% | Medium | Medium |
| DeBERTa | Transformer-based model | ~95% | Slow | Slow |
| Hybrid Ensemble | Combined traditional + DL | ~96% | Very Slow | Medium |

## Advanced Configuration

### Environment Variables

- `MLFLOW_TRACKING_URI`: MLflow server location (default: http://localhost:5000)
- `MLFLOW_REGISTRY_URI`: MLflow model registry location
- `LOGLEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `MODEL_CACHE_DIR`: Local directory for caching models

### Command Line Arguments

The main scripts accept a variety of command line arguments. Use `--help` with any script to see available options:

```powershell
python serve_model.py --help
```

## Troubleshooting

### Common Issues

1. **Model loading errors**:
   - Ensure MLflow server is running
   - Check model exists in registry with correct name/stage
   - Verify local model files are not corrupted

2. **API connection errors**:
   - Check server is running and port is accessible
   - Verify correct URLs in configuration
   - Check firewall settings

3. **Performance issues**:
   - Transformer models require significant resources
   - Consider using embedding models for better performance
   - Check for memory leaks in long-running services

### Logs and Diagnostics

Enable detailed logging for troubleshooting:

```powershell
$Env:LOGLEVEL = "DEBUG" 
python serve_model.py
```

## Contributing

Guidelines for contributing to this project:

1. Follow coding standards in existing files
2. Add tests for new functionality
3. Document code and update this README for significant changes
4. Use meaningful commit messages

## License

This project is licensed under the MIT License. 