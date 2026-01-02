# Pipeline Overview

## Architecture

The pipeline consists of the following components:

### Data Generation

- `data/simulate_gambling_data.py` - Generate synthetic gambling data

### Steps

- **preprocess/fm_encoding.py** - Feature encoding for Factorization Machines
- **train/factorization_machines.py** - FM training (SageMaker + local simulator)
- **evaluate/metrics.py** - NDCG and other evaluation metrics

### Pipelines

- `pipelines/fm_optuna_train.py` - Main FM pipeline with Optuna optimization

### Utilities

- `utils/mlflow_helpers.py` - MLflow integration utilities
- `utils/optuna_helpers.py` - Optuna optimization utilities
- `utils/config.py` - Configuration management
