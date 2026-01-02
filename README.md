# SageMaker MLflow Optuna

MLOps pipeline for hyperparameter optimization using SageMaker, MLflow, and Optuna.

## Features

- Hyperparameter optimization with Optuna
- Experiment tracking with MLflow
- AWS Factorization Machines for recommendations
- Simulated gambling dataset for testing
- Local FM simulator for development

## Quick Start

```bash
# Install dependencies
make setup

# Generate sample gambling data
make generate-data

# Run FM Optuna pipeline (local mode)
make pipeline-fm

# View experiments
make mlflow-ui
```

## Project Structure

```
├── data/               # Data generation
│   └── simulate_gambling_data.py
├── pipelines/          # Pipeline orchestration
│   └── fm_optuna_train.py
├── steps/              # Pipeline steps
│   ├── preprocess/
│   │   └── fm_encoding.py
│   ├── train/
│   │   └── factorization_machines.py
│   └── evaluate/
│       └── metrics.py
├── utils/              # Shared utilities
├── tests/              # Tests
├── docs/               # Documentation
├── Makefile
├── pyproject.toml
└── mkdocs.yml
```

## Development

```bash
make test    # Run tests
make lint    # Lint code
make format  # Format code
```
