"""FM Gambling Recommender - SageMaker FM utilities with MLflow & Optuna support."""

from scripts.fm_sagemaker import (
    write_to_s3,
    train_fm_model,
    deploy_fm_endpoint,
    predict,
)
from scripts.simulate_gambling_data import (
    generate_demo_data,
    generate_multi_brand_demo,
    BRANDS,
)

__all__ = [
    "write_to_s3",
    "train_fm_model",
    "deploy_fm_endpoint",
    "predict",
    "generate_demo_data",
    "generate_multi_brand_demo",
    "BRANDS",
]
