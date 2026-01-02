"""
Scripts module for FM-based gambling recommendation pipeline.

Contains encoding, training, and evaluation components.
"""

from scripts.fm_encoding import FMEncoder, SafeLabelEncoder, create_user_item_matrix
from scripts.factorization_machines import (
    FactorizationMachinesTrainer,
    LocalFMSimulator,
    get_fm_hyperparameter_space,
)
from scripts.metrics import calculate_rmse, calculate_mae, get_ndcg_scores

__all__ = [
    "FMEncoder",
    "SafeLabelEncoder",
    "create_user_item_matrix",
    "FactorizationMachinesTrainer",
    "LocalFMSimulator",
    "get_fm_hyperparameter_space",
    "calculate_rmse",
    "calculate_mae",
    "get_ndcg_scores",
]
