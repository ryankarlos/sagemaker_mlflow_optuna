"""
AWS SageMaker Factorization Machines training module.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class FactorizationMachinesTrainer:
    """Trainer for AWS SageMaker Factorization Machines."""

    def __init__(
        self,
        role: str,
        instance_type: str = "ml.m5.large",
        instance_count: int = 1,
        output_path: str = None,
        base_job_name: str = "fm-gambling",
    ):
        self.role = role
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.output_path = output_path
        self.base_job_name = base_job_name
        self.estimator = None

    def _get_estimator(self, hyperparameters: Dict[str, Any]):
        """Create SageMaker FM estimator."""
        try:
            import sagemaker
            from sagemaker import image_uris
            from sagemaker.estimator import Estimator

            region = sagemaker.Session().boto_region_name
            container = image_uris.retrieve("factorization-machines", region)

            return Estimator(
                image_uri=container,
                role=self.role,
                instance_count=self.instance_count,
                instance_type=self.instance_type,
                output_path=self.output_path,
                base_job_name=self.base_job_name,
                hyperparameters=hyperparameters,
            )
        except ImportError:
            raise ImportError("sagemaker package required")

    def train(
        self,
        train_data_path: str,
        validation_data_path: Optional[str] = None,
        hyperparameters: Dict[str, Any] = None,
        wait: bool = True,
    ) -> str:
        """Train FM model on SageMaker."""
        try:
            from sagemaker.inputs import TrainingInput

            default_hyperparameters = {
                "feature_dim": "auto",
                "predictor_type": "regressor",
                "num_factors": 64,
                "epochs": 20,
                "mini_batch_size": 1000,
            }

            if hyperparameters:
                default_hyperparameters.update(hyperparameters)

            self.estimator = self._get_estimator(default_hyperparameters)

            train_input = TrainingInput(
                train_data_path,
                content_type="application/x-recordio-protobuf",
            )
            inputs = {"train": train_input}

            if validation_data_path:
                inputs["test"] = TrainingInput(
                    validation_data_path,
                    content_type="application/x-recordio-protobuf",
                )

            self.estimator.fit(inputs, wait=wait)
            return self.estimator.latest_training_job.name

        except ImportError:
            raise ImportError("sagemaker package required")

    def get_model_artifacts(self) -> str:
        """Get S3 path to trained model artifacts."""
        if self.estimator is None:
            raise ValueError("No trained model available")
        return self.estimator.model_data


def get_fm_hyperparameter_space(trial) -> Dict[str, Any]:
    """Define Optuna hyperparameter search space for FM."""
    return {
        "num_factors": trial.suggest_int("num_factors", 8, 128),
        "epochs": trial.suggest_int("epochs", 10, 50),
        "mini_batch_size": trial.suggest_categorical("mini_batch_size", [500, 1000, 2000]),
        "bias_lr": trial.suggest_float("bias_lr", 0.001, 0.1, log=True),
        "factors_lr": trial.suggest_float("factors_lr", 0.0001, 0.01, log=True),
        "bias_wd": trial.suggest_float("bias_wd", 0.0001, 0.01, log=True),
        "factors_wd": trial.suggest_float("factors_wd", 0.0001, 0.01, log=True),
    }


class LocalFMSimulator:
    """Local FM simulator for testing without SageMaker."""

    def __init__(self, num_factors: int = 64, epochs: int = 20, learning_rate: float = 0.01):
        self.num_factors = num_factors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = None
        self.factors = None
        self.global_bias = 0.0

    def fit(self, X: csr_matrix, y: np.ndarray) -> "LocalFMSimulator":
        """Fit local FM model using SGD."""
        n_samples, n_features = X.shape

        self.weights = np.random.randn(n_features) * 0.01
        self.factors = np.random.randn(n_features, self.num_factors) * 0.01
        self.global_bias = np.mean(y)

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            total_loss = 0.0

            for i in indices:
                xi = X.getrow(i).toarray().flatten()
                yi = y[i]
                pred = self._predict_single(xi)
                error = pred - yi
                total_loss += error**2
                self.weights -= self.learning_rate * error * xi

            if epoch % 5 == 0:
                rmse = np.sqrt(total_loss / n_samples)
                logger.info(f"Epoch {epoch}, RMSE: {rmse:.4f}")

        return self

    def _predict_single(self, x: np.ndarray) -> float:
        """Predict for single sample."""
        linear = np.dot(self.weights, x)
        vx = np.dot(self.factors.T, x)
        interactions = 0.5 * (np.sum(vx**2) - np.sum((self.factors.T**2) @ (x**2)))
        return self.global_bias + linear + interactions

    def predict(self, X: csr_matrix) -> np.ndarray:
        """Predict for multiple samples."""
        return np.array([self._predict_single(X.getrow(i).toarray().flatten()) for i in range(X.shape[0])])

    def score(self, X: csr_matrix, y: np.ndarray) -> float:
        """Calculate RMSE score."""
        return np.sqrt(np.mean((self.predict(X) - y) ** 2))
