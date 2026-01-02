"""
Factorization Machines training pipeline with Optuna hyperparameter optimization.
"""

import argparse
import logging
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict

import boto3
import mlflow
import numpy as np
import optuna
import pandas as pd

from data.simulate_gambling_data import generate_gambling_dataset
from steps.preprocess.fm_encoding import FMEncoder, create_user_item_matrix
from steps.train.factorization_machines import LocalFMSimulator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))


def get_aws_account_id() -> str:
    """Get AWS account ID using STS."""
    sts = boto3.client("sts")
    return sts.get_caller_identity()["Account"]


def get_aws_region() -> str:
    """Get current AWS region."""
    session = boto3.session.Session()
    return session.region_name or "us-east-1"


def parse_args():
    parser = argparse.ArgumentParser(description="FM Optuna Training Pipeline")
    parser.add_argument("--n_users", type=int, default=5000)
    parser.add_argument("--n_games", type=int, default=100)
    parser.add_argument("--n_days", type=int, default=180)
    parser.add_argument("--train_days", type=int, default=150)
    parser.add_argument("--max_trials", type=int, default=20)
    parser.add_argument("--early_stopping", type=int, default=5)
    parser.add_argument("--experiment_name", type=str, default="fm-gambling-optuna")
    parser.add_argument("--local", action="store_true", help="Use local FM simulator")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ingest_features", action="store_true", help="Ingest to Feature Store")
    parser.add_argument("--project_name", type=str, default="fm-gambling-recommender")
    return parser.parse_args()


def split_by_date(
    df: pd.DataFrame,
    train_days: int,
    date_col: str = "summary_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by date into train and validation sets."""
    df[date_col] = pd.to_datetime(df[date_col])
    max_date = df[date_col].max()
    min_date = df[date_col].min()
    cutoff_date = min_date + pd.Timedelta(days=train_days)

    train_df = df[df[date_col] <= cutoff_date].copy()
    valid_df = df[df[date_col] > cutoff_date].copy()

    return train_df, valid_df


def prepare_data(
    n_users: int,
    n_games: int,
    n_days: int,
    train_days: int,
    seed: int,
    ingest_features: bool = False,
    project_name: str = "fm-gambling-recommender",
) -> Dict[str, Any]:
    """Generate and prepare data for training."""
    logger.info("Generating gambling dataset...")
    users, games, interactions, _ = generate_gambling_dataset(
        n_users=n_users,
        n_games=n_games,
        n_days=n_days,
        avg_sessions_per_user=30,
        seed=seed,
    )

    logger.info(f"Generated {len(users)} users, {len(games)} games, {len(interactions)} interactions")

    # Ingest to Feature Store if requested
    if ingest_features:
        logger.info("Ingesting features to Feature Store...")
        from utils.feature_store import FeatureStoreManager
        
        fs_manager = FeatureStoreManager(project_name=project_name)
        logger.info(f"AWS Account: {fs_manager.account_id}, Region: {fs_manager.region}")
        
        # Check feature group status
        fg_status = fs_manager.describe_feature_groups()
        for name, status in fg_status.items():
            logger.info(f"  {name}: {status['status']}")
        
        # Only ingest if feature groups exist
        all_exist = all(s.get("status") == "Created" for s in fg_status.values())
        if all_exist:
            fs_manager.ingest_all_features(users, games, interactions, wait=True)
            logger.info("Feature Store ingestion complete!")
        else:
            logger.warning("Some feature groups not found, skipping ingestion")

    train_df, valid_df = split_by_date(interactions, train_days)
    logger.info(f"Train: {len(train_df)}, Validation: {len(valid_df)}")

    encoder = FMEncoder()
    categorical_cols = ["game_type", "vip_tier", "region"]

    train_features = train_df.merge(users, on="user_id").merge(games, on="game_id")
    valid_features = valid_df.merge(users, on="user_id").merge(games, on="game_id")

    X_train, y_train = encoder.fit_transform(
        train_features, target_col="bet_qty", categorical_cols=categorical_cols
    )
    X_valid, y_valid = encoder.transform(
        valid_features, target_col="bet_qty", categorical_cols=categorical_cols
    )

    train_matrix, user_enc, item_enc = create_user_item_matrix(train_df)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "train_df": train_df,
        "valid_df": valid_df,
        "encoder": encoder,
        "train_matrix": train_matrix,
        "user_encoder": user_enc,
        "item_encoder": item_enc,
        "users": users,
        "games": games,
    }


def objective(trial: optuna.Trial, data: Dict[str, Any], use_local: bool = True) -> float:
    """Optuna objective function for FM hyperparameter optimization."""
    with mlflow.start_run(run_name=f"Trial-{trial.number}", nested=True):
        num_factors = trial.suggest_int("num_factors", 8, 64)
        epochs = trial.suggest_int("epochs", 10, 30)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)

        hyperparams = {
            "num_factors": num_factors,
            "epochs": epochs,
            "learning_rate": learning_rate,
        }
        mlflow.log_params(hyperparams)

        if use_local:
            model = LocalFMSimulator(
                num_factors=num_factors,
                epochs=epochs,
                learning_rate=learning_rate,
            )
            model.fit(data["X_train"], data["y_train"])

            train_rmse = model.score(data["X_train"], data["y_train"])
            valid_rmse = model.score(data["X_valid"], data["y_valid"])

            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("valid_rmse", valid_rmse)

            logger.info(f"Trial {trial.number}: Train RMSE={train_rmse:.4f}, Valid RMSE={valid_rmse:.4f}")

            return -valid_rmse
        else:
            raise NotImplementedError("SageMaker training not implemented")


def early_stopping_callback(study: optuna.Study, trial: optuna.FrozenTrial, rounds: int = 5):
    """Early stopping callback for Optuna."""
    if len(study.trials) < rounds:
        return

    recent_values = [t.value for t in study.trials[-rounds:] if t.value is not None]
    if len(recent_values) < rounds:
        return

    best_recent = max(recent_values)
    if study.best_value is not None and best_recent <= study.best_value - 0.001:
        logger.info(f"Early stopping: No improvement in {rounds} rounds")
        study.stop()


def main():
    args = parse_args()
    
    # Get AWS info
    account_id = get_aws_account_id()
    region = get_aws_region()
    logger.info(f"AWS Account: {account_id}, Region: {region}")

    data = prepare_data(
        n_users=args.n_users,
        n_games=args.n_games,
        n_days=args.n_days,
        train_days=args.train_days,
        seed=args.seed,
        ingest_features=args.ingest_features,
        project_name=args.project_name,
    )

    mlflow.set_experiment(args.experiment_name)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"FM-Optuna-{current_time}"):
        mlflow.log_params({
            "n_users": args.n_users,
            "n_games": args.n_games,
            "n_days": args.n_days,
            "train_days": args.train_days,
            "max_trials": args.max_trials,
            "use_local": args.local,
            "aws_account_id": account_id,
            "aws_region": region,
            "ingest_features": args.ingest_features,
        })

        study_name = f"fm_gambling_{current_time}"
        storage_path = f"results/optuna_{study_name}.db"
        Path("results").mkdir(exist_ok=True)

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=f"sqlite:///{storage_path}",
            load_if_exists=True,
        )

        fn = partial(objective, data=data, use_local=args.local)
        es_callback = partial(early_stopping_callback, rounds=args.early_stopping)

        study.optimize(fn, n_trials=args.max_trials, callbacks=[es_callback], gc_after_trial=True)

        best_params = study.best_params
        best_value = study.best_value

        logger.info(f"Best params: {best_params}")
        logger.info(f"Best RMSE: {-best_value:.4f}")

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_valid_rmse", -best_value)

        trials_df = study.trials_dataframe()
        trials_path = f"results/{study_name}_trials.parquet"
        trials_df.to_parquet(trials_path)
        mlflow.log_artifact(trials_path, artifact_path="trials")
        mlflow.log_artifact(storage_path, artifact_path="optuna_db")

    mlflow.end_run()
    return study


if __name__ == "__main__":
    main()
