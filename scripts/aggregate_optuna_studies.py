"""
Aggregate Optuna studies from MLflow artifacts.

This script downloads Optuna database files from MLflow runs and
combines them into a single database for analysis.
"""

import logging
from pathlib import Path

from utils.mlflow_helpers import download_artifacts_across_runs
from utils.optuna_helpers import copy_all_studies_to_new_file

logger = logging.getLogger(__name__)


def execute_study_agg_pipeline(
    experiment_name: str,
    optuna_target_db: str = "all_studies.db",
    local_folder: str = "results",
    mlflow_artifact_folder: str = "optuna_db",
) -> None:
    """
    Aggregate Optuna studies from MLflow experiment runs.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name
    optuna_target_db : str
        Target database filename for aggregated studies
    local_folder : str
        Local folder to download artifacts to
    mlflow_artifact_folder : str
        MLflow artifact folder name containing Optuna DBs
    """
    Path(local_folder).mkdir(exist_ok=True)

    # Download Optuna DB artifacts from all runs
    download_artifacts_across_runs(
        experiment_name, mlflow_artifact_folder, local_folder
    )

    # Combine all studies into single database
    copy_all_studies_to_new_file(
        f"{local_folder}/{optuna_target_db}",
        f"{local_folder}/{mlflow_artifact_folder}",
    )

    logger.info(f"Aggregated studies saved to {local_folder}/{optuna_target_db}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate Optuna studies")
    parser.add_argument("--experiment", type=str, required=True, help="MLflow experiment name")
    parser.add_argument("--output", type=str, default="all_studies.db", help="Output DB filename")
    parser.add_argument("--folder", type=str, default="results", help="Local folder")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    execute_study_agg_pipeline(args.experiment, args.output, args.folder)
