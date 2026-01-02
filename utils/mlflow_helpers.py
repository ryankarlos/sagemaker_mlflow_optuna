import mlflow
from mlflow.tracking import MlflowClient
import logging
import boto3
from pathlib import Path
import os
import dask
from typing import List, Dict

logger = logging.getLogger(__name__)


def delete_mlflow_runs_in_experiment(experiment_name, run_name=None):
    # Initialize the MLflow client
    client = MlflowClient()
    # Specify the experiment name or ID
    # Combining partial name match with other criteria
    experiment = client.get_experiment_by_name(experiment_name)
    # Ensure the experiment exists
    if experiment:
        experiment_id = experiment.experiment_id
        # List all runs in the experiment
        if run_name is not None:
            filter_string = f"tags.mlflow.runName LIKE '%{run_name}%'"
            runs = client.search_runs(
                experiment_ids=[experiment_id], filter_string=filter_string
            )
        else:
            runs = client.search_runs(experiment_ids=[experiment_id])
        # Delete each run
        for run in runs:
            logger.info(f"deleting run {run.info.run_name}")
            client.delete_run(run.info.run_id)
    else:
        logger.info(f"Experiment '{experiment_name}' does not exist.")


def get_experiment_runs(experiment_name) -> List[Dict]:
    """Get all runs for the specified experiment"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return runs.to_dict("records")


def setup_mlflow_server_access(mlflow_server_name):
    client = boto3.client("sagemaker")
    mlflow_server_arn = client.describe_mlflow_tracking_server(
        TrackingServerName=mlflow_server_name
    )["TrackingServerArn"]
    print(mlflow_server_arn)
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_server_arn


def download_single_artifact(run_info: Dict, folder, base_path: str) -> str:
    """Download an mlflow artifact for a single run based on root folder"""
    try:
        run_id = run_info["run_id"]
        run_path = Path(base_path)
        run_path.mkdir(exist_ok=True)
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        for artifact in artifacts:
            if folder in artifact.path:
                client.download_artifacts(run_id, artifact.path, str(run_path))
    except Exception as e:
        return f"Error downloading artifacts for run {run_id}: {str(e)}"


def download_artifacts_across_runs(experiment_name: str, folder: str, local_path: str):
    runs = get_experiment_runs(experiment_name)
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    # Create delayed objects for each download task
    delayed_tasks = [
        dask.delayed(download_single_artifact)(run, folder, str(local_path))
        for run in runs
    ]
    # Compute all tasks
    dask.compute(*delayed_tasks)
