

def copy_optuna_study_to_db(source_db_file, target_db_file):
    study_name = optuna.study.get_all_study_names(
        storage=f"sqlite:///{source_db_file}"
    )[0]
    optuna.copy_study(
        from_study_name=study_name,
        from_storage=f"sqlite:///{source_db_file}",
        to_storage=f"sqlite:///{target_db_file}",
    )


def copy_all_studies_to_new_file(target_db_file, source_db_dir):
    if Path(target_db_file).exists():
        print(f"Removing existing optuna target db {target_db_file}")
        Path(target_db_file).unlink()
    for source_file in Path(source_db_dir).iterdir():
        if source_file.suffix == ".db":
            copy_optuna_study_to_db(source_file, target_db_file)


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


def get_experiment_runs(experiment_name) -> List[Dict]:
    """Get all runs for the specified experiment"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return runs.to_dict("records")


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

    print(f"Aggregated studies saved to {local_folder}/{optuna_target_db}")


if __name__ == "__main__":
    # Aggregate studies
    execute_study_agg_pipeline(
        experiment_name=experiment_name,
        optuna_target_db="all_fm_studies.db",
        local_folder="results",
    )

    # Load aggregated study
    study = optuna.load_study(
        study_name="fm_gambling",
        storage="sqlite:///results/all_fm_studies.db"
    )

    print(f"Best RMSE: {-study.best_value:.4f}")
    print(f"Best params: {study.best_params}")