"""
SageMaker Notebook Job Pipeline for FM Optuna training.

This module provides utilities to run FM training jobs as SageMaker notebook jobs,
enabling parallel execution across multiple configurations.
"""

import sagemaker
from sagemaker.workflow.notebook_job_step import NotebookJobStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession


def define_steps_for_pipeline(
    config_dict: dict,
    image_uri: str,
    notebook_artifacts: str,
    input_notebook_name: str,
    kernel_name: str = "python3",
    instance_type: str = "ml.m5.xlarge",
    role: str = None,
    **params,
) -> list:
    """
    Define notebook job steps for the pipeline.

    Parameters
    ----------
    config_dict : dict
        Dictionary mapping config names to parameter lists
        e.g., {"config_1": {"n_users": 5000, "n_games": 100}, ...}
    image_uri : str
        SageMaker container image URI
    notebook_artifacts : str
        S3 path for notebook artifacts
    input_notebook_name : str
        Name of the notebook to execute
    kernel_name : str
        Jupyter kernel name
    instance_type : str
        SageMaker instance type
    role : str
        SageMaker execution role ARN
    **params : dict
        Additional parameters passed to all notebook jobs

    Returns
    -------
    list
        List of NotebookJobStep objects
    """
    pipeline_steps = []
    if role is None:
        role = sagemaker.get_execution_role()

    for config_name, config_params in config_dict.items():
        # Merge config params with global params
        nb_job_params = {
            "config_name": config_name,
            **config_params,
            **params,
        }

        train_description = f"FM Optuna training for {config_name}"
        train_id = f"fm-train-{config_name}"

        nb_step = NotebookJobStep(
            name=train_id,
            description=train_description,
            notebook_job_name=train_id,
            image_uri=image_uri,
            kernel_name=kernel_name,
            display_name=train_id,
            role=role,
            s3_root_uri=notebook_artifacts,
            input_notebook=input_notebook_name,
            instance_type=instance_type,
            parameters=nb_job_params,
            max_runtime_in_seconds=86400,  # 24 hours
            max_retry_attempts=3,
        )
        pipeline_steps.append(nb_step)

    return pipeline_steps


def execute_local_sagemaker_pipeline(pipeline_name: str, steps: list, role: str = None):
    """
    Execute pipeline in local mode for testing.

    Parameters
    ----------
    pipeline_name : str
        Name of the pipeline
    steps : list
        List of pipeline steps
    role : str
        SageMaker execution role ARN

    Returns
    -------
    _PipelineExecution
        Pipeline execution object
    """
    session = LocalPipelineSession()
    pipeline = Pipeline(name=pipeline_name, steps=steps, sagemaker_session=session)

    if role is None:
        role = sagemaker.get_execution_role()

    pipeline.create(role)
    execution = pipeline.start()
    return execution


def execute_sagemaker_pipeline(pipeline_name: str, steps: list, role: str = None):
    """
    Execute pipeline on SageMaker.

    Parameters
    ----------
    pipeline_name : str
        Name of the pipeline
    steps : list
        List of pipeline steps
    role : str
        SageMaker execution role ARN

    Returns
    -------
    _PipelineExecution
        Pipeline execution object
    """
    session = PipelineSession()
    pipeline = Pipeline(name=pipeline_name, steps=steps, sagemaker_session=session)

    if role is None:
        role = sagemaker.get_execution_role()

    pipeline.upsert(role)
    execution = pipeline.start()
    print(execution.describe())
    return execution
