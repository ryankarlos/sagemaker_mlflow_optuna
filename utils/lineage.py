"""
Data Lineage utilities for SageMaker Unified Studio / DataZone.

This module provides helpers to capture and track data lineage for FM training
pipelines, linking datasets, transformations, and model artifacts.

Note: SageMaker Unified Studio (built on DataZone) captures lineage automatically
for supported data sources (Glue, Redshift). For custom lineage from SageMaker
jobs, you can use the SageMaker Lineage APIs or post lineage events to DataZone.

References:
- https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/datazone-data-lineage-linking-nodes.html
- https://docs.aws.amazon.com/sagemaker/latest/dg/lineage-tracking.html
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3

logger = logging.getLogger(__name__)


class LineageTracker:
    """
    Track data lineage for FM training pipelines.
    
    Integrates with both SageMaker Lineage and DataZone for comprehensive
    lineage tracking across the ML lifecycle.
    """
    
    def __init__(
        self,
        domain_id: Optional[str] = None,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """
        Initialize lineage tracker.
        
        Parameters
        ----------
        domain_id : str, optional
            DataZone domain ID for SageMaker Unified Studio
        project_id : str, optional
            DataZone project ID
        region : str, optional
            AWS region
        """
        self.region = region or boto3.Session().region_name
        self.domain_id = domain_id
        self.project_id = project_id
        
        self.sm_client = boto3.client("sagemaker", region_name=self.region)
        self.datazone_client = boto3.client("datazone", region_name=self.region)
    
    def create_artifact(
        self,
        artifact_name: str,
        artifact_type: str,
        source_uri: str,
        properties: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create a SageMaker lineage artifact.
        
        Parameters
        ----------
        artifact_name : str
            Name of the artifact
        artifact_type : str
            Type (e.g., 'DataSet', 'Model', 'Image')
        source_uri : str
            S3 URI or other source location
        properties : dict, optional
            Additional properties
            
        Returns
        -------
        str
            Artifact ARN
        """
        try:
            response = self.sm_client.create_artifact(
                ArtifactName=artifact_name,
                ArtifactType=artifact_type,
                Source={
                    "SourceUri": source_uri,
                    "SourceTypes": [{"SourceIdType": "S3ETag"}],
                },
                Properties=properties or {},
            )
            logger.info(f"Created artifact: {response['ArtifactArn']}")
            return response["ArtifactArn"]
        except Exception as e:
            logger.error(f"Failed to create artifact: {e}")
            raise
    
    def create_context(
        self,
        context_name: str,
        context_type: str,
        description: Optional[str] = None,
        properties: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create a SageMaker lineage context (e.g., for a training run).
        
        Parameters
        ----------
        context_name : str
            Name of the context
        context_type : str
            Type (e.g., 'Experiment', 'Trial', 'Pipeline')
        description : str, optional
            Description
        properties : dict, optional
            Additional properties
            
        Returns
        -------
        str
            Context ARN
        """
        try:
            response = self.sm_client.create_context(
                ContextName=context_name,
                ContextType=context_type,
                Description=description or "",
                Properties=properties or {},
            )
            logger.info(f"Created context: {response['ContextArn']}")
            return response["ContextArn"]
        except Exception as e:
            logger.error(f"Failed to create context: {e}")
            raise
    
    def create_action(
        self,
        action_name: str,
        action_type: str,
        source_uri: str,
        status: str = "Completed",
        properties: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create a SageMaker lineage action (e.g., training job, transform).
        
        Parameters
        ----------
        action_name : str
            Name of the action
        action_type : str
            Type (e.g., 'ModelTraining', 'DataTransformation')
        source_uri : str
            Source URI (e.g., training job ARN)
        status : str
            Status of the action
        properties : dict, optional
            Additional properties
            
        Returns
        -------
        str
            Action ARN
        """
        try:
            response = self.sm_client.create_action(
                ActionName=action_name,
                ActionType=action_type,
                Source={"SourceUri": source_uri},
                Status=status,
                Properties=properties or {},
            )
            logger.info(f"Created action: {response['ActionArn']}")
            return response["ActionArn"]
        except Exception as e:
            logger.error(f"Failed to create action: {e}")
            raise
    
    def add_association(
        self,
        source_arn: str,
        destination_arn: str,
        association_type: str = "ContributedTo",
    ) -> None:
        """
        Create an association between lineage entities.
        
        Parameters
        ----------
        source_arn : str
            Source entity ARN
        destination_arn : str
            Destination entity ARN
        association_type : str
            Type of association (ContributedTo, AssociatedWith, DerivedFrom, Produced)
        """
        try:
            self.sm_client.add_association(
                SourceArn=source_arn,
                DestinationArn=destination_arn,
                AssociationType=association_type,
            )
            logger.info(f"Created association: {source_arn} -> {destination_arn}")
        except Exception as e:
            logger.error(f"Failed to create association: {e}")
            raise
    
    def track_fm_training_lineage(
        self,
        experiment_name: str,
        run_name: str,
        input_data_uri: str,
        output_model_uri: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
    ) -> Dict[str, str]:
        """
        Track complete lineage for an FM training run.
        
        Creates artifacts for input data and output model, an action for
        the training job, and associations linking them together.
        
        Parameters
        ----------
        experiment_name : str
            MLflow experiment name
        run_name : str
            MLflow run name
        input_data_uri : str
            S3 URI of input training data
        output_model_uri : str
            S3 URI of output model artifacts
        hyperparameters : dict
            Training hyperparameters
        metrics : dict
            Training metrics
            
        Returns
        -------
        dict
            ARNs of created lineage entities
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create input data artifact
        input_artifact_arn = self.create_artifact(
            artifact_name=f"fm-input-{run_name}-{timestamp}",
            artifact_type="DataSet",
            source_uri=input_data_uri,
            properties={
                "experiment": experiment_name,
                "run": run_name,
                "data_type": "training_data",
            },
        )
        
        # Create output model artifact
        output_artifact_arn = self.create_artifact(
            artifact_name=f"fm-model-{run_name}-{timestamp}",
            artifact_type="Model",
            source_uri=output_model_uri,
            properties={
                "experiment": experiment_name,
                "run": run_name,
                "model_type": "factorization_machine",
                **{f"metric_{k}": str(v) for k, v in metrics.items()},
            },
        )
        
        # Create training action
        action_arn = self.create_action(
            action_name=f"fm-training-{run_name}-{timestamp}",
            action_type="ModelTraining",
            source_uri=f"mlflow://{experiment_name}/{run_name}",
            properties={
                "algorithm": "FactorizationMachines",
                **{f"hp_{k}": str(v) for k, v in hyperparameters.items()},
            },
        )
        
        # Create associations
        self.add_association(input_artifact_arn, action_arn, "ContributedTo")
        self.add_association(action_arn, output_artifact_arn, "Produced")
        
        return {
            "input_artifact_arn": input_artifact_arn,
            "output_artifact_arn": output_artifact_arn,
            "action_arn": action_arn,
        }
    
    def get_lineage_graph(
        self,
        start_arn: str,
        direction: str = "Both",
        max_depth: int = 10,
    ) -> Dict[str, Any]:
        """
        Query the lineage graph from a starting entity.
        
        Parameters
        ----------
        start_arn : str
            ARN of the starting entity
        direction : str
            Direction to traverse (Ascendants, Descendants, Both)
        max_depth : int
            Maximum depth to traverse
            
        Returns
        -------
        dict
            Lineage graph with vertices and edges
        """
        try:
            response = self.sm_client.query_lineage(
                StartArns=[start_arn],
                Direction=direction,
                MaxDepth=max_depth,
                IncludeEdges=True,
            )
            return {
                "vertices": response.get("Vertices", []),
                "edges": response.get("Edges", []),
            }
        except Exception as e:
            logger.error(f"Failed to query lineage: {e}")
            raise


def get_source_identifier(
    resource_type: str,
    account_id: str,
    region: str,
    **kwargs,
) -> str:
    """
    Generate a sourceIdentifier for linking lineage nodes with DataZone assets.
    
    The sourceIdentifier format must match what DataZone expects for automatic
    linking between lineage nodes and catalog assets.
    
    Parameters
    ----------
    resource_type : str
        Type of resource (s3, glue_table, redshift_table)
    account_id : str
        AWS account ID
    region : str
        AWS region
    **kwargs : dict
        Resource-specific parameters
        
    Returns
    -------
    str
        Formatted sourceIdentifier
        
    Examples
    --------
    >>> get_source_identifier("s3", "123456789012", "us-east-1", bucket="my-bucket", key="data/train.csv")
    'arn:aws:s3:::my-bucket/data/train.csv'
    
    >>> get_source_identifier("glue_table", "123456789012", "us-east-1", database="mydb", table="mytable")
    'arn:aws:glue:us-east-1:123456789012:table/mydb/mytable'
    """
    if resource_type == "s3":
        bucket = kwargs.get("bucket")
        key = kwargs.get("key", "")
        return f"arn:aws:s3:::{bucket}/{key}" if key else f"arn:aws:s3:::{bucket}"
    
    elif resource_type == "glue_table":
        database = kwargs.get("database")
        table = kwargs.get("table")
        return f"arn:aws:glue:{region}:{account_id}:table/{database}/{table}"
    
    elif resource_type == "redshift_table":
        cluster = kwargs.get("cluster")
        database = kwargs.get("database")
        schema = kwargs.get("schema", "public")
        table = kwargs.get("table")
        return f"arn:aws:redshift:{region}:{account_id}:dbname:{cluster}/{database}/{schema}/{table}"
    
    elif resource_type == "sagemaker_model":
        model_name = kwargs.get("model_name")
        return f"arn:aws:sagemaker:{region}:{account_id}:model/{model_name}"
    
    elif resource_type == "feature_group":
        feature_group_name = kwargs.get("feature_group_name")
        return f"arn:aws:sagemaker:{region}:{account_id}:feature-group/{feature_group_name}"
    
    else:
        raise ValueError(f"Unknown resource type: {resource_type}")
