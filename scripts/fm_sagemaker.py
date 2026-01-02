"""
Simplified SageMaker Factorization Machines utilities.
Following the MNIST FM example pattern.
"""

import io
import boto3
import sagemaker
import numpy as np
import scipy.sparse as sparse
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput


def write_to_s3(X: sparse.csr_matrix, y: np.ndarray, bucket: str, prefix: str, key: str) -> str:
    """Write sparse data to S3 in RecordIO-protobuf format."""
    import sagemaker.amazon.common as smac
    
    buf = io.BytesIO()
    smac.write_spmatrix_to_sparse_tensor(buf, X, y.astype(np.float32))
    buf.seek(0)
    
    s3 = boto3.client("s3")
    s3_key = f"{prefix}/{key}"
    s3.put_object(Bucket=bucket, Key=s3_key, Body=buf.getvalue())
    
    return f"s3://{bucket}/{s3_key}"


def train_fm_model(
    train_path: str,
    test_path: str,
    output_path: str,
    role: str,
    n_features: int,
    num_factors: int = 64,
    epochs: int = 20,
    mini_batch_size: int = 200,
    predictor_type: str = "regressor",
    instance_type: str = "ml.c5.xlarge",
) -> Estimator:
    """
    Train FM model on SageMaker.
    
    """
    session = sagemaker.Session()
    region = session.boto_region_name
    
    # Get FM container
    container = image_uris.retrieve("factorization-machines", region)
    
    # Create estimator
    fm = Estimator(
        image_uri=container,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        output_path=output_path,
        sagemaker_session=session,
    )
    
    # Set hyperparameters
    fm.set_hyperparameters(
        feature_dim=n_features,
        predictor_type=predictor_type,
        num_factors=num_factors,
        epochs=epochs,
        mini_batch_size=mini_batch_size,
    )
    
    # Train
    fm.fit({
        "train": TrainingInput(train_path, content_type="application/x-recordio-protobuf"),
        "test": TrainingInput(test_path, content_type="application/x-recordio-protobuf"),
    })
    
    return fm


def deploy_fm_endpoint(estimator: Estimator, instance_type: str = "ml.m5.large"):
    """Deploy trained FM model to endpoint."""
    return estimator.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
    )


def predict(predictor, X: sparse.csr_matrix) -> np.ndarray:
    """Get predictions from deployed endpoint."""
    import sagemaker.amazon.common as smac
    
    buf = io.BytesIO()
    smac.write_spmatrix_to_sparse_tensor(buf, X, None)
    buf.seek(0)
    
    response = predictor.predict(buf.getvalue())
    return np.array([r.label["score"].float32_tensor.values[0] for r in response.predictions])
