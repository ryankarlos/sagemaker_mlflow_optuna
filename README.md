# SageMaker MLflow Optuna

MLOps pipeline for hyperparameter optimization using SageMaker, MLflow, and Optuna.

## Features

- Hyperparameter optimization with Optuna
- Experiment tracking with MLflow
- AWS Factorization Machines for recommendations
- Simulated gambling dataset for testing
- Local FM simulator for development
- SageMaker AI Domain with IAM authentication
- Feature Store for user/game/interaction features

## Quick Start

```bash
# Install dependencies
make setup

# Generate sample gambling data
make generate-data

# Run FM Optuna pipeline (local mode)
make pipeline-fm

# View experiments
make mlflow-ui
```

## Infrastructure Deployment

Deploy the full ML infrastructure using CloudFormation:

### Prerequisites

- AWS CLI configured with appropriate credentials
- VPC with private subnets
- IAM permissions for CloudFormation, SageMaker, S3, IAM

### Deploy Stack

```bash
aws cloudformation deploy \
  --template-file infrastructure/sagemaker-domain.yaml \
  --stack-name fm-gambling-recommender-dev \
  --parameter-overrides \
    Environment=dev \
    ProjectName=fm-gambling-recommender \
    UserProfileName=ryan \
    VpcId=<your-vpc-id> \
    SubnetIds=<subnet-1>,<subnet-2>,<subnet-3>,<subnet-4> \
    EnableFeatureStore=true \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1
```

### Stack Resources

The CloudFormation stack creates:

| Resource | Description |
|----------|-------------|
| S3 Bucket | Single bucket with prefixes for data, artifacts, feature-store, mlflow |
| SageMaker Domain | IAM-authenticated domain for notebooks and training |
| User Profile | Default user profile (configurable name) |
| Feature Groups | User, Game, and Interaction feature groups |
| MLflow Server | Managed MLflow tracking server (Small) |
| IAM Role | Execution role with S3, Feature Store, Glue, Lineage permissions |
| Security Group | VPC security group for SageMaker resources |

### Stack Outputs

After deployment, retrieve outputs:

```bash
aws cloudformation describe-stacks \
  --stack-name fm-gambling-recommender-dev \
  --query 'Stacks[0].Outputs' \
  --output table
```

Key outputs:
- `DomainUrl` - SageMaker Studio URL
- `BucketName` - S3 bucket for all data
- `SageMakerExecutionRoleArn` - IAM role ARN
- `UserFeaturesGroupName`, `GameFeaturesGroupName`, `InteractionFeaturesGroupName` - Feature group names

### Access SageMaker Studio

1. Open the AWS Console → SageMaker → Domains
2. Select `fm-gambling-recommender-dev`
3. Click on user profile `ryan`
4. Launch Studio

### S3 Bucket Structure

```
s3://fm-gambling-recommender-dev-<account-id>/
├── data/           # Training data
├── artifacts/      # Model artifacts
├── feature-store/  # Feature Store offline data
│   ├── user-features/
│   ├── game-features/
│   └── interaction-features/
├── mlflow/         # MLflow artifacts
└── sharing/        # Notebook sharing
```

See [infrastructure/README.md](infrastructure/README.md) for more details.

## Feature Store Integration

Ingest features to SageMaker Feature Store:

```python
from utils.feature_store import FeatureStoreManager

fs_manager = FeatureStoreManager(project_name="fm-gambling-recommender")
fs_manager.ingest_all_features(users_df, games_df, interactions_df)
```

Or via CLI:

```bash
python data/simulate_gambling_data.py --ingest --project fm-gambling-recommender
```

## Project Structure

```
├── data/               # Data generation
│   └── simulate_gambling_data.py
├── infrastructure/     # CloudFormation templates
│   ├── sagemaker-domain.yaml
│   ├── deploy.sh
│   └── README.md
├── notebooks/          # Jupyter notebooks
│   ├── fm_train.ipynb
│   ├── fm_pipeline.ipynb
│   └── fm_optuna_vis.ipynb
├── pipelines/          # Pipeline orchestration
│   ├── fm_optuna_train.py
│   └── sagemaker_nb_pipeline.py
├── steps/              # Pipeline steps
│   ├── preprocess/
│   │   └── fm_encoding.py
│   ├── train/
│   │   └── factorization_machines.py
│   └── evaluate/
│       └── metrics.py
├── utils/              # Shared utilities
│   ├── feature_store.py
│   ├── mlflow_helpers.py
│   ├── optuna_helpers.py
│   └── lineage.py
├── tests/              # Tests
├── docs/               # Documentation
├── Makefile
├── pyproject.toml
└── mkdocs.yml
```

## Development

```bash
make test    # Run tests
make lint    # Lint code
make format  # Format code
```
