# Infrastructure

CloudFormation template for deploying the FM Gambling Recommender ML infrastructure.

## Overview

The `sagemaker-domain.yaml` template deploys a complete ML infrastructure with:

- **SageMaker AI Domain** - IAM-authenticated ML IDE with notebooks
- **MLflow Tracking Server** - Experiment tracking and model registry
- **S3 Bucket** - Single bucket with prefixes for data, artifacts, feature store
- **Feature Store** - User, game, and interaction feature groups
- **IAM Roles** - Execution roles with S3, Feature Store, Glue, and Lineage permissions

## Quick Deploy

```bash
aws cloudformation deploy \
  --template-file infrastructure/sagemaker-domain.yaml \
  --stack-name fm-gambling-recommender-dev \
  --parameter-overrides \
    Environment=dev \
    ProjectName=fm-gambling-recommender \
    UserProfileName=ryan \
    VpcId=vpc-xxxxxxxxx \
    SubnetIds=subnet-xxx,subnet-yyy,subnet-zzz \
    EnableFeatureStore=true \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| ProjectName | fm-gambling-recommender | Resource name prefix |
| Environment | dev | Deployment environment (dev/staging/prod) |
| UserProfileName | ryan | SageMaker user profile name |
| VpcId | (required) | VPC ID for SageMaker Domain |
| SubnetIds | (required) | Comma-separated subnet IDs |
| EnableFeatureStore | true | Create Feature Store resources |

## Resources Created

| Resource | Name Pattern | Description |
|----------|--------------|-------------|
| S3 Bucket | `{project}-{env}-{account}` | All project data |
| SageMaker Domain | `{project}-{env}` | IAM-authenticated domain |
| User Profile | `{UserProfileName}` | Default user profile |
| IAM Role | `{project}-{env}-role` | Execution role |
| Security Group | `{project}-{env}-sg` | VPC security group |
| Feature Groups | `{project}-user-features`, etc. | User/Game/Interaction features |
| MLflow Server | `{project}-mlflow-{env}` | Managed tracking server |

## S3 Bucket Structure

```
s3://{project}-{env}-{account}/
├── data/              # Training and test data
├── artifacts/         # Model artifacts and outputs
├── feature-store/     # Feature Store offline storage
│   ├── user-features/
│   ├── game-features/
│   └── interaction-features/
├── mlflow/            # MLflow experiment artifacts
└── sharing/           # Notebook sharing outputs
```

## Stack Outputs

| Output | Description |
|--------|-------------|
| DomainId | SageMaker Domain ID |
| DomainUrl | SageMaker Studio URL |
| BucketName | S3 bucket name |
| SageMakerExecutionRoleArn | IAM role ARN |
| UserFeaturesGroupName | User feature group name |
| GameFeaturesGroupName | Game feature group name |
| InteractionFeaturesGroupName | Interaction feature group name |
| MLflowTrackingServerArn | MLflow server ARN |

## Access SageMaker Studio

1. Open AWS Console → SageMaker → Domains
2. Select `fm-gambling-recommender-dev`
3. Click on user profile (e.g., `ryan`)
4. Click "Open Studio"

## IAM Permissions

The execution role includes:

- `AmazonSageMakerFullAccess` - Full SageMaker access
- `AmazonSageMakerFeatureStoreAccess` - Feature Store operations
- S3 access to project bucket (GetObject, PutObject, DeleteObject, ListBucket, GetBucketAcl)
- Glue access for Feature Store catalog
- Lineage access for tracking artifacts and associations

## Cleanup

```bash
# Delete stack (S3 bucket is retained)
aws cloudformation delete-stack \
  --stack-name fm-gambling-recommender-dev \
  --region us-east-1

# To also delete the S3 bucket:
aws s3 rb s3://fm-gambling-recommender-dev-{account} --force
```

Note: S3 bucket has `DeletionPolicy: Retain` to prevent accidental data loss.

## Troubleshooting

### Stack Creation Fails

1. Check CloudFormation events:
   ```bash
   aws cloudformation describe-stack-events \
     --stack-name fm-gambling-recommender-dev \
     --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`]'
   ```

2. Common issues:
   - **S3 bucket exists**: Delete existing bucket or use different project name
   - **IAM role exists**: Delete existing role or use different project name
   - **VPC/Subnet issues**: Ensure subnets have internet access (NAT Gateway)

### MLflow Server Takes Long

MLflow server creation takes 15-20 minutes. Monitor progress:
```bash
aws sagemaker list-mlflow-tracking-servers \
  --query 'TrackingServerSummaries[*].{Name:TrackingServerName,Status:TrackingServerStatus}'
```
