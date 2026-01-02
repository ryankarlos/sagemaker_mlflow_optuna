#!/bin/bash
# Deploy SageMaker Unified Studio infrastructure
# Usage: ./deploy.sh [environment] [region]

set -e

ENVIRONMENT=${1:-dev}
REGION=${2:-us-east-1}
STACK_NAME="fm-gambling-recommender-${ENVIRONMENT}"
TEMPLATE_FILE="sagemaker-domain.yaml"

echo "Deploying stack: ${STACK_NAME}"
echo "Environment: ${ENVIRONMENT}"
echo "Region: ${REGION}"

# Validate template
echo "Validating CloudFormation template..."
aws cloudformation validate-template \
    --template-body file://${TEMPLATE_FILE} \
    --region ${REGION}

# Deploy stack
echo "Deploying CloudFormation stack..."
aws cloudformation deploy \
    --template-file ${TEMPLATE_FILE} \
    --stack-name ${STACK_NAME} \
    --parameter-overrides \
        Environment=${ENVIRONMENT} \
        ProjectName=fm-gambling-recommender \
        DomainName=fm-ml-domain-${ENVIRONMENT} \
        MLflowServerSize=Small \
        EnableFeatureStore=true \
    --capabilities CAPABILITY_NAMED_IAM \
    --region ${REGION} \
    --tags \
        Project=fm-gambling-recommender \
        Environment=${ENVIRONMENT}

# Get outputs
echo ""
echo "Stack outputs:"
aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME} \
    --region ${REGION} \
    --query 'Stacks[0].Outputs' \
    --output table

echo ""
echo "Deployment complete!"
