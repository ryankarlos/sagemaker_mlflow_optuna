#!/bin/bash
# Deploy SageMaker infrastructure
# Usage: ./deploy.sh --user <name> --bucket <name> [--region <region>]

set -e

# Default values
REGION="us-east-1"
PROJECT_NAME="sm-mlflow-optuna"
STACK_NAME="${PROJECT_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_FILE="${SCRIPT_DIR}/cloudformation.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --user)
            USER_PROFILE_NAME="$2"
            shift 2
            ;;
        --bucket)
            BUCKET_NAME="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --vpc)
            VPC_ID="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./deploy.sh --user <name> --bucket <name> [--region <region>] [--vpc <vpc-id>]"
            echo ""
            echo "Required:"
            echo "  --user    SageMaker user profile name"
            echo "  --bucket  S3 bucket name"
            echo ""
            echo "Optional:"
            echo "  --region  AWS region (default: us-east-1)"
            echo "  --vpc     VPC ID (default: uses default VPC or first available)"
            echo ""
            echo "Example:"
            echo "  ./deploy.sh --user ryan --bucket my-ml-bucket --region us-east-1"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$USER_PROFILE_NAME" ]; then
    echo "Error: --user is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$BUCKET_NAME" ]; then
    echo "Error: --bucket is required"
    echo "Use --help for usage information"
    exit 1
fi

echo "Deploying stack: ${STACK_NAME}"
echo "User Profile: ${USER_PROFILE_NAME}"
echo "Bucket: ${BUCKET_NAME}"
echo "Region: ${REGION}"

# Fetch VPC if not provided
if [ -z "$VPC_ID" ]; then
    echo "Fetching VPC..."
    # Try default VPC first
    VPC_ID=$(aws ec2 describe-vpcs \
        --filters "Name=is-default,Values=true" \
        --query 'Vpcs[0].VpcId' \
        --output text \
        --region ${REGION})
    
    # If no default VPC, use first available
    if [ -z "$VPC_ID" ] || [ "$VPC_ID" == "None" ]; then
        echo "No default VPC found, using first available VPC..."
        VPC_ID=$(aws ec2 describe-vpcs \
            --query 'Vpcs[0].VpcId' \
            --output text \
            --region ${REGION})
    fi
fi

if [ -z "$VPC_ID" ] || [ "$VPC_ID" == "None" ]; then
    echo "Error: No VPC found in region ${REGION}"
    exit 1
fi
echo "VPC ID: ${VPC_ID}"

# Fetch subnets from default VPC (get first 2)
echo "Fetching subnets..."
SUBNET_IDS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=${VPC_ID}" \
    --query 'Subnets[0:2].SubnetId' \
    --output text \
    --region ${REGION} | tr '\t' ',')

if [ -z "$SUBNET_IDS" ] || [ "$SUBNET_IDS" == "None" ]; then
    echo "Error: No subnets found in VPC ${VPC_ID}"
    exit 1
fi
echo "Subnet IDs: ${SUBNET_IDS}"

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
        ProjectName=${PROJECT_NAME} \
        UserProfileName=${USER_PROFILE_NAME} \
        BucketName=${BUCKET_NAME} \
        VpcId=${VPC_ID} \
        SubnetIds=${SUBNET_IDS} \
    --capabilities CAPABILITY_NAMED_IAM \
    --region ${REGION} \
    --tags \
        Project=${PROJECT_NAME}

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
