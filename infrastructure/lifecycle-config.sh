#!/bin/bash
# SageMaker Studio Lifecycle Configuration
# Runs on JupyterLab startup to install project dependencies

set -e

echo "Starting lifecycle configuration..."

# Navigate to home directory
cd /home/sagemaker-user

# Clone the repository if not exists
if [ ! -d "sagemaker_mlflow_optuna" ]; then
    echo "Cloning repository..."
    git clone https://github.com/ryankarlos/sagemaker_mlflow_optuna.git || echo "Clone failed, assuming local files exist"
fi

# Navigate to project directory
cd sagemaker_mlflow_optuna 2>/dev/null || cd /home/sagemaker-user

# Install dependencies using make setup if Makefile exists
if [ -f "Makefile" ]; then
    echo "Running make setup..."
    make setup
else
    echo "Installing dependencies directly..."
    pip install --upgrade pip
    pip install optuna mlflow pandas numpy scikit-learn pyarrow boto3 sagemaker
fi

echo "Lifecycle configuration complete!"
