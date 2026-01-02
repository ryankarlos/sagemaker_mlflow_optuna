#!/bin/bash
# SageMaker Studio Lifecycle Configuration
# Runs on JupyterLab startup to install project dependencies

set -e

echo "Starting lifecycle configuration..."

# Navigate to home directory
cd /home/sagemaker-user



# Navigate to project directory
cd sagemaker_mlflow_optuna 2>/dev/null || cd /home/sagemaker-user

# Install dependencies using make setup if Makefile exists

echo "Installing dependencies directly..."
pip install --upgrade pip
pip install optuna mlflow

echo "Lifecycle configuration complete!"
