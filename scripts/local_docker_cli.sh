#!/bin/bash
# Setup script for local SageMaker pipeline execution
# Installs Docker and project dependencies

set -eux

# Install Docker prerequisites
sudo apt-get -y install ca-certificates curl gnupg

# Add Docker GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get -y update

# Install Docker CLI
VERSION_STRING=5:20.10.24~3-0~ubuntu-jammy
sudo apt-get install docker-ce-cli=$VERSION_STRING docker-compose-plugin -y


