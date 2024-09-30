# FIAP - ML Engineering - Project 03 - Movie Scoring System

## Overview
This is the source code of the 3rd Tech Challenge of FIAP ML Engineering degree. The project uses various tools and services like MLflow, MinIO, Kubernetes (K8s), and Trino to manage the entire workflow, from data preparation to deployment for a movie scoring system based on [The Movie DB](https://www.themoviedb.org/).

## Folder Structure

### 1. [`app`](./app)
This folder contains the core application logic, including model building, data preparation, and deployment processes.

- **[`container/`](./app/container)**:
  - `01-build-app.sh`: Script to build the application’s Docker image.
  - `app.dockerfile`: Dockerfile for building the app container.

- **[`frontend/`](./app/frontend)**: Contains the Streamlit application.

- **[`get_movies/`](./app/get_movies)**:
  - `file_analysis.ipynb`: Notebook for analyzing the movie data files.
  - `get_movies.py`: Python script to retrieve movie data.
  - `get_movies.sh`: Shell script to automate movie data retrieval.
  - `transfer_data.py`: Script for transferring data to S3-compatible storage.

- **[`k8s/`](./app/k8s)**: 
  - `01-create-secret.sh`: Script to create The Movie DB access key Kubernetes secrets.
  - `02-get-movies.sh`: Script to deploy the data loading process as a cronjob in K8s.
  - `03-data-prep.sh`: Script to run the data preparation as a cronjob in K8s.
  - `04-model-training.sh`: Script to run the model training as a cronjob in K8s.
  - `05-streamlit.sh`: Script to deploy the Streamlit app in K8s.
  - `data-prep.yaml`: Kubernetes manifest for the data preparation job.
  - `get-movies-cronjob.yaml`: Kubernetes CronJob manifest for fetching movies data.
  - `streamlit.yaml`: Kubernetes manifest to deploy the Streamlit app.
  - `train-model.yaml`: Kubernetes manifest to train the model.

- **[`prepare_data/`](./app/prepare_data)**:
  - `01-data-prep.sh`: Script to run the data preparation process.
  - `02-train-model.sh`: Script to initiate model training.
  - `03-deploy-model.sh`: Script to deploy the trained model.
  - `data_prep.py`: Python script for data preparation tasks.
  - `model_v2.py`, `model_v3.py`, `model_v4.py`: Different versions of the model architecture. `model_v4.py` is the current version.
  - `requirements.txt`: Python dependencies for the project.

- **[`build_model/`](./app/build_model)**: 
  - `05-build-mlflow-model.sh`: Shell script to build and save the MLflow model on a Docker container.
  - `90-teste-api.sh`: Script for testing API functionalities.
  - `deploy_model.yaml`: Kubernetes YAML file for deploying the model.
  - `train_test_data_v001.ipynb`: Jupyter notebook to train and test the model.

### 2. [`env`](./env)
This folder contains environment-related configurations for different services used in the project.

- **[`minio/`](./env/minio)**: Contains scripts and configurations for MinIO (S3-compatible storage).
- **[`mlflow/`](./env/mlflow)**: Contains configurations for managing MLflow, used to tracking experiments and deploying models.
- **[`mlflow-local/`](./env/mlflow-local)**: Local configurations for running MLflow.
- **[`trino/`](./env/trino)**: Configuration related to Trino, can be used for running distributed SQL queries across various data sources.

### 3. [`.gitignore`](./.gitignore)
Specifies files and folders to be ignored by Git, ensuring sensitive or unnecessary files (such as temporary files or credentials) are not committed.

## How to Run

### Prerequisites
- Docker
- Kubernetes (K8s)
- MinIO
- MLflow
- Trino
- Python 3.x and required dependencies (see `requirements.txt`)

### Steps

## Dependencies

### 1. **Kubernetes**
- **Minikube** (for local Kubernetes) or other Kubernetes cluster.
- **kubectl**: Command-line tool to interact with your Kubernetes clusters.

### 2. **Docker**
- **Docker**: Required to build images for the application.
  
### 3. **Python >=3.10**

1. **Build the Docker Container**:
    ```bash
    cd app/container
    ./01-build-app.sh
    ```

2. **Push to an accessible container registry**.

3. **Deploy MinIO**:
    ```bash
    cd env/minio
    ./00-minio-operator.sh
    ./00-minio-dpl.sh
    ```

3. **Deploy MLflow**:
    ```bash
    cd env/mlflow
    ./01-deploy-mlflow.sh
    ```

3. **Deploy Applications**:
    ```bash
    cd app/k8s
    ./01-create-secret.sh
    ./02-get-movies.sh
    ./03-data-prep.sh
    ./04-model-training.sh
    ./05-streamlit.sh
    ```
