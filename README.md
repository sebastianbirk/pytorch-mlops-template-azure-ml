# PyTorch MLOps Template Azure ML
![pytorch mlops template azure_ml banner](https://github.com/sebastianbirk/pytorch-mlops-template-azure-ml/blob/master/docs/images/pytorch_mlops_template_azure_ml_banner.png)

## Description
This repository contains an end-to-end implementation of an image classification model in Azure, leveraging Azure's MLOps capabilities. It is shown how to develop, train, deploy and serve models in the Azure ecosystem using different Azure services such as Azure Machine Learning, Azure DevOps, Azure Kubernetes Service, Azure App Service and more. The repository can be used as a template repository to quickly bootstrap similar modeling workloads from development to production.

Specifically, the following aspects are covered in this template repository:
-	Creating a conda development environment and adding it as a Jupyter kernel as well as creating Azure Machine Learning environments for model development, training and deployment
-	Downloading public data and uploading it to an Azure Blob Storage that is connected to the Azure Machine Learning workspace
-	Training a Custom Vision model with Azure Cognitive Services to have a benchmark model
-	Training a PyTorch model with transfer learning on Azure Machine Learning
-	Evaluating the trained models leveraging Azure Machine Learning capabilities
-	Deploying the trained model to different compute targets in Azure, such as Azure Machine Learning Compute Instance, Azure Container Instances, Azure Kubernetes Service
-	Creating a flask frontend for model serving and deploying it to Azure App Service
-	Creating Azure Machine Learning pipelines for trigger-based model training, evaluation, and registration
-	Building CI/CD pipelines in Azure DevOps for unit and integration testing, automated model training, automated model deployment, building and pushing images to an Azure Container Registry and automated deployment of the Flask frontend
-	Running CI/CD pipelines within a docker container on Azure Pipelines Agents

## Repository Folder Structure
```
├── ado_pipelines                           <- ADO pipeline .yml files and other artifacts related to ADO pipelines
│   ├── templates                           <- Pipeline template modules that are used within other ADO pipelines
│   ├── variables                           <- Pipeline template files containing variables for ADO pipelines
│       ├── pipeline_variables.yml          <- Contains all pipeline variables for all pipelines
│   ├── ci_pipeline.yml                     <- Continuous integration pipeline with unit and integration tests
│   ├── docker_image_pipeline.yml           <- Pipeline to build and push Docker images to an Azure Container Registry
│   ├── flask_app_deployment_pipeline.yml   <- Continuous deployment pipeline for Flask web app
│   ├── model_deployment_pipeline.yml       <- Continuous deployment pipeline for model deployment to ACI/AKS
│   ├── model_training_pipeline.yml         <- Continuous integration pipeline for model training with an AML pipeline
├── data                                    <- Data (will be ignored by git except for example images for testing)
├── docs                                    <- All documentation
│   ├── how_to                              <- Markdown files that explain different aspects of the template
│   ├── images                              <- Images that are used in the Jupyter notebooks
├── environments                            <- Environment-related artifacts (for code execution)
│   ├── conda                               <- Conda .yml files for dev, train and deploy environments
│   ├── docker                              <- Dockerfiles
│       ├── inferencing_image               <- Dockerfile and inferencing artifacts to build the inferencing image
│       ├── mlops_pipeline_image            <- Dockerfile to build the mlops image for the Azure Pipelines Agent
├── infrastructure                          <- All artifacts related to infrastructure provisioning
│   ├── aks                                 <- AKS resource creation shell script; AKS deployment manifest
│   ├── custom_vision                       <- Custom vision resource creation shell scripts
├── notebooks                               <- Jupyter notebooks end-to-end ML system development walkthrough
│   ├── 00_environment_setup.ipynb          <- Conda env and jupyter kernel for dev; AML envs for dev, train, deploy 
│   ├── 01_dataset_setup.ipynb              <- Data download and upload to AML datastore; AML dataset registration
│   ├── 02_model_training.ipynb             <- Model training in AML; script and hyperdrive run; AML model registration
│   ├── 03_model_evaluation.ipynb           <- Evaluation of model on test set; addition of accuracy to AML model
│   ├── 04_model_deployment.ipynb           <- Model deployment to different compute targets using the AML Python SDK
│   ├── 05_model_training_pipeline.ipynb    <- Publishing of AML training pipeline with train, evaluate, register steps
│   ├── 10_custom_vision_model.ipynb        <- Model training in Custom Vision (Azure Cognitive Services) as benchmark
├── outputs                                 <- Output artifacts generated during ML lifecycle (e.g. model binaries)
├── src                                     <- Source code 
│   ├── config                              <- Configuration files (e.g. model training & evaluation parameter file)
│   ├── deployment                          <- Scripts needed for inference (e.g score.py script)
│   ├── flask_app                           <- All artifacts for the flask web frontend
│   ├── pipeline                            <- Scripts needed for automated model training with AML pipelines
│   ├── training                            <- Scripts needed for model training (e.g. train.py)
│   ├── utils                               <- Modularized utility functions that might be reused across the source code
├── testing                                 <- Testing-related artifacts
│   ├── integration                         <- Integration testing scripts
│   ├── unit                                <- Unit testing scripts (e.g. test PyTorch dataloader)
├── .amlignore                              <- Contains all artifacts not to be snapshotted by Azure Machine Learning
├── .env                                    <- Contains necessary environment variables which are used from source code
├── .gitignore                              <- Contains all artifacts that should not be checked into the git repo
├── README.md                               <- This README file (overview over the repository and documentation)
```
