# PyTorch MLOps Template Azure ML
![pytorch mlops template azure_ml banner](https://github.com/sebastianbirk/pytorch-mlops-template-azure-ml/blob/master/docs/images/pytorch_mlops_template_azure_ml_banner.png)

## Description
This repository contains an end-to-end implementation of an image classification model built with PyTorch on Azure ML, leveraging Azure ML's MLOps capabilities. It can be used as a template repository to quickly bootstrap similar modeling workloads from development to production.

## Repository Folder Structure
```
├── ado_pipelines                           <- ADO pipeline .yml files and other artifacts related to ADO pipelines
│   ├── variables                           <- ADO .yml files containing variables for ADO pipelines
├── data                                    <- Data that has to be stored locally (will be ignored by git)
├── docs                                    <- All documentation
│   ├── getting_started                     <- Documentation that explains how to use this project template
│   ├── how_to                              <- Documentation that explains specific MLOps procedures (e.g. branching)
│   ├── images                              <- Images that are used in example Jupyter notebooks
│   ├── references                          <- References to useful information (e.g. MLOps architecture diagram)
├── environments                            <- Environment-related artifacts (for code execution)
│   ├── conda                               <- Conda .yml files for dev and train/deploy environment
│   ├── docker                              <- Dockerfiles
│       ├── base_image                      <- Dockerfile which specifies the base image to build AML environments
├── notebooks                               <- Jupyter notebooks end-to-end ML system development walkthrough
│   ├── 00_environment_setup.ipynb          <- Conda env and jupyter kernel for dev; AML envs for dev, train, deploy 
│   ├── 01_dataset_setup.ipynb              <- Data download and upload to AML datastore; AML dataset registration
│   ├── 02_model_training.ipynb             <- Model training in AML; script and hyperdrive run; AML model registration
│   ├── 03_model_evaluation.ipynb           <- Evaluation of model on test set; addition of accuracy to AML model
│   ├── 04_model_deployment.ipynb           <-
│   ├── 05_model_training_pipeline.ipynb    <-
│   ├── 06_batch_scoring_pipeline.ipynb     <-
│   ├── 10_custom_vision_model.ipynb        <-
├── outputs                                 <- Output artifacts that are generated during ML lifecycle (e.g. models)
├── src                                     <- Source code 
│   ├── config                              <- Configuration files (e.g. model training & evaluation parameter file)
│   ├── deployment                          <- Scripts needed for inference (e.g score.py script)
│   ├── pipeline                            <- Scripts needed for automated model training with AML pipelines
│   ├── training                            <- Scripts needed for model training (e.g. train.py)
│   ├── utils                               <- Modularized utility functions that might be reused across the source code
├── testing                                 <- Testing-related artifacts
│   ├── integration                         <- Integration testing scripts
│   ├── smoke                               <- Smoke testing (e.g. model endpoint responsiveness)
│   ├── unit                                <- Unit testing scripts (e.g. test dataset and dataloader)
├── .amlignore                              <- Contains all artifacts that should not be snapshotted by Azure Machine Learning
├── .env                                    <- Contains necessary environment variables which are used from source code
├── .gitignore                              <- Contains all artifacts that should not be checked into the git repo
```

# Building and Running Image Locally
docker build -t dog_clf_flask_app .
docker run -ti -p 5000:5000 dog_clf_flask_app
docker run -ti -p 5000:5000 dog_clf_flask_app sh

# Pull Image from ACR
az login
az acr login --name 3d5545b15c4c49548d3823156fa90536
docker pull 3d5545b15c4c49548d3823156fa90536.azurecr.io/dog_clf_flask_app:31
docker run -ti -p 5000:5000 3d5545b15c4c49548d3823156fa90536.azurecr.io/dog_clf_flask_app:31

```python

```
