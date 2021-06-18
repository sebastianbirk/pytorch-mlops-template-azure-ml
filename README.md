# PyTorch MLOps Template Azure ML

![pytorch mlops template aml banner](https://github.com/sebastianbirk/pytorch-mlops-template-azure-ml/blob/master/docs/images/pytorch_mlops_template_aml_banner.png)

## Description

This repository contains an end-to-end implementation of an image classification model built with PyTorch on Azure ML, leveraging Azure ML's MLOps capabilities. It can be used as a template repository to quickly bootstrap similar modeling workloads (from development to production).

# Building and Running Image Locally
docker build -t dog_clf_flask_app .
docker run -ti -p 5000:5000 dog_clf_flask_app
docker run -ti -p 5000:5000 dog_clf_flask_app sh

# Pull Image from ACR
az login
az acr login --name 3d5545b15c4c49548d3823156fa90536
docker pull 3d5545b15c4c49548d3823156fa90536.azurecr.io/dog_clf_flask_app:31
docker run -ti -p 5000:5000 3d5545b15c4c49548d3823156fa90536.azurecr.io/dog_clf_flask_app:31
