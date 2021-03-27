This repository is work in progress. Soon you will find an overview and table of contents here. 
WARNING: the code is not finalized yet and certain parts may not run.

# Building and Running Image Locally
docker build -t dog_clf_flask_app .
docker run -ti -p 5000:5000 dog_clf_flask_app
docker run -ti -p 5000:5000 dog_clf_flask_app sh

# Pull Image from ACR
az login
az acr login --name 3d5545b15c4c49548d3823156fa90536
docker pull 3d5545b15c4c49548d3823156fa90536.azurecr.io/dog_clf_flask_app:31
docker run -ti -p 5000:5000 3d5545b15c4c49548d3823156fa90536.azurecr.io/dog_clf_flask_app:31