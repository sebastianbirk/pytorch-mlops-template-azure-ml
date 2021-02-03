#!/bin/bash

# Define parameters
resourceGroupName=sbirkrg$RANDOM
webJobStorageName=sbirkwjstorage$RANDOM
appServiceName=sbirkappservice$RANDOM
functionAppName=sbirkfuncapp$RANDOM
region=westeurope
deploymentContainerImageName="3d5545b15c4c49548d3823156fa90536.azurecr.io/package:208d198d9c9d809f6e26d15b57b17f727a61d8a93ef2c6cc960bc6d6e54da968"
dockerRegistryServerUrl="https://3d5545b15c4c49548d3823156fa90536.azurecr.io"
dockerRegistryServerUser="3d5545b15c4c49548d3823156fa90536"
dockerRegistryServerPassword="Z+Ji54H8QuUlVZz0xn8rWGEHgr+L9O6e"
dockerCustomImageName="3d5545b15c4c49548d3823156fa90536.azurecr.io/package:20210202060555"

# Create a resource group
az group create \
  --name $resourceGroupName\
  --location $region

# Create an azure storage account as web job store
az storage account create \
  --name $webJobStorageName \
  --location $region \
  --resource-group $resourceGroupName \
  --sku Standard_LRS

# Create an app service plan
az functionapp plan create \
  --name $appServiceName \
  --resource-group $resourceGroupName \
  --location $region \
  --sku B1
  --is-linux

# Create a function app
az functionapp create \
  --name $functionAppName \
  --storage-account $webJobStorageName \
  --plan $appServiceName \
  --resource-group $resourceGroupName \
  --functions-version 2 \
  --deployment-container-image-name $deploymentContainerImageName
