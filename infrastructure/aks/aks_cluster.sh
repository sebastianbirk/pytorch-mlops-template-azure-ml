# Define parameters
resourceGroupName=aksrg$RANDOM
aksClusterName=aks$RANDOM
acrName=3d5545b15c4c49548d3823156fa90536
region=westeurope

# Create a resource group
az group create \
  --name $resourceGroupName\
  --location $region

# Option 1: Create an AKS cluster
az aks create \
    --resource-group $resourceGroupName \
    --name $aksClusterName \
    --node-count 2 \
    --generate-ssh-keys \
    --attach-acr $acrName

# Option 2: Or use existing one instead
az aks update \
    --resource-group $resourceGroupName \
    --name $aksClusterName \
    --attach-acr $acrName

# Install kubectl
sudo az aks install-cli

# Configure kubectl to connect to the AKS cluster
az aks get-credentials \
    --resource-group $resourceGroupName \
    --name $aksClusterName
