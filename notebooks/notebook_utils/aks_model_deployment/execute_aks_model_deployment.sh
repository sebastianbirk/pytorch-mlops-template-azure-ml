# Define parameters
resourceGroupName=mlopstemplaterg
region=westeurope

# Install kubectl
sudo az aks install-cli

# Attach ACR to AKS cluster
az aks update \
    --resource-group $resourceGroupName \
    --name $aksClusterName \
    --attach-acr $acrName

# Configure kubectl to connect to the AKS cluster
az aks get-credentials \
    --resource-group $resourceGroupName \
    --name $aksClusterName

# Apply kubernetes manifest
kubectl apply -f aks_deployment_manifest.yml
