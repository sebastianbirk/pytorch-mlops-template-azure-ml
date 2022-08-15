### Azure Resource Variables ###

variable "resource_group" {
  default = "mlopstemplaterg"
}

variable "location" {
  default = "West Europe"
}

variable "aks_cluster_name" {
  default = "aks-cluster"
}

### Suffix for Azure Resource Names ###

resource "random_id" "suffix" {
  byte_length = 3
}

### Azure DevOps Variables (these should be specified as env variables) ###

variable "ado_org_service_url" {
  default = "https://dev.azure.com/<ADO_ORG_NAME>" # This needs to be defined as env variable
}

variable "ado_personal_access_token" {
  default = "<ADO_PAT>" # This needs to be defined as env variable
}

variable "tenant_id" {
  default = "<TENANT_ID>" # This needs to be defined as env variable
}

variable "subscription_id" {
  default = "<SUBSCRIPTION_ID>" # This needs to be defined as env variable
}

variable "subscription_name" {
  default = "<SUBSCRIPTION_NAME>" # This needs to be defined as env variable
}
