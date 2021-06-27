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
  default = "https://dev.azure.com/<ORG_NAME>"
}

variable "ado_personal_access_token" {
  default = "<PERSONAL_ACCESS_TOKEN>"
}