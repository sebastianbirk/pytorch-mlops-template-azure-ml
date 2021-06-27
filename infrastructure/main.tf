### Azure Resource Manager & Azure DevOps Configuration ###

terraform {
  required_providers {
    azurerm = {
      source = "hashicorp/azurerm"
      version = ">= 2.26"
    }
    azuredevops = {
      source = "microsoft/azuredevops"
      version = ">=0.1.0"
    }
  }
}

provider "azurerm" {
  features {}
}

provider "azuredevops" {
  org_service_url       = var.ado_org_service_url
  personal_access_token = var.ado_personal_access_token
}

data "azurerm_client_config" "current" {}
