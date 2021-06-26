### Resource Group for all Resources ###

resource "azurerm_resource_group" "mlops_template_rg" {
  name     = var.resource_group
  location = var.location
}