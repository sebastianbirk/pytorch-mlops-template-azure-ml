### Container Registry for AML Workspace ###

resource "azurerm_container_registry" "mlops_template_cr" {
  name                = "mlopstemplatecr${lower(random_id.suffix.hex)}"
  resource_group_name = azurerm_resource_group.mlops_template_rg.name
  location            = azurerm_resource_group.mlops_template_rg.location
  sku                 = "Premium"
  admin_enabled       = true
}
