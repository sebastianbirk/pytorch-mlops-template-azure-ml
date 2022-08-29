### Container Registry for AML Workspace ###

resource "azurerm_container_registry" "mlops_template_cr" {
  name                = "mlopstemplatecr${lower(random_id.suffix.hex)}"
  resource_group_name = azurerm_resource_group.mlops_template_rg.name
  location            = azurerm_resource_group.mlops_template_rg.location
  sku                 = "Premium"
  admin_enabled       = true
}

/*
resource "azurerm_role_assignment" "mlops_template_cr_uami_ra" {
  scope                = azurerm_container_registry.mlops_template_cr.id
  role_definition_name = "ACR Pull"
  principal_id         = azurerm_user_assigned_identity.mlops_template_user_assigned_mi.id
}
*/
