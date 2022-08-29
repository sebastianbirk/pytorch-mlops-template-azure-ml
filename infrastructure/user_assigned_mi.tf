resource "azurerm_user_assigned_identity" "mlops_template_user_assigned_mi" {
  name = "mlopstemplateuami${lower(random_id.suffix.hex)}"
  resource_group_name = azurerm_resource_group.mlops_template_rg.name
  location            = azurerm_resource_group.mlops_template_rg.location
}
