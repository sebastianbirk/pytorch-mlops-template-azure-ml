### Storage Account for AML Workspace ###

resource "azurerm_storage_account" "mlops_template_sa" {
  name                     = "mlopstemplatesa${lower(random_id.suffix.hex)}"
  location                 = azurerm_resource_group.mlops_template_rg.location
  resource_group_name      = azurerm_resource_group.mlops_template_rg.name
  account_tier             = "Standard"
  account_replication_type = "GRS"
}