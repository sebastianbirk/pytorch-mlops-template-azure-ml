### Key Vault for AML Workspace ###

resource "azurerm_key_vault" "mlops_template_kv" {
  name                     = "mlopstemplatekv${lower(random_id.suffix.hex)}"
  location                 = azurerm_resource_group.mlops_template_rg.location
  resource_group_name      = azurerm_resource_group.mlops_template_rg.name
  tenant_id                = data.azurerm_client_config.current.tenant_id
  sku_name                 = "premium"
  purge_protection_enabled = true
}