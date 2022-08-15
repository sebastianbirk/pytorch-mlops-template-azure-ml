resource "azurerm_service_plan" "mlops_template_sp" {
  name                = "mlopstemplatesp${lower(random_id.suffix.hex)}"
  resource_group_name = azurerm_resource_group.mlops_template_rg.name
  location            = azurerm_resource_group.mlops_template_rg.location
  os_type             = "Linux"
  sku_name            = "P1v2"
}

resource "azurerm_linux_web_app" "mlops_template_wa" {
  name                = "mlopstemplatewa${lower(random_id.suffix.hex)}"
  resource_group_name = azurerm_resource_group.mlops_template_rg.name
  location            = azurerm_resource_group.mlops_template_rg.location
  service_plan_id     = azurerm_service_plan.mlops_template_sp.id

  site_config {}
}
