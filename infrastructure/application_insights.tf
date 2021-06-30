### Application Insights for AML Workspace ###

resource "azurerm_application_insights" "mlops_template_ai" {
  name                = "mlopstemplateai${lower(random_id.suffix.hex)}"
  location            = azurerm_resource_group.mlops_template_rg.location
  resource_group_name = azurerm_resource_group.mlops_template_rg.name
  application_type    = "web"
}