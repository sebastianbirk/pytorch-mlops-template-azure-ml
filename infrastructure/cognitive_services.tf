### Azure Cognitive Services (Custom Vision) ###

resource "azurerm_cognitive_account" "mlops_template_cvp" {
  name                = "mlopstemplatecvp${lower(random_id.suffix.hex)}"
  location            = azurerm_resource_group.mlops_template_rg.location
  resource_group_name = azurerm_resource_group.mlops_template_rg.name
  kind                = "CustomVision.Prediction"
  sku_name            = "S0"
}

resource "azurerm_cognitive_account" "mlops_template_cvt" {
  name                = "mlopstemplatecvt${lower(random_id.suffix.hex)}"
  location            = azurerm_resource_group.mlops_template_rg.location
  resource_group_name = azurerm_resource_group.mlops_template_rg.name
  kind                = "CustomVision.Training"
  sku_name            = "S0"
}