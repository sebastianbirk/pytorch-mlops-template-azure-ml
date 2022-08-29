### Web App for Flask App Deployment ###

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
  
  identity {
    type = "SystemAssigned"
  }
  
  app_settings = {
    "DOCKER_REGISTRY_SERVER_PASSWORD" = ""
    "DOCKER_REGISTRY_SERVER_URL" = "https://mlopstemplatecr${lower(random_id.suffix.hex)}.azurecr.io"
    "DOCKER_REGISTRY_SERVER_USERNAME" = "mlopstemplatecr${lower(random_id.suffix.hex)}"
    "WEBSITES_ENABLE_APP_SERVICE_STORAGE" = "false"
  }

  site_config {
    application_stack {
      docker_image     = "mcr.microsoft.com/appsvc/staticsite"
      docker_image_tag = "latest"
    }
  }
}
