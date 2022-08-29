### Azure Machine Learning Workspace ###

resource "azurerm_machine_learning_workspace" "mlops_template_ws" {
  name                    = "mlopstemplatews${lower(random_id.suffix.hex)}"
  location                = azurerm_resource_group.mlops_template_rg.location
  resource_group_name     = azurerm_resource_group.mlops_template_rg.name
  application_insights_id = azurerm_application_insights.mlops_template_ai.id
  key_vault_id            = azurerm_key_vault.mlops_template_kv.id
  storage_account_id      = azurerm_storage_account.mlops_template_sa.id
  container_registry_id   = azurerm_container_registry.mlops_template_cr.id

  identity {
    type = "SystemAssigned"
  }
}

### Azure Machine Learning Computes ###

resource "azurerm_machine_learning_compute_instance" "mlops_template_ci" {
  name                          = "mlopstemplateci${lower(random_id.suffix.hex)}"
  location                      = azurerm_resource_group.mlops_template_rg.location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.mlops_template_ws.id
  virtual_machine_size          = "STANDARD_DS3_V2"
}

resource "azurerm_machine_learning_compute_cluster" "mlops_template_cc_cpu" {
  name                          = "mlopstemplatecccpu${lower(random_id.suffix.hex)}"
  location                      = azurerm_resource_group.mlops_template_rg.location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.mlops_template_ws.id
  vm_priority                   = "Dedicated"
  vm_size                       = "STANDARD_DS3_V2"

  identity {
    type = "SystemAssigned"
  }

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 2
    scale_down_nodes_after_idle_duration = "PT15M" # 15 minutes
  }
}
  
resource "azurerm_machine_learning_compute_cluster" "mlops_template_cc_gpu" {
  name                          = "mlopstemplateccgpu${lower(random_id.suffix.hex)}"
  location                      = azurerm_resource_group.mlops_template_rg.location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.mlops_template_ws.id
  vm_priority                   = "Dedicated"
  vm_size                       = "STANDARD_NC6"

  identity {
    type = "SystemAssigned"
  }

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 2
    scale_down_nodes_after_idle_duration = "PT15M" # 15 minutes
  }
}
