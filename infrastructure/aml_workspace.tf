### Azure Machine Learning Workspace ###

resource "azurerm_machine_learning_workspace" "mlops_template_ws" {
  name                    = "mlopstemplatews${lower(random_id.suffix.hex)}"
  location                = azurerm_resource_group.mlops_template_rg.location
  resource_group_name     = azurerm_resource_group.mlops_template_rg.name
  application_insights_id = azurerm_application_insights.mlops_template_ai.id
  key_vault_id            = azurerm_key_vault.mlops_template_kv.id
  storage_account_id      = azurerm_storage_account.mlops_template_sa.id

  identity {
    type = "SystemAssigned"
  }
}

### Azure Machine Learning Computes ###

resource "null_resource" "mlops_template_compute_targets" {
  triggers = {
    aml_workspace_name = "${azurerm_machine_learning_workspace.mlops_template_ws.id}"
  }

  provisioner "local-exec" {
    command="az ml computetarget create amlcompute --max-nodes 2 --min-nodes 0 --name cpu-cluster --vm-size Standard_DS3_v2 --idle-seconds-before-scaledown 300 --assign-identity [system] --resource-group ${azurerm_machine_learning_workspace.mlops_template_ws.resource_group_name} --workspace-name ${azurerm_machine_learning_workspace.mlops_template_ws.name}"
  }

  provisioner "local-exec" {
    command="az ml computetarget create amlcompute --max-nodes 2 --min-nodes 0 --name gpu-cluster --vm-size Standard_NC6 --idle-seconds-before-scaledown 300 --assign-identity [system] --resource-group ${azurerm_machine_learning_workspace.mlops_template_ws.resource_group_name} --workspace-name ${azurerm_machine_learning_workspace.mlops_template_ws.name}"
  }

  provisioner "local-exec" {
    command="az ml computetarget create computeinstance --name mlopstemplateci${lower(random_id.suffix.hex)} --vm-size Standard_DS3_v2 --resource-group ${azurerm_machine_learning_workspace.mlops_template_ws.resource_group_name} --workspace-name ${azurerm_machine_learning_workspace.mlops_template_ws.name}"
  }
 
  depends_on = [azurerm_machine_learning_workspace.mlops_template_ws]
}