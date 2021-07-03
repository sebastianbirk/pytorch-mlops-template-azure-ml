### AKS Cluster attached to AML Workspace ###

resource "azurerm_kubernetes_cluster" "mlops_template_aks" {
  name                = "mlopstemplateaks${lower(random_id.suffix.hex)}"
  location            = azurerm_resource_group.mlops_template_rg.location
  resource_group_name = azurerm_resource_group.mlops_template_rg.name
  dns_prefix          = "aks"

  default_node_pool {
    name       = "default"
    node_count = 3
    vm_size    = "Standard_DS2_v2"
  }
  
  identity {
    type = "SystemAssigned"
  }
  
  provisioner "local-exec" {
    command = "az ml computetarget attach aks -n ${var.aks_cluster_name} -i ${azurerm_kubernetes_cluster.mlops_template_aks.id} -g ${azurerm_resource_group.mlops_template_rg.name} -w ${azurerm_machine_learning_workspace.mlops_template_ws.name}"
  }
  
  depends_on = [azurerm_machine_learning_workspace.mlops_template_ws]
}