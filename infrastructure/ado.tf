### Azure DevOps Project ###

resource "azuredevops_project" "mlops_template_project" {
  name               = "mlopstemplateproject"
  description        = "PyTorch MLOps Template Project for Azure ML"
  visibility         = "private"
  version_control    = "Git"
  work_item_template = "Agile"
}

### Template Git Repository ###

resource "azuredevops_git_repository" "mlops_template_git_repo" {
  project_id = azuredevops_project.mlops_template_project.id
  name       = "pytorch-mlops-template-azure-ml"
  initialization {
    init_type   = "Import"
    source_type = "Git"
    source_url  = "https://github.com/sebastianbirk/pytorch-mlops-template-azure-ml"
  }
}

### Azure DevOps Pipelines ###

resource "azuredevops_build_definition" "mlops_template_ci_pipeline" {
  project_id = azuredevops_project.mlops_template_project.id
  name       = "ci_pipeline"
  path       = "\\${azuredevops_project.mlops_template_project.name}"

  ci_trigger {
    use_yaml = true
  }

  repository {
    repo_type   = "TfsGit"
    repo_id     = azuredevops_git_repository.mlops_template_git_repo.id
    branch_name = azuredevops_git_repository.mlops_template_git_repo.default_branch
    yml_path    = "ado_pipelines/ci_pipeline.yml"
  }
}

resource "azuredevops_build_definition" "mlops_template_docker_image_pipeline" {
  project_id = azuredevops_project.mlops_template_project.id
  name       = "docker_image_pipeline"
  path       = "\\${azuredevops_project.mlops_template_project.name}"

  ci_trigger {
    use_yaml = true
  }

  repository {
    repo_type   = "TfsGit"
    repo_id     = azuredevops_git_repository.mlops_template_git_repo.id
    branch_name = azuredevops_git_repository.mlops_template_git_repo.default_branch
    yml_path    = "ado_pipelines/docker_image_pipeline.yml"
  }
}

resource "azuredevops_build_definition" "mlops_template_flask_app_deployment_pipeline" {
  project_id = azuredevops_project.mlops_template_project.id
  name       = "flask_app_deployment_pipeline"
  path       = "\\${azuredevops_project.mlops_template_project.name}"

  ci_trigger {
    use_yaml = true
  }

  repository {
    repo_type   = "TfsGit"
    repo_id     = azuredevops_git_repository.mlops_template_git_repo.id
    branch_name = azuredevops_git_repository.mlops_template_git_repo.default_branch
    yml_path    = "ado_pipelines/flask_app_deployment_pipeline.yml"
  }
}

resource "azuredevops_build_definition" "mlops_template_model_deployment_pipeline" {
  project_id = azuredevops_project.mlops_template_project.id
  name       = "model_deployment_pipeline"
  path       = "\\${azuredevops_project.mlops_template_project.name}"

  ci_trigger {
    use_yaml = true
  }

  repository {
    repo_type   = "TfsGit"
    repo_id     = azuredevops_git_repository.mlops_template_git_repo.id
    branch_name = azuredevops_git_repository.mlops_template_git_repo.default_branch
    yml_path    = "ado_pipelines/model_deployment_pipeline.yml"
  }
}

resource "azuredevops_build_definition" "mlops_template_model_training_pipeline" {
  project_id = azuredevops_project.mlops_template_project.id
  name       = "model_training_pipeline"
  path       = "\\${azuredevops_project.mlops_template_project.name}"

  ci_trigger {
    use_yaml = true
  }

  repository {
    repo_type   = "TfsGit"
    repo_id     = azuredevops_git_repository.mlops_template_git_repo.id
    branch_name = azuredevops_git_repository.mlops_template_git_repo.default_branch
    yml_path    = "ado_pipelines/model_training_pipeline.yml"
  }
}

### Azure DevOps Service Connections ###

resource "azuredevops_serviceendpoint_azurerm" "mlops_template_serviceendpoint_azurerm" {
  project_id                = azuredevops_project.mlops_template_project.id
  service_endpoint_name     = "azure-resource-connection"
  azurerm_spn_tenantid      = var.tenant_id
  azurerm_subscription_id   = var.subscription_id
  azurerm_subscription_name = var.subscription_name
}

resource "azuredevops_serviceendpoint_azurecr" "mlops_template_serviceendpoint_azurecr" {
  project_id                = azuredevops_project.mlops_template_project.id
  service_endpoint_name     = "acr-connection"
  resource_group            = azurerm_resource_group.mlops_template_rg.name
  azurecr_spn_tenantid      = var.tenant_id
  azurecr_name              = azurerm_container_registry.mlops_template_cr.name
  azurecr_subscription_id   = var.subscription_id
  azurecr_subscription_name = var.subscription_name
}
