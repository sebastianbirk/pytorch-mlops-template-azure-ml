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

resource "azuredevops_serviceendpoint_azurerm" "mlops_template_azurerm_serviceendpoint" {
  project_id                = azuredevops_project.mlops_template_project.id
  service_endpoint_name     = "azure-resource-connection"
  azurerm_spn_tenantid      = "00000000-0000-0000-0000-000000000000"
  azurerm_subscription_id   = "00000000-0000-0000-0000-000000000000"
  azurerm_subscription_name = "Example Subscription Name"
}
