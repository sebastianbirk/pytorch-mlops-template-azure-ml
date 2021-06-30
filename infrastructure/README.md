# Infrastructure

## Run Terraform

Run below from the project root folder:
```console
$ cd infrastructure
$ az login --tenant <TENANT_ID>
$ az account set --subscription <SUBSCRIPTION_ID>
$ az extension add -n azure-cli-ml
$ terraform init
$ terraform plan
$ terraform apply
```

export TF_VAR_ado_org_service_url="https://dev.azure.com/mlopstemplateorg"
export TF_VAR_ado_personal_access_token="x4s2tvuditorjjv2cbsswdvydhdlswx5tzno5lkw4pldhf7fyupa"

## Resources
Installing Terraform:
https://learn.hashicorp.com/tutorials/terraform/install-cli

Azure DevOps Provider Terraform 
https://registry.terraform.io/providers/microsoft/azuredevops/latest/docs/guides/authenticating_using_the_personal_access_token
https://www.microsoft.com/de-de/techwiese/cloud-native-community-blog/einfuehrung-in-den-azure-devops-terraform-provider.aspx

Rolling out an AML enterprise environment via Terraform (Private Link Setup):
https://github.com/csiebler/azure-machine-learning-terraform

Provisioning an AML Workspace via Terraform:
https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/resources/machine_learning_workspace

Attaching an AKS Cluster to the AML Workspace:
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-kubernetes?tabs=azure-cli

It's important to know that terraform uses credentials stored by the Azure CLI to access the Azure resource manager.

https://fizzylogic.nl/2019/1/30/deploying-resources-on-azure-with-terraform
