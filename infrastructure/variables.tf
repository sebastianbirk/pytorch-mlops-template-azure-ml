### Azure Resource Variables ###

variable "resource_group" {
  default = "mlopstemplaterg"
}

variable "location" {
  default = "West Europe"
}

variable "aks_name" {
  default = "aks-cluster"
}

### Suffix for Azure Resource Names ###

resource "random_id" "suffix" {
  byte_length = 8
}