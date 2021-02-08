from aml.utils.env_variables import EnvVariables
from azureml.core import Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.exceptions import ComputeTargetException


def get_compute(
    workspace: Workspace,
    compute_name: str,
    vm_size: str,
    for_batch_scoring: bool = False,
    use_vnet: bool = False
) -> ComputeTarget: # NOQA E501
    """
    Retrieve existing compute or create new one according to input specifications
    """
    try:
        # Retrieve compute target with compute_name if it exists
        if compute_name in workspace.compute_targets:
            compute_target = workspace.compute_targets[compute_name]
            if compute_target and type(compute_target) is AmlCompute:
                print("Found existing compute target " + compute_name + ", so using it.") # NOQA
        else:
            # Load environment variables
            env_variables = EnvVariables()

            # Define compute target configuration (either without vnet or with vnet)
            if use_vnet == False:
                compute_config = AmlCompute.provisioning_configuration(
                    vm_size=vm_size,
                    vm_priority=env_variables.vm_priority if not for_batch_scoring \
                                                          else env_variables.vm_priority_scoring,  # NOQA E501
                    min_nodes=env_variables.min_nodes if not for_batch_scoring \
                                                      else env_variables.min_nodes_scoring,  # NOQA E501
                    max_nodes=env_variables.max_nodes if not for_batch_scoring \
                                                      else env_variables.max_nodes_scoring,  # NOQA E501
                    idle_seconds_before_scaledown="300"
                )
            else:
                compute_config = AmlCompute.provisioning_configuration(
                    vm_size=vm_size,
                    vm_priority=env_variables.vm_priority if not for_batch_scoring \
                                                           else env_variables.vm_priority_scoring,  # NOQA E501
                    min_nodes=env_variables.min_nodes if not for_batch_scoring \
                                                      else env_variables.min_nodes_scoring,  # NOQA E501
                    max_nodes=env_variables.max_nodes if not for_batch_scoring \
                                                      else env_variables.max_nodes_scoring,  # NOQA E501
                    idle_seconds_before_scaledown="300",
                    vnet_resourcegroup_name=env_variables.vnet_resourcegroup_name,
                    vnet_name=env_variables.vnet_name,
                    subnet_name=env_variables.subnet_name
                )

            # Create compute target
            compute_target = ComputeTarget.create(
                workspace, compute_name, compute_config
            )

            # Wait for completion
            compute_target.wait_for_completion(
                show_output=True, min_node_count=None, timeout_in_minutes=10
            )

        return compute_target
        
    except ComputeTargetException as ex:
        print(ex)
        print("An error occurred trying to provision compute.")
        exit(1)