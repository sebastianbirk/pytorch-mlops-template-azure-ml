# Import libraries
import os
from azureml.core import Dataset, Datastore, Workspace, Environment
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.exceptions import ComputeTargetException
from azureml.core.runconfig import DEFAULT_CPU_IMAGE, DEFAULT_GPU_IMAGE

# Import created modules
from .env_variables import EnvVariables


def get_compute(workspace: Workspace, compute_name: str, vm_size: str, for_batch_scoring: bool = False):  # NOQA E501
    try:
        if compute_name in workspace.compute_targets:
            compute_target = workspace.compute_targets[compute_name]
            if compute_target and type(compute_target) is AmlCompute:
                print("Found existing compute target " + compute_name + " so using it.") # NOQA
        else:
            e = EnvVariables()
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=vm_size,
                vm_priority=e.vm_priority if not for_batch_scoring else e.vm_priority_scoring,  # NOQA E501
                min_nodes=e.min_nodes if not for_batch_scoring else e.min_nodes_scoring,  # NOQA E501
                max_nodes=e.max_nodes if not for_batch_scoring else e.max_nodes_scoring,  # NOQA E501
                idle_seconds_before_scaledown="300"
                #    #Uncomment the below lines for VNet support
                #    vnet_resourcegroup_name=vnet_resourcegroup_name,
                #    vnet_name=vnet_name,
                #    subnet_name=subnet_name
            )
            compute_target = ComputeTarget.create(
                workspace, compute_name, compute_config
            )
            compute_target.wait_for_completion(
                show_output=True, min_node_count=None, timeout_in_minutes=10
            )
        return compute_target
    except ComputeTargetException as ex:
        print(ex)
        print("An error occurred trying to provision compute.")
        exit(1)


def get_environment(
    workspace: Workspace,
    environment_name: str,
    conda_dependencies_file: str,
    create_new: bool = False,
    enable_docker: bool = None,
    use_gpu: bool = False
):
    try:
        env_variables = EnvVariables()
        environments = Environment.list(workspace=workspace)
        restored_environment = None
        for env in environments:
            if env == environment_name:
                restored_environment = environments[environment_name]

        if restored_environment is None or create_new:
            new_env = Environment.from_conda_specification(
                environment_name,
                os.path.join(conda_dependencies_file),  # NOQA: E501
            )  # NOQA: E501
            restored_environment = new_env
            if enable_docker is not None:
                restored_environment.docker.enabled = enable_docker
                restored_environment.docker.base_image = DEFAULT_GPU_IMAGE if use_gpu else DEFAULT_CPU_IMAGE  # NOQA: E501
            restored_environment.register(workspace)

        if restored_environment is not None:
            print(restored_environment)
        return restored_environment
    except Exception as e:
        print(e)
        exit(1)


def register_dataset(
    aml_workspace: Workspace,
    dataset_name: str,
    datastore_name: str,
    file_path: str
) -> Dataset:
    datastore = Datastore.get(aml_workspace, datastore_name)
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, file_path))
    dataset = dataset.register(workspace=aml_workspace,
                               name=dataset_name,
                               create_new_version=True)

    return dataset