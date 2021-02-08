import os
from aml.utils.env_variables import EnvVariables
from azureml.core import Environment, Workspace
from azureml.core.runconfig import DEFAULT_CPU_IMAGE, DEFAULT_GPU_IMAGE


def get_environment(
    workspace: Workspace,
    environment_name: str,
    conda_dependencies_file: str,
    create_new: bool = False,
    enable_docker: bool = None,
    use_gpu_base_image: bool = False,
    base_image: str = ""
) -> Environment:
    """
    Retrieve existing environment or create new one according to input specifications
    """

    try:
        # Load environment variables
        env_variables = EnvVariables()
        
        # Retrieve environment with environment_name if it exists
        environments = Environment.list(workspace=workspace)
        restored_environment = None
        for env in environments:
            if env == environment_name:
                restored_environment = environments[environment_name]

        # Create new environment if it does not exist or create_new=True
        if restored_environment is None or create_new:
            new_env = Environment.from_conda_specification(
                environment_name,
                os.path.join(env_variables.conda_env_directory, conda_dependencies_file) # NOQA: E501
            ) # NOQA: E501
            restored_environment = new_env

            # Enable docker and configure base_image
            if enable_docker is not None:
                restored_environment.docker.enabled = enable_docker
                # Use default base image
                if base_image == "":
                    restored_environment.docker.base_image = DEFAULT_GPU_IMAGE if use_gpu_base_image \
                                                                               else DEFAULT_CPU_IMAGE  # NOQA: E501
                # Use specified base image
                else:
                    restored_environment.docker.base_image = base_image

            # Register environment in AML workspace
            restored_environment.register(workspace)

        if restored_environment is not None:
            print(restored_environment)

        return restored_environment
        
    except Exception as e:
        print(e)
        exit(1)