"""
This script is based on the pipeline scripts in https://github.com/microsoft/MLOpsPython. 
"""

# Append src folder to sys.path to be able to import all necessary modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import libraries
import os
from azureml.core import Workspace, Dataset, Datastore
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep

# Import created modules
from utils import EnvVariables, get_compute, get_environment


def main():
    # Load environment variables
    env_variables = EnvVariables()

    # Get Azure Machine Learning workspace
    aml_workspace = Workspace.get(name=env_variables.workspace_name,
                                  subscription_id=env_variables.subscription_id,
                                  resource_group=env_variables.resource_group)

    print("Retrieved AML workspace:")
    print(aml_workspace)
    print("")

    # Get Azure Machine Learning cluster
    aml_compute = get_compute(workspace=aml_workspace,
                              compute_name=env_variables.compute_name,
                              vm_size=env_variables.vm_size)
                              
    if aml_compute is not None:
        print("Retrieved AML compute cluster:")
        print(aml_compute)
        print("")

    # Get Azure Machine Learning environment
    environment = get_environment(aml_workspace,
                                  env_variables.aml_training_environment_name,
                                  conda_dependencies_file=env_variables.aml_training_environment_conda_file_path,
                                  create_new=env_variables.rebuild_training_environment)

    # Create run configuration with retrieved environment                              
    run_config = RunConfiguration()
    run_config.environment = environment

    # Retrieve datastore if one is specified, otherwise retrieve workspace default datastore
    if env_variables.datastore_name:
        datastore_name = env_variables.datastore_name
    else:
        datastore_name = aml_workspace.get_default_datastore().name

    # Add datastore to run config environment variables
    run_config.environment.environment_variables["DATASTORE_NAME"] = datastore_name  # NOQA: E501

    # Define pipeline parameters
    model_name_param = PipelineParameter(name="model_name",
                                         default_value=env_variables.model_name)

    dataset_version_param = PipelineParameter(name="dataset_version",
                                              default_value=env_variables.dataset_version)

    data_file_path_param = PipelineParameter(name="data_file_path",
                                             default_value="none")

    caller_run_id_param = PipelineParameter(name="caller_run_id",
                                            default_value="none")

    # Get dataset name
    dataset_name = env_variables.dataset_name

    # Check to see if dataset exists
    # If not then use local data path to load data
    if dataset_name not in aml_workspace.datasets:

        # Local data path
        data_folder_name = "../data/fowl_data"

        if not os.path.exists(data_folder_name):
            raise Exception(f"Could not find data at {data_folder_name}."
                             "If you have bootstrapped your project, you will need to provide data.")  # NOQA: E501

        # Upload file to default datastore in workspace
        datatstore = Datastore.get(aml_workspace, datastore_name)
        target_path = "data/fowl_data"
        datatstore.upload_files(files=[file_name],
                                target_path=target_path,
                                overwrite=True,
                                show_progress=False)

        # Register dataset
        path_on_datastore = os.path.join(target_path, data_folder_name)
        dataset = Dataset.Tabular.from_delimited_files(path=(datatstore, path_on_datastore))
        dataset = dataset.register(workspace=aml_workspace,
                                   name=dataset_name,
                                   description="fowl dataset containing training and validation data",
                                   tags={"format": "JPG"},
                                   create_new_version=True)

    # Create a PipelineData to pass data between steps
    pipeline_data = PipelineData("pipeline_data", datastore=aml_workspace.get_default_datastore())

    # Create train step using PythonScriptStep
    train_step = PythonScriptStep(name="Train Model",
                                  script_name=env_variables.pipeline_train_script_path,
                                  compute_target=aml_compute,
                                  source_directory=env_variables.src_dir,
                                  outputs=[pipeline_data],
                                  arguments=["--model_name", model_name_param,
                                             "--step_output", pipeline_data,
                                             "--dataset_version", dataset_version_param,
                                             "--data_file_path", data_file_path_param,
                                             "--caller_run_id", caller_run_id_param,
                                             "--dataset_name", dataset_name],
                                  runconfig=run_config,
                                  allow_reuse=True)

    print("Training step has been created.")

    # Create evaluate step using PythonScriptStep
    evaluate_step = PythonScriptStep(name="Evaluate Model",
                                     script_name=env_variables.pipeline_evaluate_script_path,
                                     compute_target=aml_compute,
                                     source_directory=env_variables.src_dir,
                                     arguments=["--model_name", model_name_param,
                                                "--allow_run_cancel", env_variables.allow_pipeline_run_cancel],
                                     runconfig=run_config,
                                     allow_reuse=False)

    print("Evaluate step has been created.")

    # Create register step using PythonScriptStep
    register_step = PythonScriptStep(name="Register Model ",
                                     script_name=env_variables.pipeline_register_script_path,
                                     compute_target=aml_compute,
                                     source_directory=env_variables.src_dir,
                                     inputs=[pipeline_data],
                                     arguments=["--model_name", model_name_param,
                                                "--step_input", pipeline_data],
                                     runconfig=run_config,
                                     allow_reuse=False)

    print("Register step has been created.")

    # Check run_evaluation flag to include or exclude evaluation step.
    if (env_variables.pipeline_run_evaluation).lower() == "true":
        print("Include evaluation step before register step.")
        evaluate_step.run_after(train_step)
        register_step.run_after(evaluate_step)
        steps = [train_step, evaluate_step, register_step]
    else:
        print("Exclude evaluation step and directly run register step.")
        register_step.run_after(train_step)
        steps = [train_step, register_step]

    # Create the pipeline using the pipeline steps
    train_pipeline = Pipeline(workspace=aml_workspace, steps=steps)
    train_pipeline._set_experiment_name
    train_pipeline.validate()

    # Publish the pipeline
    published_pipeline = train_pipeline.publish(name=env_variables.pipeline_name,
                                                description="Model training/retraining pipeline",
                                                version=env_variables.build_id)
                                                
    print(f"Published pipeline: {published_pipeline.name}")
    print(f"for build {published_pipeline.version}")


if __name__ == "__main__":
    main()