# Add root directory to sys path to be able to import all modules
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

# Import libraries
from aml.utils import get_compute, get_environment
from aml.utils import EnvVariables
from azureml.core import Dataset, Datastore, Workspace
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from src.utils import download_data


def main():
    # Load environment variables
    env_variables = EnvVariables()

    # Retrieve workspace
    ws = Workspace.get(
        name=env_variables.workspace_name,
        subscription_id=env_variables.subscription_id,
        resource_group=env_variables.resource_group)
    print("Retrieved workspace:")
    print(ws)

    # Retrieve or create compute cluster
    aml_compute = get_compute(
        ws,
        env_variables.compute_name,
        env_variables.vm_size)
    if aml_compute is not None:
        print("Retrieved compute cluster:")
        print(aml_compute)

    # Retrieve or create environment
    environment = get_environment(
        ws,
        env_variables.aml_env_name,
        conda_dependencies_file=env_variables.aml_env_train_conda_dep_file,
        create_new=env_variables.rebuild_env)  
    print("Retrieved environment:")
    print(environment)

    # Create a run configuration
    run_config = RunConfiguration()
    run_config.environment = environment
    # Pass datastore as environment variable to the run configuration
    # Use default datastore if nothing else specified
    if env_variables.datastore_name:
        datastore_name = env_variables.datastore_name
    else:
        datastore_name = ws.get_default_datastore().name
    run_config.environment.environment_variables["DATASTORE_NAME"] = datastore_name  # NOQA: E501

    # Configure pipeline parameters
    model_name_param = PipelineParameter(name="model_name", default_value=env_variables.model_name)  # NOQA: E501
    dataset_version_param = PipelineParameter(
        name="dataset_version",
        default_value=env_variables.dataset_version)

    data_file_path_param = PipelineParameter(
        name="data_file_path",
         default_value="none")

    caller_run_id_param = PipelineParameter(
        name="caller_run_id",
         default_value="none") # NOQA: E501

    # Get dataset name
    dataset_name = env_variables.dataset_name

    # Check to see if dataset exists
    if dataset_name not in ws.datasets:
        # This call creates an example CSV from sklearn sample data. If you
        # have already bootstrapped your project, you can comment this line
        # out and use your own CSV.
        download_data(
            archive_file = "./data/fowl_data.zip",
            zip_dir = "./data")

        # Use a CSV to read in the data set.
        file_name = zip_dir + "/fowl_data/train"

        if not os.path.exists(file_name):
            raise Exception("Could not find example image")

        # Upload file to default datastore in workspace
        datatstore = Datastore.get(ws, datastore_name)
        target_path = "training-data/"
        datatstore.upload_files(
            files=[file_name],
            target_path=target_path,
            overwrite=True,
            show_progress=False,
        )

        # Register dataset
        path_on_datastore = os.path.join(target_path, file_name)
        dataset = Dataset.Tabular.from_delimited_files(
            path=(datatstore, path_on_datastore))
        dataset = dataset.register(
            workspace=ws,
            name=dataset_name,
            description="fowl image classification training data",
            tags={"format": "png"},
            create_new_version=True)

    # Create a PipelineData object to pass data between steps
    pipeline_data = PipelineData(
        "pipeline_data",
        datastore=ws.get_default_datastore())

    # Create PythonScriptStep for training
    train_step = PythonScriptStep(
        name="Train model",
        script_name=env_variables.train_script_path,
        compute_target=aml_compute,
        source_directory=env_variables.sources_directory_train,
        outputs=[pipeline_data],
        arguments=[
            "--model_name", model_name_param,
            "--step_output", pipeline_data,
            "--dataset_version", dataset_version_param,
            "--data_file_path", data_file_path_param,
            "--caller_run_id", caller_run_id_param,
            "--dataset_name", dataset_name,
        ],
        runconfig=run_config,
        allow_reuse=True,
    )
    print("Train model step created")

    # Create PythonScriptStep for evaluation
    evaluate_step = PythonScriptStep(
        name="Evaluate model",
        script_name=env_variables.evaluate_script_path,
        compute_target=aml_compute,
        source_directory=env_variables.sources_directory_train,
        arguments=[
            "--model_name", model_name_param,
            "--allow_run_cancel", env_variables.allow_run_cancel,
        ],
        runconfig=run_config,
        allow_reuse=False,
    )
    print("Evaluate model step created")

    # Create PythonScriptStep for model registration
    register_step = PythonScriptStep(
        name="Register model",
        script_name=env_variables.register_script_path,
        compute_target=aml_compute,
        source_directory=env_variables.sources_directory_train,
        inputs=[pipeline_data],
        arguments=[
            "--model_name", model_name_param,
            "--step_input", pipeline_data
        ],  # NOQA: E501
        runconfig=run_config,
        allow_reuse=False,
    )

    print("Register model step created")

    # Check run_evaluation flag to include or exclude evaluation step.
    if (env_variables.run_evaluation).lower() == "true":
        print("Include evaluation step before register step.")
        evaluate_step.run_after(train_step)
        register_step.run_after(evaluate_step)
        steps = [train_step, evaluate_step, register_step]
    else:
        print("Exclude evaluation step and directly run register step.")
        register_step.run_after(train_step)
        steps = [train_step, register_step]

    # Create training pipeline
    train_pipeline = Pipeline(workspace=ws, steps=steps)
    train_pipeline._set_experiment_name
    train_pipeline.validate()

    # Publish training pipeline
    published_pipeline = train_pipeline.publish(
        name=env_variables.pipeline_name,
        description="Model training/retraining pipeline",
        version=env_variables.build_id,
    )
    
    print(f"Published pipeline: {published_pipeline.name}")
    print(f"for build {published_pipeline.version}")


if __name__ == "__main__":
    main()