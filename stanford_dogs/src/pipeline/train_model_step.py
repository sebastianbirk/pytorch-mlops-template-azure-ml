"""
This script is based on the pipeline scripts in https://github.com/microsoft/MLOpsPython. 
"""

# Append src folder to sys.path to be able to import all necessary modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import libraries
import argparse
import json
import os
import torch
from azureml.core import Dataset, Datastore, Workspace
from azureml.core.run import Run

# Import created modules
from utils import EnvVariables, get_model_metrics, load_data, register_dataset
from training.train import fine_tune_model


def main():
    print("Running train_model_step.py")

    # Get AML run context
    run = Run.get_context()
    
    # Load environment variables
    env_variables = EnvVariables()

    # If script is run locally during development
    if (run.id.startswith("OfflineRun")):
    
        # Manually specify run_id for step run (useful to query previous runs)
        run_id = "1337leet-zf12-a7h4-kl12-jh3u8qw3l66hr"

        workspace_name = env_variables.workspace_name
        subscription_id = env_variables.subscription_id
        resource_group = env_variables.resource_group
        experiment_name = env_variables.experiment_name
    
        # Retrieve workspace and experiment using specified env_variables
        ws = Workspace.get(name=workspace_name,
                           subscription_id=subscription_id,
                           resource_group=resource_group)
        exp = Experiment(ws, experiment_name)
    
    # If script is run remotely (online)
    else:
        # Retrieve workspace and experiment from run
        ws = run.experiment.workspace
        exp = run.experiment
        run_id = run.parent.id

    # Parse input arguments
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--caller_run_id",
                        type=str,
                        help="Pipeline caller run id, for example ADF pipeline run id"
                        default="local")
    parser.add_argument("--dataset_name",
                        type=str,
                        help="AML dataset name",
                        default="stanford-dogs-dataset")
    parser.add_argument("--dataset_version",
                        type=str,
                        help="AML dataset version",
                        default=1.0)
    parser.add_argument("--data_file_path",
                        type=str,
                        help=("Local file path to data. If specified, a new version \
                               of the dataset will be registered to AML"),
                        default="none")
    parser.add_argument("--model_name",
                        type=str,
                        help="Name of the Model",
                        default="dog-clf-model")
    parser.add_argument("--step_output",
                        type=str,
                        help="Pipeline output for passing data to the next step")
    args = parser.parse_args()

    print(f"Argument [caller_run_id]: {args.caller_run_id}")
    print(f"Argument [dataset_name]: {args.dataset_name}")
    print(f"Argument [dataset_version]: {args.dataset_version}")
    print(f"Argument [data_file_path]: {args.data_file_path}")
    print(f"Argument [model_name]: {args.model_name}")
    print(f"Argument [step_output]: {args.step_output}")

    dataset_name = args.dataset_name
    dataset_version = args.dataset_version
    data_file_path = args.data_file_path
    model_name = args.model_name
    model_file = model_name + ".pt" # Add file extension according to used framework
    step_output_path = args.step_output

    print("Getting training parameters")
    # Load the training parameters from the parameters file
    with open("config/pipeline_parameters.json") as f:
        pars = json.load(f)
    try:
        train_args = pars["training"]
    except KeyError:
        print("Could not load training parameters from file.")
        train_args = {}

    # Log the training parameters to the step run and to the pipeline run
    print(f"Parameters: {train_args}")
    for (k, v) in train_args.items():
        run.log(k, v) # step run
        run.parent.log(k, v) # pipeline run

    # Get the dataset
    if (dataset_name): # if dataset_name is not an emptry string
        if (data_file_path == "none"):
            dataset = Dataset.get_by_name(workspace=run.experiment.workspace,
                                          name=dataset_name,
                                          version=dataset_version)
        else:
            dataset = register_dataset(aml_workspace=run.experiment.workspace,
                                       dataset_name=dataset_name,
                                       datastore_name=env_variables.datastore_name,
                                       file_path=data_file_path)
    else:
        e = ("No dataset provided.")
        print(e)
        raise Exception(e)

    # Link dataset to the step run so it is trackable in the UI
    run.input_datasets["training_data"] = dataset

    # Tag dataset_id to the pipeline run
    run.parent.tag("dataset_id", value=dataset.id)

    # Download the dataset to the compute target
    dataset.download(target_path="./data", overwrite=True)

    # Load training and validation data
    dataloaders, dataset_sizes, class_names = load_data("./data")

    # Train the model
    model = fine_tune_model(
        num_epochs=train_args["num_epochs"],
        num_classes=len(class_names),
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        learning_rate=train_args["learning_rate"],
        momentum=train_args["momentum"])

    # Evaluate and log the model metrics to the step run and the pipeline run
    metrics = get_model_metrics(model, dataloaders, dataset_sizes)
    for (k, v) in metrics.items():
        run.log(k, v) # step run
        run.parent.log(k, v) # pipeline run

    # Pass model file to next pipeline step
    os.makedirs(step_output_path, exist_ok=True)
    model_output_path = os.path.join(step_output_path, model_file)
    torch.save(model, model_output_path) # Replace saving method according to used framework (e.g. joblib for sklearn)

    # Also upload model file to run outputs for history
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", model_file)
    torch.save(model, output_path) # Replace saving method according to used framework (e.g. joblib for sklearn)

    run.tag("run_type", value="train")
    print(f"The following tags are now present: {run.tags}.")

    run.complete()


if __name__ == "__main__":
    main()