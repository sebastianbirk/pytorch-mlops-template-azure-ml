"""
This script is based on the pipeline scripts in https://github.com/microsoft/MLOpsPython. 
"""

# Append src folder to sys.path to be able to import all necessary modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import libraries
import argparse
# import joblib
import json
import os
import torch
from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Workspace

# Import created modules
from utils import  get_model_metrics, load_data, register_dataset
from training.train import fine_tune_model


def main():
    print("Running train_aml.py")

    parser = argparse.ArgumentParser("train")

    parser.add_argument("--model_name",
                        type=str,
                        help="Name of the Model",
                        default="fowl-model")

    parser.add_argument("--step_output",
                        type=str,
                        help="PipelineData output for passing data to the next step")

    parser.add_argument("--dataset_version",
                        type=str,
                        help="Dataset version")

    parser.add_argument("--data_file_path",
                        type=str,
                        help=("Data file path. if specified,\
                               a new version of the dataset will be registered"))

    parser.add_argument("--caller_run_id",
                        type=str,
                        help="Caller run id, for example ADF pipeline run id")

    parser.add_argument("--dataset_name",
                        type=str,
                        help=("Dataset name. Dataset must be passed by name\
                              to always get the desired dataset version"))

    args = parser.parse_args()

    print("Argument [model_name]: %s" % args.model_name)
    print("Argument [step_output]: %s" % args.step_output)
    print("Argument [dataset_version]: %s" % args.dataset_version)
    print("Argument [data_file_path]: %s" % args.data_file_path)
    print("Argument [caller_run_id]: %s" % args.caller_run_id)
    print("Argument [dataset_name]: %s" % args.dataset_name)

    model_name = args.model_name
    step_output_path = args.step_output
    dataset_version = args.dataset_version
    data_file_path = args.data_file_path
    dataset_name = args.dataset_name

    run = Run.get_context()

    print("Getting training parameters")
    # Load the training parameters from the parameters file
    with open("config/pipeline_parameters.json") as f:
        pars = json.load(f)
    try:
        train_args = pars["training"]
    except KeyError:
        print("Could not load training values from file.")
        train_args = {}

    # Log the training parameters
    print(f"Parameters: {train_args}")
    for (k, v) in train_args.items():
        run.log(k, v)
        run.parent.log(k, v)

    # Get the dataset
    if (dataset_name):
        if (data_file_path == "none"):
            dataset = Dataset.get_by_name(run.experiment.workspace, dataset_name, dataset_version)  # NOQA: E402, E501
        else:
            dataset = register_dataset(run.experiment.workspace,
                                       dataset_name,
                                       os.environ.get("DATASTORE_NAME"),
                                       data_file_path)
    else:
        e = ("No dataset provided")
        print(e)
        raise Exception(e)

    # Link dataset to the step run so it is trackable in the UI
    run.input_datasets["training_data"] = dataset
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

    # Evaluate and log the model metrics
    metrics = get_model_metrics(model, dataloaders, dataset_sizes)
    for (k, v) in metrics.items():
        run.log(k, v)
        run.parent.log(k, v)

    # Pass model file to next step
    os.makedirs(step_output_path, exist_ok=True)
    model_output_path = os.path.join(step_output_path, model_name)
    torch.save(model, model_output_path + ".pt")
    #joblib.dump(value=model, filename=model_output_path)

    # Also upload model file to run outputs for history
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", model_name)
    torch.save(model, output_path + ".pt")
    #joblib.dump(value=model, filename=output_path)

    run.tag("run_type", value="train")
    print(f"Tags now present for run: {run.tags}.")

    run.complete()


if __name__ == '__main__':
    main()