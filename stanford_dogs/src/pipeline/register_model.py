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
import sys
import torch
from azureml.core import Run, Experiment, Workspace, Dataset
from azureml.core.model import Model as AMLModel
from pathlib import Path

# Import created modules
from utils import EnvVariables, register_aml_model


def main():

    run = Run.get_context()

    # If script is run locally during development (offline)
    if (run.id.startswith("OfflineRun")):

        # Load environment variables
        env_variables = EnvVariables()

        workspace_name = env_variables.workspace_name
        experiment_name = env_variables.experiment_name
        resource_group = env_variables.resource_group
        subscription_id = env_variables.subscription_id

        # Manually specify run_id (useful to query previous runs)
        run_id = "bd184a18-2ac8-4951-8e78-e290bef3b012"

        # Retrieve workspace and experiment using specified env_variables
        ws = Workspace.get(name=workspace_name,
                           subscription_id=subscription_id,
                           resource_group=resource_group)
        exp = Experiment(ws, experiment_name)

    # if script is run remotely (online)
    else:
        # Retrieve workspace and experiment from run
        ws = run.experiment.workspace
        exp = run.experiment
        run_id = "amlcompute"

    parser = argparse.ArgumentParser("register")

    parser.add_argument("--run_id",
                        type=str,
                        help="Training run ID")

    parser.add_argument("--model_name",
                        type=str,
                        help="Name of the Model",
                        default="fowl-model")

    parser.add_argument("--step_input",
                        type=str,
                        help="Input from previous steps")

    args = parser.parse_args()

    if (args.run_id is not None):
        run_id = args.run_id

    if (run_id == "amlcompute"):
        run_id = run.parent.id

    model_name = args.model_name
    model_path = args.step_input

    print("Getting registration parameters")

    # Load the registration parameters from the parameters file
    with open("config/pipeline_parameters.json") as f:
        pars = json.load(f)
    try:
        register_args = pars["registration"]
    except KeyError:
        print("Could not load registration values from file.")
        register_args = {"tags": []}

    model_tags = {}
    for tag in register_args["tags"]:
        try:
            # Get evaluation metrics from parent run and tag them to model
            mtag = run.parent.get_metrics()[tag]
            model_tags[tag] = mtag
        except KeyError:
            print(f"Could not find {tag} metric on parent run.")

    # Load the model
    print("Loading model from " + model_path)
    model_file = os.path.join(model_path, model_name) + ".pt"
    model = torch.load(model_file)
    # model = joblib.load(model_file)
    parent_tags = run.parent.get_tags()
    try:
        build_id = parent_tags["BuildId"]
    except KeyError:
        build_id = None
        print("BuildId tag not found on parent run.")
        print(f"Tags present: {parent_tags}")
    try:
        build_uri = parent_tags["BuildUri"]
    except KeyError:
        build_uri = None
        print("BuildUri tag not found on parent run.")
        print(f"Tags present: {parent_tags}")

    if (model is not None):
        dataset_id = parent_tags["dataset_id"]
        if (build_id is None):
            register_aml_model(
                model_file,
                model_name,
                model_tags,
                exp,
                run_id,
                dataset_id)
        elif (build_uri is None):
            register_aml_model(
                model_file,
                model_name,
                model_tags,
                exp,
                run_id,
                dataset_id,
                build_id)
        else:
            register_aml_model(model_file,
                               model_name,
                               model_tags,
                               exp,
                               run_id,
                               dataset_id,
                               build_id,
                               build_uri)
    else:
        print("Model not found. Skipping model registration.")
        sys.exit(0)


if __name__ == '__main__':
    main()