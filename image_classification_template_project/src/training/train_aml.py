# Add root directory to sys path to be able to import all modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import other libraries
import argparse
import joblib
import json
import os
import torch
from aml.utils import register_dataset
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Dataset, Datastore, Experiment, Workspace
from azureml.core.run import Run
from src.utils import eval_model, fine_tune_model, load_data


def main():

    print("=" * 20)
    print("RUNNING PIPELINE TRAIN SCRIPT")
    print("=" * 20)

    # Get the run context
    run = Run.get_context()

    if (run.id.startswith("OfflineRun")):
        print("Run is an offline run")
        print("-" * 20)
        from dotenv import load_dotenv
        # For local development, set values in this section
        print("Load environment variables...")
        load_dotenv(dotenv_path=Path(__file__).parents[2] / "config/.env")
        tenant_id = os.environ.get("TENANT_ID")
        subscription_id = os.environ.get("SUBSCRIPTION_ID")
        resource_group = os.environ.get("RESOURCE_GROUP")
        workspace_name = os.environ.get("WORKSPACE_NAME")
        experiment_name = os.environ.get("EXPERIMENT_NAME")
        app_id = os.environ.get("SP_APP_ID")
        app_secret = os.environ.get("SP_APP_SECRET")

        # run_id useful to query previous runs
        run_id = "57fee47f-5ae8-441c-bc0c-d4c371f32d70"

        print("Set up service principal authentication...")
        svc_pr_auth = ServicePrincipalAuthentication(
            tenant_id=tenant_id,
            service_principal_id=app_id,
            service_principal_password=app_secret)

        print("Connect to workspace...")
        ws = Workspace.get(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group,
            auth=svc_pr_auth)

        print("Retrieve experiment...")
        exp = Experiment(ws, experiment_name)

    else:
        print("Run is an online run")
        print("-" * 20)
        print("Connect to workspace...")
        ws = run.experiment.workspace
        print("Retrieve experiment...")
        exp = run.experiment
        run_id = "amlcompute"

    print("Parse script input parameters...")
    parser = argparse.ArgumentParser("train")

    parser.add_argument(
        "--model_file_name",
        type=str,
        help="Name of the model file",
        default="fowl_model.pt")

    parser.add_argument(
        "--step_output_path",
        type=str,
        help=("Output for passing data to next step"))

    parser.add_argument(
        "--dataset_name",
        type=str,
        help=("Dataset name. Dataset must be passed by name \
              to always get the desired dataset version \
              rather than the one used during pipeline creation"))

    parser.add_argument(
        "--dataset_version",
        type=str,
        help=("Dataset version"))

    parser.add_argument(
        "--data_file_path",
        type=str,
        help=("Data file path. If specified,\
               a new version of the dataset will be registered"))

    parser.add_argument(
        "--caller_run_id",
        type=str,
        help=("Caller run id, for example ADF pipeline run id"))

    args = parser.parse_args()

    print("Script input parameters are:")
    print(f"Argument [model_file_name]: {args.model_file_name}")
    print(f"Argument [step_output_path]: {args.step_output_path}")
    print(f"Argument [dataset_name]: {args.dataset_name}")
    print(f"Argument [dataset_version]: {args.dataset_version}")
    print(f"Argument [data_file_path]: {args.data_file_path}")
    print(f"Argument [caller_run_id]: {args.caller_run_id}")

    model_file_name = args.model_file_name
    step_output_path = args.step_output_path
    dataset_name = args.dataset_name
    dataset_version = args.dataset_version
    data_file_path = args.data_file_path

    print("Retrieving training parameters from config...")

    # Load the training parameters from the parameters file
    with open(Path(__file__).resolve().parents[2] / "config/parameters.json") as f:
        pars = json.load(f)
    try:
        train_args = pars["training"]
    except KeyError:
        print("Could not load training hyperparameters from config")
        train_args = {}

    print("Config training hyperparameters are:")
    # Log the training parameters
    print(f"Parameters: {train_args}")
    for (k, v) in train_args.items():
        run.log(k, v)
        # run.parent.log(k, v)

    # Get the dataset
    if (dataset_name):
        if (data_file_path == "none"):
            print("Retrieving dataset...")
            target_path = "data"
            dataset = Dataset.get_by_name(ws, dataset_name, dataset_version)  # NOQA: E402, E501
            dataset.download(target_path=target_path, overwrite=True)
        else:
            print("Registering dataset...")
            target_path = data_file_path
            dataset = register_dataset(ws,
                                       dataset_name,
                                       os.environ.get("DATASTORE_NAME"),
                                       data_file_path)
                        
    else:
        e = ("No dataset provided")
        print(e)
        raise Exception(e)

    if not run.id.startswith("OfflineRun"):
        print("Link dataset to run...")
        # Link dataset to the step run so it is trackable in the UI
        run.input_datasets["training_data"] = dataset
        run.parent.tag("dataset_id", value=dataset.id)

    print("Load data into dataloader...")
    dataloaders, dataset_sizes, class_names = load_data(target_path)

    print("Run model training function...")
    # Train the model
    model = fine_tune_model(num_epochs=train_args["num_epochs"],
                            num_classes=len(class_names),
                            dataloaders=dataloaders,
                            dataset_sizes=dataset_sizes,
                            learning_rate=train_args["learning_rate"],
                            momentum=train_args["momentum"])


    # # Evaluate and log the metrics returned from the train function
    # metrics = get_model_metrics(model, data)
    # for (k, v) in metrics.items():
    #     run.log(k, v)
    #     run.parent.log(k, v)

    # Pass model file to next step
    os.makedirs(step_output_path, exist_ok=True)
    model_output_path = os.path.join(step_output_path, model_file_name)
    torch.save(model, model_output_path)
    print("=" * 20)
    print(f"Model saved in {model_output_path}.")

    # Also upload model file to run outputs for history
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", model_file_name)
    torch.save(model, output_path)
    print(f"Model uploaded to run outputs.")

    run.tag("run_type", value="train")
    print(f"Tags now present for run: {run.tags}")

    run.complete()


if __name__ == '__main__':
    main()
