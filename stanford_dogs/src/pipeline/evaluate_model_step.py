"""
This script is based on the pipeline scripts in https://github.com/microsoft/MLOpsPython. 
"""

# Append src (root) folder to sys.path to be able to import all necessary modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import libraries
import argparse
import json
import traceback
from azureml.core import Dataset, Datastore, Experiment, Workspace
from azureml.core.run import Run

# Import created modules
from utils import EnvVariables, get_model


def main():
    print("Running evaluate_model_step.py")

    # Get AML run context
    run = Run.get_context()
    
    # If script is run locally during development
    if (run.id.startswith("OfflineRun")):
        
        # Load environment variables
        env_variables = EnvVariables()

        workspace_name = env_variables.workspace_name
        experiment_name = env_variables.experiment_name
        resource_group = env_variables.resource_group
        subscription_id = env_variables.subscription_id
    
        # Manually specify run_id (useful to query previous runs)
        run_id = "1338leet-5ae8-441c-bc0c-d4c371f32d70"
    
        # Retrieve workspace and create experiment using specified env_variables
        ws = Workspace.get(name=workspace_name,
                           subscription_id=subscription_id,
                           resource_group=resource_group)
        exp = Experiment(ws, experiment_name)
    
    # if script is run remotely (online)
    else:
        # Retrieve workspace and experiment from run
        ws = run.experiment.workspace
        exp = run.experiment
        run_id = run.parent.id
    
    # Parse input arguments
    parser = argparse.ArgumentParser("evaluate")
    parser.add_argument("--model_name",
                        type=str,
                        help="Name of the AML Model",
                        default="dog-classification-model")
    parser.add_argument("--allow_run_cancel",
                        type=str,
                        help=("Set this to false to avoid evaluation step from cancelling run\
                              after an unsuccessful evaluation"),  # NOQA: E501
                        default="true")
    args = parser.parse_args()

    model_name = args.model_name
    allow_run_cancel = args.allow_run_cancel

    print("Getting evaluation parameters")
   # Load the evaluation parameters from the parameters file
    with open("config/pipeline_parameters.json") as f:
        pars = json.load(f)
    try:
        evaluation_args = pars["evaluation"]
    except KeyError:
        print("Could not load evaluation parameters from file.")
        evaluation_args = {"metric_eval": "none"}

    # Retrieve evaluation metric
    metric_eval =  evaluation_args["metric_eval"]
    
    # Parameterize the matrices on which the models should be compared
    # Add golden data set on which all the model performance can be evaluated
    try:
        tag_name = "experiment_name"
    
        # Retrieve model
        model = get_model(model_name=model_name,
                          tag_name=tag_name,
                          tag_value=exp.name,
                          aml_workspace=ws)
    
        if (model is not None):
            # Initialize production model accuracy
            production_model_accuracy = 50
            # Get production model accuracy from model tags
            if (metric_eval in model.tags):
                production_model_accuracy = float(model.tags[metric_eval])
                print(run.parent.get_metrics().get(metric_eval))
            # Get new model accuracy
            new_model_accuracy = float(run.parent.get_metrics().get(metric_eval))
            if (production_model_accuracy is None or new_model_accuracy is None):
                print("Unable to find", metric_eval, "metrics, "
                      "exiting evaluation")
                if((allow_run_cancel).lower() == "true"):
                    run.parent.cancel()
            else:
                print(
                    "Current Production model accuracy: {}, "
                    "New trained model accuracy: {}".format(
                        production_model_accuracy, new_model_accuracy
                    )
                )
            # Continue with the run only if accuracy is better than production model accuracy
            if (new_model_accuracy > production_model_accuracy):
                print("New trained model performs better, "
                      "thus it should be registered")
            else:
                print("New trained model metric is worse than or equal to "
                      "production model so skipping model registration.")
                if((allow_run_cancel).lower() == 'true'):
                    run.parent.cancel()
        # Continue with the run if no model is registered yet
        else:
            print("This is the first model, "
                  "thus it should be registered")
    
    except Exception:
        traceback.print_exc(limit=None, file=None, chain=True)
        print("Something went wrong trying to evaluate. Exiting.")
        raise

    run.complete()
    
    
if __name__ == "__main__":
    main()