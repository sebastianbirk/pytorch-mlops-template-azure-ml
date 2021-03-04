"""
This script is based on the pipeline scripts in https://github.com/microsoft/MLOpsPython. 
"""

# Append src folder to sys.path to be able to import all necessary modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import libraries
import argparse
import traceback
from azureml.core import Run

# Import created modules
from utils import EnvVariables, get_model

def main():
    print("Running evaluate_model_step.py")

    run = Run.get_context()
    
    # If script is run locally during development
    if (run.id.startswith("OfflineRun")):
        
        # Load environment variables
        env_variables = EnvVariables()
    
        sources_dir = env_variables.src_dir
        if (sources_dir is None):
            sources_dir = "src"
    
        workspace_name = env_variables.workspace_name
        experiment_name = env_variables.experiment_name
        resource_group = env_variables.resource_group
        subscription_id = env_variables.subscription_id
        tenant_id = env_variables.tenant_id
        model_name = env_variables.model_name
        app_id = env_variables_sp_app_id
        app_secret = env_variables.sp_app_secret
        build_id = env_variables.build_buildid
    
        # Manually specify run_id (useful to query previous runs)
        run_id = "57fee47f-5ae8-441c-bc0c-d4c371f32d70"
    
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
    
    parser = argparse.ArgumentParser("evaluate")
    
    parser.add_argument("--run_id",
                        type=str,
                        help="Training run ID")
    
    parser.add_argument("--model_name",
                        type=str,
                        help="Name of the Model",
                        default="fowl-model")
    
    parser.add_argument("--allow_run_cancel",
                        type=str,
                        help=("Set this to false to avoid evaluation step from cancelling run\
                              after an unsuccessful evaluation"),  # NOQA: E501
                        default="true")
    
    args = parser.parse_args()
    
    if (args.run_id is not None):
        run_id = args.run_id
    
    if (run_id == "amlcompute"):
        run_id = run.parent.id
    
    model_name = args.model_name
    metric_eval = "val_accuracy"
    
    allow_run_cancel = args.allow_run_cancel
    # Parameterize the matrices on which the models should be compared
    # Add golden data set on which all the model performance can be evaluated
    try:
        firstRegistration = False
        tag_name = "experiment_name"
    
        model = get_model(model_name=model_name,
                          tag_name=tag_name,
                          tag_value=exp.name,
                          aml_workspace=ws)
    
        if (model is not None):
            production_model_accuracy = 50
            if (metric_eval in model.tags):
                production_model_accuracy = float(model.tags[metric_eval])
                print(run.parent.get_metrics().get(metric_eval))
            new_model_accuracy = float(run.parent.get_metrics().get(metric_eval))
            if (production_model_accuracy is None or new_model_accuracy is None):
                print("Unable to find", metric_eval, "metrics, "
                      "exiting evaluation")
                if((allow_run_cancel).lower() == 'true'):
                    run.parent.cancel()
            else:
                print(
                    "Current Production model accuracy: {}, "
                    "New trained model accuracy: {}".format(
                        production_model_accuracy, new_model_accuracy
                    )
                )
    
            if (new_model_accuracy > production_model_accuracy):
                print("New trained model performs better, "
                      "thus it should be registered")
            else:
                print("New trained model metric is worse than or equal to "
                      "production model so skipping model registration.")
                if((allow_run_cancel).lower() == 'true'):
                    run.parent.cancel()
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