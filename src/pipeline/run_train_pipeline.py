"""
This script is based on the pipeline scripts in https://github.com/microsoft/MLOpsPython.
"""

# Append src (root) folder to sys.path to be able to import all necessary modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import libraries
import argparse
from azureml.core import Experiment, Workspace
from azureml.pipeline.core import PublishedPipeline

# Import created modules
from utils import EnvVariables


def main():

    parser = argparse.ArgumentParser("register")
    parser.add_argument("--output_pipeline_id_file",
                        type=str,
                        default="pipeline_id.txt",
                        help="Name of a file to write pipeline ID to")
    parser.add_argument("--skip_train_execution",
                        action="store_true",
                        help=("Do not trigger the execution. "
                              "Use this in Azure DevOps when using a server job to trigger"))
    args = parser.parse_args()

    # Load environment variables
    env_variables = EnvVariables()

    # Get Azure Machine Learning workspace
    ws = Workspace.get(name=env_variables.workspace_name,
                       subscription_id=env_variables.subscription_id,
                       resource_group=env_variables.resource_group)

    # Find the pipeline that was published by the specified build ID
    pipelines = PublishedPipeline.list(ws)
    matched_pipes = []

    for p in pipelines:
        if p.name == env_variables.pipeline_name:
            if p.version == env_variables.build_id:
                matched_pipes.append(p)

    if (len(matched_pipes) > 1):
        published_pipeline = None
        raise Exception(f"Multiple active pipelines are published for build {env_variables.build_id}.")  # NOQA: E501
    elif (len(matched_pipes) == 0):
        published_pipeline = None
        raise KeyError(f"Unable to find a published pipeline for this build {env_variables.build_id}")  # NOQA: E501
    else:
        published_pipeline = matched_pipes[0]
        print("Published pipeline id is", published_pipeline.id)

        # Save the Pipeline ID for other ADO jobs after script is complete
        if args.output_pipeline_id_file is not None:
            with open(args.output_pipeline_id_file, "w") as out_file:
                out_file.write(published_pipeline.id)

        if (args.skip_train_execution is False):
            pipeline_parameters = {"model_name": env_variables.model_name}
            tags = {"BuildId": env_variables.build_id}
            if (env_variables.build_uri is not None):
                tags["BuildUri"] = env_variables.build_uri

            experiment = Experiment(workspace=ws,
                                    name=env_variables.experiment_name)
            
            run = experiment.submit(published_pipeline,
                                    tags=tags,
                                    pipeline_parameters=pipeline_parameters)

            print(f"Pipeline run with run ID {run.id} initiated.")


if __name__ == "__main__":
    main()