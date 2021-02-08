import argparse
from aml.utils import EnvVariables
from azureml.pipeline.core import PublishedPipeline
from azureml.core import Experiment, Workspace


def main():

    # Parse script input arguments
    parser = argparse.ArgumentParser("Run pipeline")
    parser.add_argument(
        "--output_pipeline_id_file",
        type=str,
        default="pipeline_id.txt",
        help="Name of a file to write pipeline id to")
    parser.add_argument(
        "--skip_train_execution",
        action="store_true", # set True as default value
        help=("Do not trigger the execution. "
              "Use this in Azure DevOps when using a server job to trigger"))
    args = parser.parse_args()

    # Load environment variables
    env_variables = EnvVariables()

    # Retrieve workspace
    ws = Workspace.get(
        name=env_variables.workspace_name,
        subscription_id=env_variables.subscription_id,
        resource_group=env_variables.resource_group)

    # Find the pipeline that was published by the specified build ID
    pipelines = PublishedPipeline.list(ws)
    matched_pipes = []

    for p in pipelines:
        if p.name == env_variables.pipeline_name:
            if p.version == env_variables.build_id:
                matched_pipes.append(p)

    # Error handling if more than 1 or no pipeline is found
    if(len(matched_pipes) > 1):
        published_pipeline = None
        raise Exception(f"Multiple active pipelines are published for build {env_variables.build_id}.")  # NOQA: E501
    elif(len(matched_pipes) == 0):
        published_pipeline = None
        raise KeyError(f"Unable to find a published pipeline for this build {env_variables.build_id}")  # NOQA: E501
    else:
        published_pipeline = matched_pipes[0]
        print(f"Published pipeline id is {published_pipeline.id}")

        # Save the Pipeline ID for other ADO jobs after script is complete
        if args.output_pipeline_id_file is not None:
            with open(args.output_pipeline_id_file, "w") as out_file:
                out_file.write(published_pipeline.id)

        # Run the pipeline if not skip_train_execution
        if (args.skip_train_execution is False):
            # Configure pipeline parameters and tags
            pipeline_parameters = {"model_name": env_variables.model_name}
            tags = {"BuildId": env_variables.build_id}
            if (env_variables.build_uri is not None):
                tags["BuildUri"] = env_variables.build_uri

    	    # Create experiment for pipeline run
            experiment = Experiment(
                workspace=ws,
                name=env_variables.experiment_name)

            # Submit pipeline run
            run = experiment.submit(
                published_pipeline,
                tags=tags,
                pipeline_parameters=pipeline_parameters)

            print(f"Pipeline run with run id {run.id} initiated")


if __name__ == "__main__":
    main()
