import argparse
import os
import shutil
from aml.utils.env_variables import EnvVariables
from azureml.core import Workspace
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model 

# Load environment variables
env_variables = EnvVariables()

# Get Azure Machine Learning workspace
ws = Workspace.get(
    name=env_variables.workspace_name,
    subscription_id=env_variables.subscription_id,
    resource_group=env_variables.resource_group
)

# Parse output_image_location_file parameter
parser = argparse.ArgumentParser("Create scoring image")
parser.add_argument(
    "--output_image_location_file",
    type=str,
    help=("Name of a file to write image location to, "
          "in format REGISTRY.azurecr.io/IMAGE_NAME:IMAGE_VERSION")
)
args = parser.parse_args()

# Retrieve model
model = Model(ws, name=env_variables.model_name, version=env_variables.model_version)


sources_dir = env_variables.sources_directory_train
if (sources_dir is None):
    sources_dir = "src"
score_script = os.path.join(".", sources_dir, env_variables.score_script)
score_file = os.path.basename(score_script)
path_to_scoring = os.path.dirname(score_script)
cwd = os.getcwd()
# Copy conda_dependencies.yml into scoring as this method does not accept relative paths. # NOQA: E501
shutil.copy(os.path.join(".", sources_dir,
                         "conda_dependencies.yml"), path_to_scoring)
os.chdir(path_to_scoring)

# Create environment from yml file
scoring_env = Environment.from_conda_specification(
    name="scoringenv",
    file_path="conda_dependencies.yml"
)  # NOQA: E501

# Create inference config using entry script and environment
inference_config = InferenceConfig(
    entry_script=score_file, environment=scoring_env
)

package = Model.package(ws, [model], inference_config)

package.wait_for_creation(show_output=True)
# Display the package location/ACR path
print(package.location)

os.chdir(cwd)

if package.state != "Succeeded":
    raise Exception("Image creation status: {package.creation_state}")

print("Package stored at {} with build log {}".format(package.location, package.package_build_log_uri))  # NOQA: E501

# Save the Image Location for other AzDO jobs after script is complete
if args.output_image_location_file is not None:
    print("Writing image location to %s" % args.output_image_location_file)
    with open(args.output_image_location_file, "w") as out_file:
        out_file.write(str(package.location))
