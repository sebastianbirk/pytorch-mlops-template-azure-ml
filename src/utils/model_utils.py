# Import libraries
import os
import torch
import traceback
from azureml.core import Run
from azureml.core import Dataset, Workspace
from azureml.core.model import Model as AMLModel


def get_current_workspace() -> Workspace:
    """
    Retrieves and returns the current workspace.
    Will not work when ran locally.
    Parameters:
    None
    Return:
    The current workspace.
    """
    run = Run.get_context(allow_offline=False)
    experiment = run.experiment
    return experiment.workspace


def get_model(
    model_name: str,
    model_version: int = None,  # If none, return latest model
    tag_name: str = None,
    tag_value: str = None,
    aml_workspace: Workspace = None
) -> AMLModel:
    """
    Retrieves and returns a model from the workspace by its name
    and (optional) tag.
    Parameters:
    aml_workspace (Workspace): aml.core Workspace that the model lives.
    model_name (str): name of the model we are looking for
    (optional) model_version (str): model version. Latest if not provided.
    (optional) tag (str): the tag value & name the model was registered under.
    Return:
    A single aml model from the workspace that matches the name and tag, or
    None.
    """
    if aml_workspace is None:
        print("No workspace defined - using current experiment workspace.")
        aml_workspace = get_current_workspace()

    # fix the tags section
    tags = None
    if tag_name is not None or tag_value is not None:
        # Both a name and value must be specified to use tags.
        if tag_name is None or tag_value is None:
            raise ValueError(
                "model_tag_name and model_tag_value should both be supplied"
                + "or excluded"  # NOQA: E501
            )
        tags = [[tag_name, tag_value]]

    model = None
    if model_version is not None:
        # TODO(tcare): Finding a specific version currently expects exceptions
        # to propagate in the case we can't find the model. This call may
        # result in a WebserviceException that may or may not be due to the
        # model not existing.
        model = AMLModel(
            aml_workspace,
            name=model_name,
            version=model_version)
    else:
        models = AMLModel.list(
            aml_workspace, name=model_name, latest=True)
        if len(models) == 1:
            model = models[0]
        elif len(models) > 1:
            raise Exception("Expected only one model")

    return model


def get_model_metrics(model, dataloaders, dataset_sizes):
    """
    Calculate the model metrics for the trained (tuned) model.
    In this case, the test accuracy will be calculated and returned.
    :param model: the final trained (tuned) model
    :param dataloaders: a dictionary of torch dataloaders containing the test dataloader
    :param dataset_sizes: a dictionary of dataset sizes containing the test dataset size
    :return metrics: a dictionary containing all model metrics (here test acc)
    """

    # Leverage GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    running_correct_preds = 0

    for inputs, labels in dataloaders["test"]: # this should be test data?
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_correct_preds += torch.sum(preds == labels.data)

    test_acc = running_correct_preds.double() / dataset_sizes["test"]

    metrics = {"test_acc": test_acc.item()}

    return metrics


def model_already_registered(model_name, exp, run_id):
    model_list = AMLModel.list(exp.workspace, name=model_name, run_id=run_id)
    if len(model_list) >= 1:
        e = ("Model name:", model_name, "in workspace",
             exp.workspace, "with run_id ", run_id, "is already registered.")
        print(e)
        raise Exception(e)
    else:
        print("Model is not registered for this run.")


def register_aml_model(
    model_path,
    model_name,
    model_tags,
    model_properties,
    exp,
    run_id,
    dataset_id,
    build_id: str = 'none',
    build_uri=None
):
    try:
        tagsValue = {"run_id": run_id,
                     "experiment_name": exp.name}
        tagsValue.update(model_tags)
        if (build_id != "none"):
            model_already_registered(model_name, exp, run_id)
            tagsValue["BuildId"] = build_id
            if (build_uri is not None):
                tagsValue["BuildUri"] = build_uri

        model = AMLModel.register(
            workspace=exp.workspace,
            model_name=model_name,
            model_path=model_path,
            tags=tagsValue,
            properties=model_properties,
            model_framework=AMLModel.Framework.PYTORCH,
            model_framework_version=torch.__version__,
            datasets=[("training data",
                       Dataset.get_by_id(exp.workspace, dataset_id))])
        os.chdir("..")
        print(
            "Model registered: {} \nModel Description: {} "
            "\nModel Version: {}".format(
                model.name, model.description, model.version
            )
        )
    except Exception:
        traceback.print_exc(limit=None, file=None, chain=True)
        print("Model registration failed")
        raise