from azureml.core import Dataset, Datastore, Workspace

def get_dataset(
    workspace: Workspace,
    dataset_name: str,
    target_path: str = "fowl-classification-example",
    download: bool = False
) -> Dataset:
    """
    Retrieve dataset from Azure ML workspace
    """
    try:
        if dataset_name not in workspace.datasets:
            raise Exception(f"Dataset with name {dataset_name} does not exist in the workspace")
        else:
            dataset = Dataset.get_by_name(workspace, name=dataset_name)
            if download:
                dataset.download(target_path=target_path, overwrite=True)

        return dataset
        
    except Exception as e:
        print(e)
        exit(1)
