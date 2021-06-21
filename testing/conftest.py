# Import libraries
import pytest

# Import created modules
from src.utils.data_utils import load_data, load_unnormalized_train_data

# Define test fixtures
@pytest.fixture
def get_unnormalized_dataloader():
    unnormalized_dataloader, _, _ = load_unnormalized_train_data("data")
    return unnormalized_dataloader

@pytest.fixture
def get_dataloaders():
    dataloaders, _, _ = load_data("data")
    return dataloaders
