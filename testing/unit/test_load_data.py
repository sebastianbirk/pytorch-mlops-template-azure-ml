# Import libraries
import pytest

# Import created modules
from src.utils.data_utils import load_data

# Define unit test
@pytest.mark.unit
def test_load_data():
    dataloaders, dataset_sizes, class_names = load_data("data")
    assert type(dataloaders) == dict
    assert type(dataset_sizes) == dict
    assert type(class_names) == list
