# Import libraries
import numpy as np
import os
import pytest
import torchvision

# Import created modules
from src.utils.data_utils import load_data, imshow

# Define unit tests
@pytest.mark.unit
def test_load_data():
    dataloaders, dataset_sizes, class_names = load_data("data")
    assert type(dataloaders) == dict
    assert type(dataset_sizes) == dict
    assert type(class_names) == list
    
@pytest.mark.unit
def test_view_data():
    dataloaders, _, _ = load_data("data")
    dataiter = iter(dataloaders["val"])
    images, _ = dataiter.next()
    imshow(torchvision.utils.make_grid(images))