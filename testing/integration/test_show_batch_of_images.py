# Import libraries
import pytest
import torchvision

# Import created modules
from src.utils.data_utils import load_data, show_batch_of_images    

# Define integration test
@pytest.mark.integration
def test_show_batch_of_images(get_dataloaders):
    dataiter = iter(get_dataloaders["val"])
    images, _ = dataiter.next()
    show_batch_of_images(torchvision.utils.make_grid(images))

# # Define integration test
# @pytest.mark.integration
# def test_show_batch_of_images():
#     dataloaders, _, _ = load_data("data")
#     dataiter = iter(dataloaders["val"])
#     images, _ = dataiter.next()
#     show_batch_of_images(torchvision.utils.make_grid(images))