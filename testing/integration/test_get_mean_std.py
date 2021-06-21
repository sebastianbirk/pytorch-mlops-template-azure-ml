# Import libraries
import pytest
import torch

# Import created modules
from src.utils.data_utils import get_mean_std

# # Import created modules
# from src.utils.data_utils import get_mean_std, load_unnormalized_train_data

# # Define test fixtures
# @pytest.fixture
# def get_unnormalized_dataloader():
#     unnormalized_dataloader, _, _ = load_unnormalized_train_data("data")
#     return unnormalized_dataloader

# Define integration tests
@pytest.mark.integration
def test_get_mean_std(get_unnormalized_dataloader):
    train_data_mean, train_data_std = get_mean_std(get_unnormalized_dataloader)
    assert type(train_data_mean) == torch.Tensor
    assert type(train_data_std) ==  torch.Tensor