"""Definition of dataset classes and image-specific transform classes"""

from .stanford_dogs_dataset import (
    StanfordDogsDataset,
    RescaleTransform,
    NormalizeTransform,
    FlattenTransform,
    ComposeTransform,
    compute_image_mean_and_std,
)
