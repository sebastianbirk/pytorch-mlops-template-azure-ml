import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transforms

from torchvision import datasets
from typing import Tuple

def load_data(data_dir: str) -> Tuple[dict, dict, list]:
    """
    Load the train, val and test data.
    :param data_dir: path where the images are stored
    :return (image_dataloaders, dataset_sizes, class_names)
        image_dataloaders: dictionary of train, val, and test Pytorch dataloaders
        dataset_sizes: dictionary of train, val and test Pytorch dataset lengths
        class_names: list of all classes
    """

    # Data augmentation and normalization for training set
    # Just normalization for validation and test set
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }
    
    # Dictionary of image datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ["train", "val", "test"]}
    
    # Dictionary of image dataloaders
    image_dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                        shuffle=True, num_workers=2) 
                         for x in ["train", "val", "test"]}
    
    # Dictionary of dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}
    
    # List of class names
    class_names = image_datasets["train"].classes
    
    return image_dataloaders, dataset_sizes, class_names


def show_image(image_path: str) -> None:
    """
    Load and show an example image
    :param image_path: path of the image to be loaded
    """
    # Read in example image
    img = mpimg.imread(image_path)

    # Check format of image
    print(f"Image shape: {img.shape}")

    # Show example image
    imgplot = plt.imshow(img)
