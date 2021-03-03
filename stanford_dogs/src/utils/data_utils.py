# Import libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import scipy.io
import shutil
import tarfile
import torch
import torchvision.transforms as transforms
import tqdm
import urllib
from pathlib import Path
from PIL import Image
from torchvision import datasets
from typing import Tuple


def download_file(download_url: str,
                  file_dir: str,
                  file_name: str,
                  skip_if_dir_exists: bool = False,
                  force_dir_deletion: bool = False) -> None:
    """
    Download a file
    :param download_url: url from where to download
    :param file_dir: directory to which to download
    :param file_name: name of the file
    :param force_dir_deletion: flag that indicates whether to delete the existing directory before the download
    """
    
    # Remove file directory if it exists
    if force_dir_deletion and os.path.exists(file_dir):
        shutil.rmtree(file_dir)
        print(f"Directory {file_dir} has been removed.")
    
    # Check if download should be triggered
    if not os.path.exists(file_dir):
    
        # Create file directory if it does not exist
        os.makedirs(file_dir, exist_ok=True)
    
        # Download the file
        file_path = os.path.join(file_dir, file_name)
        print("Downloading " + download_url + " to " + file_path + ".")
        urllib.request.urlretrieve(download_url, filename=file_path, reporthook=generate_bar_updater())


def download_stanford_dogs_archives():
    """
    Download the two stanford dogs archives:
    - images.tar ("http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar")
    - lists.tar ("http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar")
    """
    
    # Define archive files to be downloaded from the stanford vision website
    archive_file_list = ["images.tar", "lists.tar"]

    # Download archive files
    for i, archive_file in enumerate(archive_file_list):
        download_file(download_url="http://vision.stanford.edu/aditya86/ImageNetDogs/" + archive_file,
                      file_dir=os.path.join(Path(__file__).resolve().parents[2],
                                            "data/archives",
                                            archive_file.split(".")[0]),
                      file_name=archive_file,
                      force_dir_deletion=True)        


def extract_stanford_dogs_archives(archive_dir: str = os.path.join(Path(__file__).resolve().parents[2], "data/archives"),
                                   target_dir: str = os.path.join(Path(__file__).resolve().parents[2], "data"),
                                   remove_archives: bool = True) -> None:
    """
    Extract the stanford dogs image archive and separate the images into training,
    validation and test set
    :param archive_dir: parent path of the archive files to be extracted
    :param target_dir: path of the target directory where the files should be extracted to
    :param remove_archives: flag that indicates whether the archives are removed after extraction
    """
 
    # Specify directory paths
    training_dir = os.path.join(target_dir, "train")
    validation_dir = os.path.join(target_dir, "val")
    test_dir = os.path.join(target_dir, "test")    
    list_dir = os.path.join(archive_dir, "lists")
    images_dir = os.path.join(archive_dir, "images")
    
    # Remove directories if they exist
    for directory in [training_dir, validation_dir, test_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Directory {directory} has been removed.")

    # Extract lists.tar archive
    with tarfile.open(os.path.join(list_dir, "lists.tar"), "r") as lists_tar:
        lists_tar.extractall(path=target_dir)
                             
    print("Lists.tar archive has been extracted successfully.")
    
    # Load list.mat files
    train_list_mat = scipy.io.loadmat(os.path.join(target_dir, "train_list.mat"))
    test_list_mat = scipy.io.loadmat(os.path.join(target_dir, "test_list.mat"))
    
    training_and_val_files = []
    test_files = []
    
    # Extract training data file names
    for array in train_list_mat["file_list"]:
        training_and_val_files.append(array[0][0])

    # Extract test data file names
    for array in test_list_mat["file_list"]:
        test_files.append(array[0][0])
                             
    print("File lists have been read successfully.")
    print("Extracting images.tar archive...")
                             
    # Extract images.tar archive
    with tarfile.open(os.path.join(images_dir, "images.tar"), "r") as images_tar:
        training_val_idx = 0
        for member in tqdm.tqdm(images_tar.getmembers()):
            if member.isreg(): # Skip if TarInfo is not files
                member.name = member.name.split("/", 1)[1] # Retrieve only relevant part of file name
                
                # Extract files to corresponding directories
                if member.name in training_and_val_files: # Every 5th file goes to the validation data
                    training_val_idx+=1
                    if training_val_idx % 5 != 0:
                        images_tar.extract(member, training_dir)
                    else:
                        images_tar.extract(member, validation_dir)
                    
                elif member.name in test_files:
                    images_tar.extract(member, test_dir)
                             
    print("Images.tar archive has been extracted successfully.")

    # Remove list.mat files
    os.remove(os.path.join(target_dir, "file_list.mat"))
    os.remove(os.path.join(target_dir, "test_list.mat"))
    os.remove(os.path.join(target_dir, "train_list.mat"))
    
    # Remove archive files if flag is set to true
    if remove_archives:
        print("Removing archive files.")
        os.remove(os.path.join(list_dir, "lists.tar"))
        os.remove(os.path.join(images_dir, "images.tar"))


def generate_bar_updater():
    """
    Create a tqdm reporthook function for urlretrieve
    :returns: bar_update function which can be used by urlretrieve 
              to display and update a progress bar
    """
    
    pbar = tqdm.tqdm(total=None)

    # Define progress bar update function
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def load_data(data_dir: str) -> Tuple[dict, dict, list]:
    """
    Load the train, val and test data.
    :param data_dir: path where the images are stored
    :return (dataloaders, dataset_sizes, class_names)
        dataloaders: dictionary of train, val, and test torch dataloaders
        dataset_sizes: dictionary of train, val and test torch dataset lengths
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
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=2) 
                         for x in ["train", "val", "test"]}
    
    # Dictionary of dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}
    
    # List of class names
    class_names = image_datasets["train"].classes
    
    return dataloaders, dataset_sizes, class_names


def preprocess_image(image_file):
    """
    Preprocess an input image.
    :param image_file: Path to the input image
    :return image.numpy(): preprocessed image as numpy array
    """
    
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = Image.open(image_file)
    image = data_transforms(image).float()
    image = image.clone().detach()
    image = image.unsqueeze(0)
    
    return image.numpy()


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


def show_batch_of_images(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # transpose dimensions from Pytorch format to default numpy format
    plt.show()
