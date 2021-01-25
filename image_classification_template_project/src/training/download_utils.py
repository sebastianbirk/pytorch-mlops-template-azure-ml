import os
import scipy.io
import shutil
import tarfile
import tqdm
import urllib


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
    :param skip_if_dir_exists: flag that indicates whether to skip the download if the directory already exists
    :param force_dir_deletion: flag that indicates whether to delete the existing directory before the download
    """
    
    # Remove file directory if it exists
    if force_dir_deletion and os.path.exists(file_dir):
        shutil.rmtree(file_dir)
        print(f"Directory {file_dir} has been removed.")
    
    # Check if download should be triggered
    if not os.path.exists(file_dir) or not skip_if_dir_exists:
    
        # Create file directory if it does not exist
        os.makedirs(file_dir, exist_ok=True)
    
        # Download the file
        file_path = os.path.join(file_dir, file_name)
        print("Downloading " + download_url + " to " + file_path + ".")
        urllib.request.urlretrieve(download_url, filename=file_path, reporthook=generate_bar_updater())
        

def extract_stanford_dogs_archive(archive_dir_path: str = "../data",
                                  target_dir_path: str = "../data",
                                  remove_archives: bool = True) -> None:
    """
    Extract the stanford dogs image archive and separate the images into training,
    validation and test set
    :param archive_dir_path: path of the "image.tar" and "lists.tar" files to be extracted
    :param target_dir_path: path of the target directory where the files should be extracted to
    :param remove_archives: flag that indicates whether the archives are removed after extraction
    """
 
    # Specify directory paths
    training_dir = os.path.join(target_dir_path, "train")
    validation_dir = os.path.join(target_dir_path, "val")
    test_dir = os.path.join(target_dir_path, "test")    
    
    # Remove directories if they exist
    for directory in [training_dir, validation_dir, test_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Directory {directory} has been removed.")

    # Extract lists.tar archive
    with tarfile.open(os.path.join(archive_dir_path, "lists.tar"), "r") as lists_tar:
        lists_tar.extractall(path=archive_dir_path)
                             
    print("Lists.tar archive has been extracted successfully.")
    
    # Load list.mat files
    train_list_mat = scipy.io.loadmat(os.path.join(archive_dir_path, "train_list.mat"))
    test_list_mat = scipy.io.loadmat(os.path.join(archive_dir_path, "test_list.mat"))
    
    training_files = []
    test_and_val_files = []
    
    # Extract training data file names
    for array in train_list_mat["file_list"]:
        training_files.append(array[0][0])

    # Extract test data file names
    for array in test_list_mat["file_list"]:
        test_and_val_files.append(array[0][0])
                             
    print("File lists have been read successfully.")
    print("Extracting images.tar archive...")
                             
    # Extract images.tar archive
    with tarfile.open(os.path.join(archive_dir_path, "images.tar"), "r") as images_tar:
        test_val_idx = 0
        for member in tqdm.tqdm(images_tar.getmembers()):
            if member.isreg(): # Skip if TarInfo is not files
                member.name = member.name.split("/", 1)[1] # Retrieve only relevant part of file name
                
                # Extract files to corresponding directories
                if member.name in training_files:
                    images_tar.extract(member, training_dir)
                    
                elif member.name in test_and_val_files: # Every 2nd file goes to the validation data
                    test_val_idx+=1
                    if test_val_idx % 2 != 0:
                        images_tar.extract(member, validation_dir)
                    else:
                        images_tar.extract(member, test_dir)
                             
    print("Images.tar archive has been extracted successfully.")

    # Remove list.mat files
    os.remove(os.path.join(archive_dir_path, "file_list.mat"))
    os.remove(os.path.join(archive_dir_path, "test_list.mat"))
    os.remove(os.path.join(archive_dir_path, "train_list.mat"))
    
    # Remove archive files if flag is set to true
    if remove_archives:
        print("Removing archive files.")
        os.remove(os.path.join(archive_dir_path, "lists.tar"))
        os.remove(os.path.join(archive_dir_path, "images.tar"))

                             
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
