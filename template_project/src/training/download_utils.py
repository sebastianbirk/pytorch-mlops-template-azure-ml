import os
import pickle
import urllib
import tarfile
import tqdm


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


def _is_tarxz(file_name):
    return file_name.endswith(".tar.xz")


def _is_tar(file_name):
    return file_name.endswith(".tar")


def _is_targz(file_name):
    return file_name.endswith(".tar.gz")


def _is_tgz(file_name):
    return file_name.endswith(".tgz")


def _is_gzip(file_name):
    return file_name.endswith(".gz") and not file_name.endswith(".tar.gz")


def _is_zip(file_name):
    return file_name.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=False):
    """
    Extract a given archive
    :param from_path: path to archive which should be extracted
    :param to_path: path to which archive should be extracted
        default: parent directory of from_path
    :param remove_finished: if set to True, delete archive after extraction
    """

    # If archive does not exist, do nothing
    if not os.path.exists(from_path):
        print(f"There is no archive {from_path}")
        return

    # If no to_path is indicated, use parent directory of from_path as to_path
    if to_path is None:
        to_path = os.path.dirname(from_path)
        
    print(f"Extracting archive {from_path} to {to_path}")

    # Check for file extension and extract the archive
    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(
            to_path,
            os.path.splitext(os.path.basename(from_path))[0]
        )
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as zip_:
            zip_.extractall(to_path)
    else:
        raise ValueError(f"Extraction of {from_path} not supported")

    # Remove archive if flag is True
    if remove_finished:
        print(f"Removing archive file {from_path}")
        os.remove(from_path)


def download_and_extract_archive(download_url, file_dir, archive_file_name, force_download=False):
    """
    Download and extract a given archive
    :param download_url: url from where to download
    :param file_dir: root directory to which to download
    :param archive_file_name: name of the archive
    :param force_download: if set to True, always download dataset
    """
    
    # Check if download should be triggered
    if not os.path.exists(file_dir) or force_download:
        
        # Remove file directory if it exists
        if os.path.exists(file_dir):
            shutil.rmtree(file_dir)
    
        # Create file directory if it does not exist
        os.makedirs(file_dir, exist_ok=True)
    
        # Download the archive
        file_path = os.path.join(file_dir, archive_file_name)
        print("Downloading " + download_url + " to " + file_path)
        urllib.request.urlretrieve(download_url, filename=file_path, reporthook=generate_bar_updater())
    
        # Extract the archive
        extract_archive(from_path=file_path, remove_finished=True)
        
        
def unpickle_file(file_path):
    with open(file_path, "rb") as file:
        dict = pickle.load(file, encoding="bytes")
    return dict
