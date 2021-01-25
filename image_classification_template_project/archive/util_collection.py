import pickle

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
        
        
def download_and_extract_archive(download_url, file_dir, archive_file_name, skip_if_dir_exists=False, force_dir_deletion=False):
    """
    Download and extract a given archive
    :param download_url: url from where to download
    :param file_dir: root directory to which to download
    :param archive_file_name: name of the archive
    :param skip_if_dir_exists: if set to True, skip the download if the directory already exists
    :param force_dir_deletion: if set to True, delete the existing directory before the download
    """
    
    # Download the archive
    download_file(download_url=download_url, file_dir=file_dir, file_name=archive_file_name,
                  skip_if_dir_exists=skip_if_dir_exists, force_dir_deletion=force_dir_deletion)
    
    # Extract the archive
    extract_archive(from_path=file_path, remove_finished=True)
        
                                     
def unpickle_file(file_path, encoding="bytes"):
    with open(file_path, "rb") as file:
        unpickled_object = pickle.load(file, encoding=encoding)
    return unpickled_object