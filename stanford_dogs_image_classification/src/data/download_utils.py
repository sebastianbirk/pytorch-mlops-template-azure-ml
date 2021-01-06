"""
Definition of utility functions for dataset archive download and extraction
Adjusted from https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py
"""

import os
import shutil
import urllib.request
import tarfile
import zipfile
import gzip
import tqdm


def gen_bar_updater():
    """tqdm report hook for urlretrieve"""
    pbar = tqdm.tqdm(total=None)

    # Define progress bar update function
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def download_url(url, root_path, file_name):
    """
    Download a file with given name from a given url to a given directory
    :param url: url from where to download
    :param root_path: root directory to which to download
    :param file_name: name under which the file should be saved
    """

    # Determine file path and create root_path directory if necessary
    file_path = os.path.join(root_path, file_name)
    os.makedirs(root_path, exist_ok=True)

    # Download the file if necessary using a progress bar
    if not os.path.exists(file_path):
        print("Downloading " + url + " to " + file_path)
        urllib.request.urlretrieve(
            url,
            file_path,
            reporthook=gen_bar_updater()
        )

    return file_path


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
        return

    print(f"Extracting archive {from_path}")

    # If no to_path is indicated, use parent directory of from_path as to_path
    if to_path is None:
        to_path = os.path.dirname(from_path)

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
        os.remove(from_path)


def download_and_extract_dataset(
    image_archive_url, image_archive_name=None,
    annotation_archive_url, annotation_archive_name=None,
    dataset_dir, force_download=False):
    """
    Download and extract image and annotation archive files
    :param image_archive_url: URL to download image archive file from
    :param annotation_archive_url: URL to download annotation archive file from
    :param dataset_dir: name of dataset directory where data will be stored
    :param image_archive_name: name of downloaded compressed image archive file
    :param annotation_archive_name: name of downloaded compressed annotation
        archive file
    :param force_download: if set to True, always download dataset
        (even if it already exists)
    """

    # Check whether download is necessary
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir)
    or force_download:

        # Remove directory if it exists
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)

        # Download and extract image archive and rename folders
        if image_archive_name is not None:
            image_archive_file_path = download_url(image_archive_url,
                                                   dataset_dir,
                                                   image_archive_name)
            extract_archive(image_archive_file_path, remove_finished=True)
            shutil.move(os.path.join(dataset_dir, "Images"),
                        os.path.join(dataset_dir, "images"))

        if label_archive_name is not None:
            label_archive_file_path = download_url(label_archive_url,
                                                   dataset_dir,
                                                   label_archive_name)
            extract_archive(label_archive_file_path, remove_finished=True)
            shutil.move(os.path.join(dataset_dir, "Annotations"),
                        os.path.join(dataset_dir, "annotations"))

        
