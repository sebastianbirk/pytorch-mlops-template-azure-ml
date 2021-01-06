"""Definition of dataset base class"""

from abc import ABC, abstractmethod
from .download_utils import download_and_extract_dataset


class Dataset(ABC):
    """
    Abstract Dataset Base Class
    All subclasses must define __getitem__() and __len__()
    """

    def __init__(self, root_path, image_download_url=None,
                 annotation_download_url=None, force_download=False):

        self.root_path = root_path

        # The actual archive name should be all the text of the url after the
        # last '/'.
        if image_download_url is not None:
            image_archive_name = image_download_url[image_download_url.rfind('/')+1:]
            self.image_archive_name = image_archive_name

        if annotation_download_url is not None:
            annotation_archive_name = annotation_download_url[annotation_download_url.rfind('/')+1:]
            self.annotation_archive_name = annotation_archive_name

            download_and_extract_dataset(
                dataset_dir=root_path,
                force_download=force_download,
                image_archive_url=image_download_url,
                image_archive_name=image_archive_name,
                annotation_archive_url=annotation_download_url,
                annotation_archive_name=annotation_archive_name
            )
  

    @abstractmethod
    def __getitem__(self, index):
        """Return data sample at given index"""

    @abstractmethod
    def __len__(self):
        """Return size of the dataset"""
