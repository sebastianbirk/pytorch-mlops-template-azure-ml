"""Definition of dataset base class"""

from abc import ABC, abstractmethod
from .download_utils import download_dataset


class Dataset(ABC):
    """
    Abstract Dataset Base Class
    All subclasses must define __getitem__() and __len__()
    """

    def __init__(self, root_path, image_download_url=None,
                 label_download_url=None, force_download=False):

        self.root_path = root_path

        # The actual archive name should be all the text of the url after the
        # last '/'.
        if image_download_url is not None:
            image_archive_name = image_download_url[image_download_url.rfind('/')+1:]
            self.image_archive_name = image_archive_name

        if label_download_url is not None:
            label_archive_name = label_download_url[label_download_url.rfind('/')+1:]
            self.label_archive_name = label_archive_name

            download_dataset(
                image_archive_url=image_download_url,
                image_archive_name=image_archive_name,
                label_archive_url=label_download_url,
                label_archive_name=label_archive_name,
                dataset_dir=root_path,
                force_download=force_download,
            )
  

    @abstractmethod
    def __getitem__(self, index):
        """Return data sample at given index"""

    @abstractmethod
    def __len__(self):
        """Return size of the dataset"""
