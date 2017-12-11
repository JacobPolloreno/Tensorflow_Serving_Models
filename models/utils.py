import os
import tarfile

from tqdm import tqdm
from urllib.request import urlretrieve


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def download_dataset_and_uncompress(dataset_dir: str,
                                    url: str,
                                    filename: str=None):
    """Downloads and uncompresses dataset from url, expects tar.gz file

    """
    filename = filename or url.split('/')[-1]

    if not os.path.isfile(filename):
        with DLProgress(unit='B',
                        unit_scale=True,
                        miniters=1,
                        desc='download dataset') as pbar:
            urlretrieve(
                url,
                filename,
                pbar.hook)

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(dataset_dir)
        tar.close()

    statinfo = os.stat(filename)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')


def create_checkpoints_dir(checkpoints_dir: str='checkpoints/'):
    """Creates the checkpoints directory if it does not exist

   """
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
