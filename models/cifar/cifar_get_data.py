import sys

sys.path.append('..')

from utils import download_dataset_and_uncompress

CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


if __name__ == "__main__":
    download_dataset_and_uncompress(
        '.',
        CIFAR_URL)
