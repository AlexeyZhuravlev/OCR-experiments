"""
Basic dataset classes for storing image bases for OCR
Datasets return dict {"image": image, "string": string}
"""

import random
import six
import lmdb
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch.nn import functional as F
from PIL import Image
import numpy as np

class DataItemKeys:
    """
    Keys, returned in dict by all dataset classes, to avoid misprints
    """
    IMAGE = "image"
    IMAGE_WIDTH = "image_width"
    STRING = "string"

class LmdbDataset(Dataset):
    """
    Common lmdb-format dataset for storing image bases for OCR
    Seeks all images by keys image-XXXXXXXXX and their text labels by label-XXXXXXXXX
    Key num-samples should contain total image count
    """

    def __init__(self, root, image_format="RGB", case_sensitive=False):

        self.root = root
        self.image_format = image_format
        self.case_sensitive = case_sensitive
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False,
                             readahead=False, meminit=False)

        if not self.env:
            raise RuntimeError("cannot create lmdb from {}".format(root))

        with self.env.begin(write=False) as txn:
            self.n_samples = int(txn.get("num-samples".encode()))

    def get_all_symbols(self):
        """
        Returns set of all symbols contained in dataset labels
        """
        all_symbols = set()
        with self.env.begin(write=False) as txn:
            for index in range(self.n_samples):
                index += 1
                label_key = 'label-%09d'.encode() % index
                label = txn.get(label_key).decode('utf-8')

                all_symbols |= set(label)

        return all_symbols

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        # lmbd starts indexing from 1
        index += 1

        with self.env.begin(write=False) as txn:
            label_key = "label-{:09d}".format(index).encode()
            label = txn.get(label_key).decode("utf-8")
            img_key = "image-{:09d}".format(index).encode()
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert(self.image_format)
            except IOError:
                raise RuntimeError("Corrupted image for {}".format(index))

        if not self.case_sensitive:
            label = label.lower()

        item = {
            DataItemKeys.IMAGE: np.array(img),
            DataItemKeys.STRING: label
        }

        return item

class PaddedDatasetWithTransforms(Dataset):
    """
    Wraps any existing OCR dataset and image transforms function to apply when getting elements
    Also padds smaller images to pad_width
    """
    def __init__(self, dataset, transforms, pad_width):
        self.dataset = dataset
        self.tranforms = transforms
        self.pad_width = pad_width

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        image = self.tranforms(image=item[DataItemKeys.IMAGE])["image"]

        width = image.shape[-1]
        padding_right = self.pad_width - width
        image = F.pad(image, (0, padding_right))

        item[DataItemKeys.IMAGE] = image
        item[DataItemKeys.IMAGE_WIDTH] = width

        return item
