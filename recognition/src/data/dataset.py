"""
Basic dataset classes for storing image bases for OCR
Datasets return dict {"image": image, "string": string}
"""

import random
import six
import lmdb
from torch.utils.data import Dataset, ConcatDataset, Subset
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

    def __init__(self, root, image_format="RGB"):

        self.root = root
        self.image_format = image_format
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

        item = {
            DataItemKeys.IMAGE: np.array(img),
            DataItemKeys.STRING: label
        }

        return item


class DataSubsetProvider(Dataset):
    """
    Wrapper on top of another dataset, which acts as subset of initial dataset of fixed size
    Returns another portion of data when calling next_subset
    """
    def __init__(self, dataset, subset_size):
        assert subset_size <= len(dataset)

        self.dataset = dataset
        self.subset_size = subset_size
        self.all_indices = [i for i in range(len(self.dataset))]
        random.shuffle(self.all_indices)
        self.slice_start = 0
        self.current_subset = None
        self._update_subset()

    def next_subset(self):
        """Subset update: return next portion of data"""
        self.slice_start += self.subset_size
        # End of sequence has reached: starting from the beginning
        if self.slice_start + self.subset_size > len(self.dataset):
            random.shuffle(self.all_indices)
            self.slice_start = 0
        self._update_subset()

    def _update_subset(self):
        indices = self.all_indices[self.slice_start:self.slice_start + self.subset_size]
        self.current_subset = Subset(self.dataset, indices)

    def __len__(self):
        return self.subset_size

    def __getitem__(self, index):
        return self.current_subset[index]


class HybridDataset(ConcatDataset):
    """
    Concatenation of several datasets with ability to reset all elements
    of instance DataSubsetProvider inside
    """
    def __init__(self, datasets):
        super().__init__(datasets)
        self.datasets = datasets

    def update_data(self):
        """Picks next portion of data for shiftable datasets"""
        for dataset in self.datasets:
            if isinstance(dataset, DataSubsetProvider):
                dataset.next_subset()

class DatasetWithTransforms(Dataset):
    """
    Wraps any existing OCR dataset and image transforms function to apply when getting elements
    """
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.tranforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        image = self.tranforms(image=item[DataItemKeys.IMAGE])["image"]
        item[DataItemKeys.IMAGE] = image
        item[DataItemKeys.IMAGE_WIDTH] = image.shape[2]

        return item
