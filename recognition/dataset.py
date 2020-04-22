from torch.utils.data import Dataset
from PIL import Image
import lmdb
import six
import sys
import re

class LmdbDataset(Dataset):
    """
    Common lmdb-format dataset for storing image bases for OCR
    Seeks all images by keys image-XXXXXXXXX and their text labels by label-XXXXXXXXX
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

        return img, label
