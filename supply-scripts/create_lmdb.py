import lmdb

class LmdbDatasetCreator:
    def __init__(self, result_path, cache_size=1000, map_size=1000000):
        self.env = lmdb.open(result_path, map_size=map_size)
        self.cache = {}
        self.cache_size = cache_size
        self.num_elements = 0

    def add_instance(self, image_bin, label):
        """Registers new instance in the dataset"""
        self.num_elements += 1
        image_key = "image-{:09d}".format(self.num_elements).encode()
        label_key = "label-{:09d}".format(self.num_elements).encode()
        self.cache[image_key] = image_bin
        self.cache[label_key] = label.encode()

        if self.num_elements % self.cache_size == 0:
            self._dump_cache()

    def close(self):
        """Dumps all non-written cache elements and closes DB"""
        self.cache['num-samples'.encode()] = str(self.num_elements).encode()
        self._dump_cache()
        self.env.close()

    def _dump_cache(self):
        with self.env.begin(write=True) as txn:
            for key, value in self.cache.items():
                txn.put(key, value)
        self.cache.clear()

    def _dum_cache_impl(self):
        with self.env.begin(write=True) as txn:
            for key, value in self.cache.items():
                txn.put(key, value)
