import scipy.io
from create_lmdb import LmdbDatasetCreator
import os
from tqdm import tqdm

SIZE = 40 * 1024 * 1024

def convert_iit5k(root, data_file_name, target_db):
    data_file_path = os.path.join(root, data_file_name)
    data = scipy.io.loadmat(data_file_path)
    elements = data['traindata'][0]
    database = LmdbDatasetCreator(target_db, map_size=SIZE)

    for element in tqdm(elements):
        image_name = element[0][0]
        groundtruth = element[1][0]
        image_path = os.path.join(root, image_name)
        image_bytes = open(image_path, "rb").read()

        database.add_instance(image_bytes, groundtruth)

    database.close()

convert_iit5k(r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Sources\IIIT5K", "traindata.mat",
              r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Compiled\Real\Train\IIIT5K")
