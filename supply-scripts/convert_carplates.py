import io
from tqdm import tqdm
from create_lmdb import LmdbDatasetCreator
import codecs
import numpy as np
import pandas as pd
from PIL import Image

SIZE = 30 * 1024 * 1024

def convert_carplates(images_file, groundtruth_file, target_db):
    images = np.load(images_file)
    groundtruth = list(pd.read_csv(groundtruth_file)["Number"])
    base = LmdbDatasetCreator(target_db, map_size=SIZE)

    for image, annotation in tqdm(zip(images, groundtruth)):
        image = Image.fromarray(image)
        byte_ref = io.BytesIO()
        image.save(byte_ref, format="JPEG")
        image_bytes = byte_ref.getvalue()

        base.add_instance(image_bytes, annotation)

    base.close()

convert_carplates(r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Sample\car-numbers\plates_train.npy", 
                  r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Sample\car-numbers\numbers_train.csv", 
                  r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Sample\Train")
