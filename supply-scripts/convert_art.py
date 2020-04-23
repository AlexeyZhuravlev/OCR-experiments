import os
import json
from tqdm import tqdm
from create_lmdb import LmdbDatasetCreator
import codecs

# ArT ~ 600MB
MAP_SIZE = round(600 * 1024 * 1024)
SYMBOLS = set(open(r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Compiled\AlphanumericWithPunct.txt").read())

def convert_art(source_dir, target_db):
    elements_path = os.path.join(source_dir, "train_task2_labels.json")
    elements = json.load(codecs.open(elements_path, encoding="utf-8"))
    images_path = os.path.join(source_dir, "train_task2_images")

    database = LmdbDatasetCreator(target_db, map_size=MAP_SIZE)

    for key, description in tqdm(elements.items()):
        assert len(description) == 1
        description = description[0]

        image_path = os.path.join(images_path, "{}.jpg".format(key))
        if description["language"] != "Latin":
            continue

        label = description["transcription"]
        if len(set(label) - SYMBOLS) > 0:
            continue
        image_bytes = open(image_path, "rb").read()

        database.add_instance(image_bytes, label)

    database.close()

convert_art(r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Sources\RRC-ArT", r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Compiled\Real\TRAIN_RRC_ArT")