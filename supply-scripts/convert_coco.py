from create_lmdb import LmdbDatasetCreator
import codecs
import os
from tqdm import tqdm

SIZE = 40 * 1024 * 1024

def convert_coco(gt_file, images_path, target_db):
    groundtruth = codecs.open(gt_file, encoding="utf-8").read().split()
    database = LmdbDatasetCreator(target_db, map_size=SIZE)

    for line in tqdm(groundtruth):
        split_result = line.split(",")
        if len(split_result) != 2:
            continue

        number, annotation = split_result[0], split_result[1]
        if "\\" in annotation or "|" in annotation:
            continue

        image_name = "{}.jpg".format(number)
        image_bytes = open(os.path.join(images_path, image_name), "rb").read()

        database.add_instance(image_bytes, annotation)

    database.close()

convert_coco(r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Sources\COCO-Text-words-trainval\val_words_gt.txt", 
             r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Sources\COCO-Text-words-trainval\val_words",
             r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Compiled\Real\TRAIN_COCO_VAL")