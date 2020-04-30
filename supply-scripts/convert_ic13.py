from create_lmdb import LmdbDatasetCreator
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import ast

SIZE = 90 * 1024 * 1024

def convert_ic13(source_dir, gt_file, target_db):
    gt_file = os.path.join(source_dir, gt_file)
    database = LmdbDatasetCreator(target_db, map_size=SIZE)

    for line in tqdm(open(gt_file, encoding="utf-8").readlines()):
        coma_pos = line.find(",")
        filename = line[:coma_pos]

        annotation_str = line[coma_pos + 2:-1]
        annotation = ast.literal_eval(annotation_str)

        image_path = os.path.join(source_dir, filename)
        image_bytes = open(image_path, "rb").read()

        database.add_instance(image_bytes, annotation)

    database.close()

#convert_ic13(r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Sources\IC13-test", "gt.txt", r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Compiled\Real\Test\IC13")
convert_ic13(r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Sources\IC15-train", "gt.txt", r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Compiled\Real\Train\IC15")