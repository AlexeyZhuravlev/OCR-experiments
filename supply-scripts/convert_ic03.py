from create_lmdb import LmdbDatasetCreator
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

SIZE = 20 * 1024 * 1024

def convert_svt(source_dir, xml_name, target_db):
    xml_path = os.path.join(source_dir, xml_name)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    database = LmdbDatasetCreator(target_db, map_size=SIZE)

    for image in tqdm(root):
        attr = image.attrib
        image_name = attr['file']
        annotation = attr['tag']
        image_path = os.path.join(source_dir, image_name)
        image_bytes = open(image_path, "rb").read()

        database.add_instance(image_bytes, annotation)

    database.close()

convert_svt(r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Sources\IC03-train", "word.xml", r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Compiled\Real\Train\IC03")
