from create_lmdb import LmdbDatasetCreator
import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import io

SIZE = 20 * 1024 * 1024

def convert_svt(source_dir, xml_name, target_db):
    xml_path = os.path.join(source_dir, xml_name)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    database = LmdbDatasetCreator(target_db, map_size=SIZE)

    for image in tqdm(root):
        image_name = image.find('imageName').text
        rects = image.find('taggedRectangles')
        image_path = os.path.join(source_dir, image_name)
        image = Image.open(image_path)
        for rect in rects:
            attr = rect.attrib
            annotation = rect.find('tag').text
            x, y, w, h = float(attr["x"]), float(attr["y"]), float(attr["width"]), float(attr["height"])
            cropped = image.crop((x, y, x + w, y + h))
            byte_ref = io.BytesIO()
            cropped.save(byte_ref, format="JPEG")
            image_bytes = byte_ref.getvalue()

            database.add_instance(image_bytes, annotation)

    database.close()

convert_svt(r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Sources\svt\svt1", "train.xml", r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Compiled\Real\Train\SVT")
#convert_svt(r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Sources\IC03-scene-test\SceneTrialTest", "words.xml", r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Compiled\Real\Test\IC03")