import zipfile
from create_lmdb import LmdbDatasetCreator
from tqdm import tqdm
import os

# SynthAdd ~ 3.5 GB
MAP_SIZE = round(3.5 * 1073741824)
ROOT_NAME = "SynthText_Add"
ANNOTATIONS = "annotationlist"

def convert_synthadd(source_archive_path, target_db_path):
    archive = zipfile.ZipFile(source_archive_path)
    database = LmdbDatasetCreator(target_db_path, map_size=MAP_SIZE)

    for i in range(1, 21):
        annotations_file_path = "{}/{}/gt_{}.txt".format(ROOT_NAME, ANNOTATIONS, i)
        image_dir = "{}/crop_img_{}".format(ROOT_NAME, i)
        print("Working with {}".format(annotations_file_path))

        annotations = archive.open(annotations_file_path).read().decode("utf-8").split()
        for annotation in tqdm(annotations):
            idx = annotation.find(",")
            filename = annotation[:idx]
            groundtruth = annotation[idx+2:-1]
            filepath = "{}/{}".format(image_dir, filename)
            image_bytes = archive.open(filepath).read()

            database.add_instance(image_bytes, groundtruth)

    database.close()
    archive.close()


convert_synthadd(r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Sources\SynthText_Add.zip", r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Compiled\Synthetic\SynthAdd")
