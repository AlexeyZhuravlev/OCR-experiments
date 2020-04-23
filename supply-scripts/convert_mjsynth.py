import tarfile
from create_lmdb import LmdbDatasetCreator
from tqdm import tqdm

# MJSynth ~ 26.5 GB
MAP_SIZE = round(26.5 * 1073741824)
IMAGES_COUNT = 8919273

def convert_mjsynth(source_archive_path, target_db_path):
    archive = tarfile.open(source_archive_path)
    database = LmdbDatasetCreator(target_db_path, map_size=MAP_SIZE)

    pbar = tqdm(total=IMAGES_COUNT)
    while True:
        member = archive.next()
        if member is None:
            break

        if not member.isfile():
            continue

        filename = member.name
        if not filename.endswith(".jpg"):
            continue

        image = archive.extractfile(member).read()
        label = filename.split("_")[-2]

        database.add_instance(image, label)
        pbar.update(1)

    pbar.close()
    database.close()

convert_mjsynth(r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Sources\mjsynth.tar.gz", r"C:\Users\User\Documents\PhD-study\OCR-TextRecognition-Data\Compiled\Synthetic\MJ")
