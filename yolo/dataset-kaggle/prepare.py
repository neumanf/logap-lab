import logging
import requests
import zipfile
import os
import random
import shutil


def download_dataset():
    logging.info('Downloading dataset...')
    # https://www.kaggle.com/datasets/stmlen/windturbine-damage-dataset-yolo-format
    response = requests.get('https://www.kaggle.com/api/v1/datasets/download/stmlen/windturbine-damage-dataset-yolo-format')

    logging.info('Saving dataset...')
    with open('./windturbine-damage-dataset-yolo-format.zip', 'wb') as f:
        f.write(response.content)

    logging.info('Extracting dataset...')
    with zipfile.ZipFile('./windturbine-damage-dataset-yolo-format.zip', 'r') as zip:
        zip.extractall('.')

    os.remove('./windturbine-damage-dataset-yolo-format.zip')

def process_dataset():
    logging.info('Processing dataset...')

    source_images = 'data/NordTank1/images'
    source_labels = 'data/NordTank1/labels'
    dest_dir = 'dataset'

    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, split, 'labels'), exist_ok=True)

    image_files = [f for f in os.listdir(source_images) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(image_files)

    total = len(image_files)
    train_split = int(total * train_ratio)
    val_split = int(total * (train_ratio + val_ratio))

    splits = {
        'train': image_files[:train_split],
        'val': image_files[train_split:val_split],
        'test': image_files[val_split:]
    }

    for split, files in splits.items():
        for img_file in files:
            label_file = os.path.splitext(img_file)[0] + '.txt'

            shutil.copy(os.path.join(source_images, img_file), os.path.join(dest_dir, split, 'images', img_file))
            shutil.copy(os.path.join(source_labels, label_file), os.path.join(dest_dir, split, 'labels', label_file))

    shutil.rmtree('./data')


if __name__ == '__main__':
    download_dataset()
    process_dataset()