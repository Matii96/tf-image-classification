import os
import math
import random
import numpy as np
import json
import cv2

#load configuration
root_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(root_dir, 'model', 'model.ckpt')
train_data_name = 'train_data.json'
train_data_path = os.path.join(root_dir, 'model', train_data_name)
dataset_path = os.path.join(root_dir, 'dataset')

with open('config.json', encoding='utf-8') as json_data:
    config = json.load(json_data)

train_data = None
if os.path.isfile(train_data_path):
    with open(train_data_path, encoding='utf-8') as json_data:
        train_data = json.load(json_data)

progress_last = -1
def display_progress(i, acc, loss, width=100):
    progress = int(width * i / config['training']['n_epoch'])
    update_statistics = (i%config['training']['display_step'] == 0)

    #Clear current line
    global progress_last
    if progress > progress_last or update_statistics:
        print('{0}\r'.format(' '*117), end='')

    #Print information about test accuracy and loss
    if update_statistics:
        print("#%d Accuracy=%.2f%%, loss=%.2f" % (i, acc*100, loss))

    #Update progress bar
    if progress > progress_last or update_statistics:
        print('Progress: [{0}{1}] {2}%\r'.format('#'*progress, ' '*(width - progress), progress), end='')
    progress_last = progress

def convert_image_to_2d_array(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, (config['model']['image_width'], config['model']['image_height']), cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.divide(img, 255)
    img = list(img)
    for i in range(0, len(img)):
        img[i] = [[pixel] for pixel in img[i]]

    #Finishing job
    return img

#Collect labels of all classes
def load_labels():
    #No train_data file saved in model
    if train_data is None:
        if not os.path.isdir('dataset'):
            return None
        return os.listdir(dataset_path)
    return train_data['labels']
labels = load_labels()

#Collect all data set file names
dataset = None
def load_dataset():
    dataset = []
    for label in labels:
        images_set = []
        for file in os.listdir(os.path.join(dataset_path, label)):
            if not os.path.isfile(os.path.join(dataset_path, label, file)):
                continue
            images_set.append(file)
        dataset.append(images_set)

    #Finishing job
    return dataset

#Read random batch from dataset
def random_batch(size):
    global dataset
    if dataset is None:
        dataset = load_dataset()

    result = []
    images_per_label = math.ceil(size / len(labels))

    for images_setID in range(0, len(dataset)):
        images_set = dataset[images_setID]

        random_set = []
        random_set += images_set * int(images_per_label // len(images_set))
        random_set += random.sample(images_set, images_per_label % len(images_set))

        for image in random_set:
            file_path = os.path.join(labels[images_setID], image)
            result.append({
                'file_path': file_path,
                'images_setID': images_setID
            })

    #Pick first size elements of result
    result = random.sample(result, size)

    result_x = []
    result_y = []
    for row in result:
        result_x.append(convert_image_to_2d_array(
            os.path.join(dataset_path, row['file_path'])))
        label_vector = [0] * len(labels)
        label_vector[row['images_setID']] = 1
        result_y.append(label_vector)

    return result_x, result_y
