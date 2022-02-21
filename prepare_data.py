from os import listdir
from os.path import isfile
from enum import Enum
from re import X
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras
import numpy as np
import math
import random

Dataset = Enum("Dataset", "carabid thirty eighty ninety")

directories = {Dataset.carabid: "datasets/carabid", \
    Dataset.thirty: "datasets/30 animal dataset/ANIMAL-N30/ANIMALS", \
    Dataset.eighty: "datasets/80 animal dataset/train", \
    Dataset.ninety: "datasets/90 animal dataset/animals/animals"}


class GroupGenerator(keras.utils.Sequence):
    """
    A keras Sequence to be used as an image generator for the model.
    """

    def __init__(self, filenames, batchsize, input_size):
        self.batchsize = batchsize
        self.filenames = filenames
        self.inputsize = input_size
        self.shuffle()

    def __len__(self):
        return math.ceil(len(self.x) / self.batchsize)

    def names_at_batch(self, idx):
        x_names = self.x[idx * self.batchsize:(idx + 1) * self.batchsize]
        y_names = np.asarray(self.y[idx * self.batchsize:(idx + 1) * self.batchsize])
        return x_names, y_names

    def __getitem__(self, idx):
        x_names = self.x[idx * self.batchsize:(idx + 1) * self.batchsize]
        y_names = np.asarray(self.y[idx * self.batchsize:(idx + 1) * self.batchsize])

        result = np.asarray([[img_to_array(load_img(file_name, target_size=(299, 299))) / 255.0 for file_name in file_array] for file_array in x_names])
        return result, y_names

    def shuffle(self):
        groups = []
        for category, files in self.filenames.items():
            random.shuffle(files)
            subgroup = []
            for file in files:
                subgroup.append(file)
                if len(subgroup) == self.inputsize:
                    groups.append((subgroup, category))
                    subgroup = []

        random.shuffle(groups)
        self.x = np.asarray([group[0] for group in groups])
        self.y = np.asarray([group[1] for group in groups])

    def num_classes(self):
        return len(self.filenames)


class SingleGenerator(keras.utils.Sequence):
    """
    A keras Sequence to be used as an image generator for the model.
    """

    def __init__(self, x, y, batchsize):
        self.batchsize = batchsize
        self.x = x
        self.y = y
        self.len = math.ceil(len(self.x) / self.batchsize)

    def __len__(self):
        return self.len

    def names_at_batch(self, idx):
        x_names = self.x[idx * self.batchsize:(idx + 1) * self.batchsize]
        y_names = np.asarray(self.y[idx * self.batchsize:(idx + 1) * self.batchsize])
        return x_names, y_names

    def __getitem__(self, idx):
        x_names = self.x[idx * self.batchsize:(idx + 1) * self.batchsize]
        y_names = np.asarray(self.y[idx * self.batchsize:(idx + 1) * self.batchsize])

        result = np.asarray([img_to_array(load_img(file_name, target_size=(299, 299))) / 255.0 for file_name in x_names])
        return result, y_names

    def num_classes(self):
        cats = []
        for cat in self.y:
            if cat not in cats:
                cats.append(cat)
        return len(cats)


def get_filenames(dataset):
    x = []
    y = []
    directory = directories[dataset]
    category_num = 0
    
    for category in listdir(directory):
        if isfile(category):
            continue

        for file in listdir("{0}/{1}".format(directory, category)):
            if file[-4:] != ".jpg":
                continue
            x.append("{0}/{1}/{2}".format(directory, category, file))
            y.append(category_num)
        category_num += 1

    return x, y


def make_image_dict(x, y):
    names = {}
    for i in range(len(x)):
       files = names.get(y[i], []) 
       files.append(x[i])
       names[y[i]] = files

    return names
    

def prep_data(dataset, batch_size, group_size):
    x, y = get_filenames(dataset)

    # 15% of all the images are set aside as the test set
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

    # 17% of the non-test images are set aside as the validation set
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.17, random_state=42)

    # make generators
    train_gen = GroupGenerator(make_image_dict(x_train, y_train), batch_size, group_size)
    val_gen = GroupGenerator(make_image_dict(x_val, y_val), batch_size, group_size)
    test_gen = GroupGenerator(make_image_dict(x_test, y_test), batch_size, group_size)

    return train_gen, val_gen, test_gen


def prep_data_single(dataset, batch_size):
    x, y = get_filenames(dataset)

    # 15% of all the images are set aside as the test set
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

    # 17% of the non-test images are set aside as the validation set
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.17, random_state=42)

    # make generators
    train_gen = SingleGenerator(x_train, y_train, batch_size)
    val_gen = SingleGenerator(x_val, y_val, batch_size)
    test_gen = SingleGenerator(x_test, y_test, batch_size)

    return train_gen, val_gen, test_gen
