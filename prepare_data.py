from os import listdir
from os.path import isfile
from enum import Enum
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


class GroupGeneratorFromDataset(keras.utils.Sequence):
    """
    A keras Sequence to be used as an image generator for the model.
    """

    def __init__(self, dataset_names, batch_size, filenames, input_size):
        self.inputsize = input_size
        self.filenames = filenames
        self.batchsize = batch_size
        self.dataset_names = dataset_names
        self._filter_files()
        self.shuffle()

    def _filter_files(self):
        filtered_names = [name.replace("\\", "/") for name in self.dataset_names]
        filtered_filenames = {}
        for category, files in self.filenames.items():
            filtered_filenames[category] = [file for file in files if file in filtered_names]
        self.filenames = filtered_filenames


    def __len__(self):
        return math.ceil(len(self.x) / self.batchsize)

    def names_at_batch(self, idx):
        x_names = self.x[idx * self.batchsize:(idx + 1) * self.batchsize]
        y_names = np.asarray(self.y[idx * self.batchsize:(idx + 1) * self.batchsize])
        return x_names, y_names

    def __getitem__(self, idx):
        x_names = self.x[idx * self.batchsize:(idx + 1) * self.batchsize]
        y_names = np.asarray(self.y[idx * self.batchsize:(idx + 1) * self.batchsize])

        # result = np.asarray([[img_to_array(load_img(file_name, target_size=(299, 299))) / 255.0 for file_name in file_array] for file_array in x_names])
        result = np.asarray([[preprocess_input(img_to_array(load_img(file_name, target_size=(299, 299)))) for file_name in file_array] for file_array in x_names])
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

        # result = np.asarray([[img_to_array(load_img(file_name, target_size=(299, 299))) / 255.0 for file_name in file_array] for file_array in x_names])
        result = np.asarray([[preprocess_input(img_to_array(load_img(file_name, target_size=(299, 299)))) for file_name in file_array] for file_array in x_names])
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


def get_filenames(dataset):
    x = []
    y = []
    directory = directories[dataset]
    category_num = 0
    
    for category in listdir(directory):
        if isfile(category):
            continue

        for file in listdir("{0}/{1}".format(directory, category)):
            if file[-4:] not in [".jpg", "jpeg", ".png", ".gif"]:
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
    

def prep_data(dataset, keras_train_dataset, keras_val_dataset, batch_size, group_size):
    filenames = make_image_dict(*get_filenames(dataset))
    
    # make generators
    train_gen = GroupGeneratorFromDataset(keras_train_dataset.file_paths, batch_size, filenames, group_size)
    val_gen = GroupGeneratorFromDataset(keras_val_dataset.file_paths, batch_size, filenames, group_size)

    return train_gen, val_gen


def prep_data_single(dataset, batch_size):
    train = keras.preprocessing.image_dataset_from_directory(directories[dataset], batch_size=batch_size, image_size=(299, 299), seed=42, validation_split=0.15, subset='training')
    val = keras.preprocessing.image_dataset_from_directory(directories[dataset], batch_size=batch_size, image_size=(299, 299), seed=42, validation_split=0.15, subset='validation')
    return train, val
    