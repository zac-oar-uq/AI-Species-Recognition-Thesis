from os import listdir
from os.path import isfile
from enum import Enum
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras
import numpy as np
import math
import albumentations as A
import random

Dataset = Enum("Dataset", "carabid eighty ninety")

# directories to load each of the three datasets
directories = {Dataset.carabid: "../datasets/carabid", \
    Dataset.eighty: "../datasets/80 animal dataset/train", \
    Dataset.ninety: "../datasets/90 animal dataset/animals/animals"}

# number of classes in each dataset
class_num = {Dataset.eighty: 80, Dataset.ninety: 90, Dataset.carabid: 291}


def num_classes(dataset):
    return class_num[dataset]


class MultiImageRNNGenerator(keras.utils.Sequence):
    """
    Keras generator which provides model inputs in the form of multiple images of the same class.
    The inputs are formatted to work with the multi_image_rnn_classifier model.
    """
    def __init__(self, dataset_names, batch_size, filenames, num_inputs):
        self.num_inputs = num_inputs
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

        result = np.asarray([[preprocess_input(img_to_array(load_img(file_name, target_size=(299, 299)))) for file_name in file_array] for file_array in x_names])
        return result, y_names

    def shuffle(self):
        groups = []
        for category, files in self.filenames.items():
            random.shuffle(files)
            subgroup = []
            for file in files:
                subgroup.append(file)
                if len(subgroup) == self.num_inputs:
                    groups.append((subgroup, category))
                    subgroup = []

        random.shuffle(groups)
        self.x = np.asarray([group[0] for group in groups])
        self.y = np.asarray([group[1] for group in groups])

    def num_classes(self):
        return len(self.filenames)


class RandAugGenerator(keras.utils.Sequence):
    """
    Keras generator which creates inputs for a model by creating sets of multiple randomly
    augmented images from a single original image.
    Inputs are formatted to work with the randaug_classifier model.
    """
    def __init__(self, dataset_names, batch_size, filenames, input_size, augment):
        self.num_inputs = input_size
        self.filenames = filenames
        self.batchsize = batch_size
        self.dataset_names = dataset_names
        self.augment = augment
        self._filter_files()
        self.shuffle()

    def _filter_files(self):
        filtered_names = [name.replace("\\", "/") for name in self.dataset_names]
        filtered_filenames = {}
        for category, files in self.filenames.items():
            filtered_filenames[category] = [file for file in files if file in filtered_names]
        self.filenames = filtered_filenames

    def _create_augment_array(self, img):
        result = [preprocess_input(img)]
        for i in range(self.inputsize - 1):
            result.append(preprocess_input(self.augment(image=img)["image"]))
        return np.asarray(result)

    def __len__(self):
        return math.ceil(len(self.x) / self.batchsize)

    def names_at_batch(self, idx):
        x_names = self.x[idx * self.batchsize:(idx + 1) * self.batchsize]
        y_names = np.asarray(self.y[idx * self.batchsize:(idx + 1) * self.batchsize])
        return x_names, y_names

    def __getitem__(self, idx):
        x_names = self.x[idx * self.batchsize:(idx + 1) * self.batchsize]
        y_names = np.asarray(self.y[idx * self.batchsize:(idx + 1) * self.batchsize])

        result = np.asarray([self._create_augment_array(np.asarray(img_to_array(load_img(file_name[0], target_size=(299, 299))), dtype='uint8')) for file_name in x_names])
        return result, y_names

    def shuffle(self):
        groups = []
        for category, files in self.filenames.items():
            subgroup = []
            for file in files:
                subgroup.append(file)
                groups.append((subgroup, category))
                subgroup = []

        random.shuffle(groups)
        self.x = np.asarray([group[0] for group in groups])
        self.y = np.asarray([group[1] for group in groups])

    def num_classes(self):
        return len(self.filenames)


class EnsembleAugGenerator(keras.utils.Sequence):
    """
    Keras generator which creates inputs for a model by creating a grey-augmented and 
    blur-augmented copies from a single original image.
    Inputs are formatted to work with the ensemble_aug_classifier model.
    """
    def __init__(self, dataset_names, batch_size, filenames):
        self.filenames = filenames
        self.batchsize = batch_size
        self.dataset_names = dataset_names

        self.gray_aug = A.Compose([A.ToGray(p=1.0)])
        self.blur_aug = A.Compose([A.Blur(p=1.0)])

        self._filter_files()
        self.shuffle()

    def _filter_files(self):
        filtered_names = [name.replace("\\", "/") for name in self.dataset_names]
        filtered_filenames = {}
        for category, files in self.filenames.items():
            filtered_filenames[category] = [file for file in files if file in filtered_names]
        self.filenames = filtered_filenames
    
    def _create_augment_array(self, images):
        return [np.array([preprocess_input(img) for img in images]), np.array([preprocess_input(self.gray_aug(image=img)["image"]) for img in images]), 
                np.array([preprocess_input(self.blur_aug(image=img)["image"]) for img in images])]

    def __len__(self):
        return math.ceil(len(self.x) / self.batchsize)

    def names_at_batch(self, idx):
        x_names = self.x[idx * self.batchsize:(idx + 1) * self.batchsize]
        y_names = np.asarray(self.y[idx * self.batchsize:(idx + 1) * self.batchsize])
        return x_names, y_names

    def __getitem__(self, idx):
        x_names = self.x[idx * self.batchsize:(idx + 1) * self.batchsize]
        y_names = np.asarray(self.y[idx * self.batchsize:(idx + 1) * self.batchsize])
        result = self._create_augment_array([np.asarray(img_to_array(load_img(file_name[0], target_size=(299, 299))), dtype='uint8') for file_name in x_names])
        return result, y_names

    def shuffle(self):
        groups = []
        for category, files in self.filenames.items():
            subgroup = []
            for file in files:
                subgroup.append(file)
                groups.append((subgroup, category))
                subgroup = []

        random.shuffle(groups)
        self.x = np.asarray([group[0] for group in groups])
        self.y = np.asarray([group[1] for group in groups])

    def num_classes(self):
        return len(self.filenames)


class MultiImageEnsembleGenerator(keras.utils.Sequence):
    """
    Keras generator which prepares model inputs in sets of multiple images.
    Inputs are formatted for ensemble models.
    """
    def __init__(self, dataset_names, batch_size, filenames, num_inputs):
        self.num_inputs = num_inputs
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

        result = [[] for i in range(self.num_inputs)]
        for group in x_names:
            for i, file in enumerate(group):
                result[i].append(preprocess_input(img_to_array(load_img(file, target_size=(299, 299)))))
            
        return [np.array(i) for i in result], y_names

    def shuffle(self):
        groups = []
        for category, files in self.filenames.items():
            random.shuffle(files)
            subgroup = []
            for file in files:
                subgroup.append(file)
                if len(subgroup) == self.num_inputs:
                    groups.append((subgroup, category))
                    subgroup = []

        random.shuffle(groups)
        self.x = np.asarray([group[0] for group in groups])
        self.y = np.asarray([group[1] for group in groups])

    def num_classes(self):
        return len(self.filenames)


def get_filenames(dataset):
    """
    Retrieves all filenames of images from a selected dataset and returns them along
    with an array of their associated classes.
    """
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
    """
    Makes a dictionary in which all keys are the names of a dataset's labels and
    all entries are a list of image filenames associated with the key's label.
    """
    names = {}
    for i in range(len(x)):
       files = names.get(y[i], []) 
       files.append(x[i])
       names[y[i]] = files
    return names


def do_nothing(image):
    """
    Default augment option: does not augment an image in any way.
    """
    return {"image": image}


def prep_multi_img_rnn_dataset(dataset, keras_train_dataset, keras_val_dataset, batch_size, num_inputs):
    """
    Prepares a train and validation set for a multi_img_rnn_classifier model.
    """
    filenames = make_image_dict(*get_filenames(dataset))
    train_gen = MultiImageRNNGenerator(keras_train_dataset.file_paths, batch_size, filenames, num_inputs)
    val_gen = MultiImageRNNGenerator(keras_val_dataset.file_paths, batch_size, filenames, num_inputs)
    return train_gen, val_gen


def prep_ensemble_aug_dataset(dataset, keras_train_dataset, keras_val_dataset, batch_size):
    """
    Prepares a train and validation set for a ensemble_aug_classifier model.
    """
    filenames = make_image_dict(*get_filenames(dataset))
    train_gen = EnsembleAugGenerator(keras_train_dataset.file_paths, batch_size, filenames)
    val_gen = EnsembleAugGenerator(keras_val_dataset.file_paths, batch_size, filenames)
    return train_gen, val_gen


def prep_rand_aug_dataset(dataset, keras_train_dataset, keras_val_dataset, batch_size, group_size, augment=do_nothing):
    """
    Prepares a train and validation set for a randaug_classifier model.
    """
    if not augment:
        augment = do_nothing
    filenames = make_image_dict(*get_filenames(dataset))
    train_gen = RandAugGenerator(keras_train_dataset.file_paths, batch_size, filenames, group_size, augment)
    val_gen = RandAugGenerator(keras_val_dataset.file_paths, batch_size, filenames, group_size, augment)
    return train_gen, val_gen


def prep_multi_img_ensemble_dataset(dataset, keras_train_dataset, keras_val_dataset, batch_size, num_inputs):
    """
    Prepares a train and validation set for an ensemble classifier model.
    """
    filenames = make_image_dict(*get_filenames(dataset))
    train_gen = MultiImageEnsembleGenerator(keras_train_dataset.file_paths, batch_size, filenames, num_inputs)
    val_gen = MultiImageEnsembleGenerator(keras_val_dataset.file_paths, batch_size, filenames, num_inputs)
    return train_gen, val_gen


def prep_dataset(dataset, batch_size):
    """
    Prepares a basic train and validation set for a model.
    """
    train = keras.preprocessing.image_dataset_from_directory(directories[dataset], batch_size=batch_size, image_size=(299, 299), seed=42, validation_split=0.15, subset='training')
    val = keras.preprocessing.image_dataset_from_directory(directories[dataset], batch_size=batch_size, image_size=(299, 299), seed=42, validation_split=0.15, subset='validation')
    return train, val
