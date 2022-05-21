# Zachary Oar (45314669) Thesis Project: AI Species Recognition

## The Project

With biodiversity consistently declining around the world, the ability to easily recognise and keep track of global species grows all the more valuable.  Currently, the act of manually classifying existing and new species of animals, insects and plants is a time-consuming task that requires years of experience in an uncommon profession.  The prospect of AI species recognition is exciting for those working in fields like taxonomy and zoology and it eases the significant barrier of entry for those going into these fields.  This project aims to implement new approaches to species classification using machine learning algorithms.

## Machine Learning Algorithms

Currently, the most popular method of AI species identification is deep learning – in particular with convolutional neural networks.  Convolutional networks are especially useful in image classification because they are effective at extracting meaningful features from input pixels – things such as shape outlines, subject colours, and so on.  This process of feature extraction is optimised during training, and it is these characteristics of the input that are propagated though the many layers of a model to return a classification.

### InceptionV3

This repository makes use of the InceptionV3 network - a prebuilt model developed by Google, with weights pre-trained from the annual ImageNet competition.  The InceptionV3 model is very effective for use in image feature extraction, providing a network with important characteristics for classification tasks.  As such, it has been re-trained for use as a feature extractor in many of the models in this repository.

### Multi-Image Classifiers

The standard approach to image classification is using a network uses a single image as a model's input.  However, an approach worth consideration is the use of multiple images as a model's input.  The provision of more input images to a model should (in theory) provide more features to be used for classification, and ultimately lead to improved accuracy.  Most models in this repository aim to implement a means of classifying multiple images.

### Image Augmentation

Image augmentation is a means of altering an image such that a model assesses it to be a new unique piece of data.  This is typically used during the training of a model to artificially increase the size of a model's training set.  Throughout this project, a new use for image augmentation was tested: using augmentation to artificially create new inputs for a multi-image classifier.

## Repository Structure

### GUI/

This folder contains code to run the interactive GUI, allowing a user to select an image from their filesystem.  The selected image is shown on the interface, and the top-3 guesses determined by a backend classifier network will be displayed on screen.

### logs/

This folder contains all Tensorboard logs created during the training and validation of models.  Each model type is separated into three categories: "carabid", "eighty" and "ninety"; each representing logs for models trained on each of the three respective datasets used throughout this project.

### model-code/

Contains all code the build, train and validate the models used throughout this project.  Also contains code to prepare data for models and visualise the extracted features of a selected model.

### model-saves/

All models are saved under a folder named "model-saves", which has been omitted from this repository due to the size of the save files.  To download a model for yourself, use [this Google Drive link](https://drive.google.com/drive/folders/1pT_3Mf6giyD1RMzpwDOIQyeXtXNtmsDe?usp=sharing).

## Datasets

In all code for the project assumes that datasets are downloaded and stored under a folder names "datasets"
The following datasets were used thoughout the project:

### [carabid](https://www.kaggle.com/datasets/kmldas/insect-identification-from-habitus-images)

Dataset containing 63364 images of beetles, from 291 different species.  Images are all top-down, so there is little room for different perspectives between images.

### [nintey](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)

Dataset containing 5400 images of mammals, insects and birds from 90 different species.  Images vary greatly in persepective, size and number of subjects.

### [eighty](https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset?select=train)

Dataset containing 27831 images of mammals, insects and birds from 80 different species.  Images are scraped from Google Images, so they are very inconsistent in terms of quality, size, perspective, etc.

## Requirements

All code for this project was run on an environment with the following:

* Python 3.10.4
* Tensorflow 2.8.0
* Numpy 1.22.3
* Albumentations 1.1.0
