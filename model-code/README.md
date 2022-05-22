# Model Code

This directory contains all contains all code the build, train and validate the models used throughout this project.  Also contains code to prepare data for models and visualise/analyse the extracted features of a selected model.

## 3_img_ensemble.ipynb

Jupyter notebook file containing all code to build and train a 3-image ensemble model on a selected dataset.  Running the model requires that a feature extractor model has been trained on the same dataset and saved.  Three of these models are loaded and used to extract features from unique images, before passing this information on to a concatenation layer and set of dense layers for classification.

## augment_testing.ipynb

Jupter notebook file used to assess the accuracy of augmented datasets on augment-trained models.  E.g. computes and compares the accuracy on a grayscale dataset, blurred dataset and un-augmented dataset using a model trained on un-augmented images.

## ensemble_aug_classifier.ipynb

Jupyter notebook file containing code to build and train the ensemble-augment classifier, the final model of the project.  This model comprises of an ensemble of three feature extractors, each trained on images with different augments: one is trained on un-augmented images, one on greyed images and one on blurred images.  The features extracted by these models are then passed onto different states of an RNN, which computes a classification.

## extractor_trainer.ipynb

Jupyter notebook file containing all code to build and train an InceptionV3 extractor model on a selected dataset.  There is also an option to apply augments to the dataset, if the extractor is to be used on augmented images in a classifier model (e.g. in the ensemble_aug_classifier model).

## feature_diffs.ipynb

Jupyter notebook file containing code to compute and visualise the feature maps from the first convolutional layers of standard-trained, grey-trained and blur-trained extractor models.  These outputs can be used to assess the differences in extracted features between all three extractor types.

## multi_img_rnn_classifier.ipynb

Jupyter notebook containing code to build and train a multi-image RNN classifier.  The model uses a single InceptionV3 extractor, which sequentially extract features from an input of multiple images and passes the output to different states of an RNN for classification.

## prepare_data.py

Python file used as a module for all model files in the project.  Provides a means of preparing the various different datasets needed to run each model.  Datasets are passed to models as custom generator classes, which perform all operations needed for an image to be predicted by a model - opening the image, converting the image to a 299x299x3 array, applying any specified augments, and preprocessing/normalising the data.

## randaug_classifier.ipynb

Jupyter file containing code to run a random-augment classifier - a model which uses the multi-image classifier architecture (as used in multi_img_rnn_classifier.ipynb) but uses a single original image per classification, with the remaining images of each model input being created as randomly augmented copies of the original.
