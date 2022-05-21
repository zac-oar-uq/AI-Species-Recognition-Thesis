import tensorflow as tf
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib

matplotlib.rcParams.update({'font.size': 8})


class NinetyModel():
    def __init__(self):
        self._model = tf.keras.models.load_model("../model-saves/multi-aug-classifiers/ninety/NINETY-ENSEMBLE-AUG-CLASSIFIER/classifier/savefile.hdf5")
        self._gray_aug = A.Compose([A.ToGray(p=1.0)])
        self._blur_aug = A.Compose([A.Blur(p=1.0)])
        self.standard_img = np.zeros((299, 299, 3))
        self.blurred_img = np.zeros((299, 299, 3))
        self.grayed_img = np.zeros((299, 299, 3))
        self.initialise_labels()
        self.load_feature_mappers()
        self.load_inception_model()
        self.first_pass()

    def load_inception_model(self):
        """
        Loads an Inception classifier.
        """
        full_model = tf.keras.models.load_model(f"../model-saves/extractors/ninety/NINETY-EXTRACTOR/extractor/savefile.hdf5")
        self._inception_model = tf.keras.Sequential([full_model.layers[0].layers[0], 
                                                        full_model.layers[0].layers[-1],
                                                        full_model.layers[-1]])

    def load_feature_mappers(self):
        """
        Loads three extractors used to plot the feature outputs from the first convolutional
        layer of the aug classifier.
        """
        dataset_name = "ninety"
        standard_model_path = f"../model-saves/extractors/{dataset_name}/{dataset_name.upper()}-EXTRACTOR/extractor/savefile.hdf5"
        standard_extractor = tf.keras.models.load_model(standard_model_path).layers[0].layers[-1]
        gray_model_path = f"../model-saves/extractors/{dataset_name}/{dataset_name.upper()}-GRAY-EXTRACTOR/extractor/savefile.hdf5"
        gray_extractor = tf.keras.models.load_model(gray_model_path).layers[0].layers[-1]
        blur_model_path = f"../model-saves/extractors/{dataset_name}/{dataset_name.upper()}-BLUR-EXTRACTOR/extractor/savefile.hdf5"
        blur_extractor = tf.keras.models.load_model(blur_model_path).layers[0].layers[-1]
        self._standard_model = tf.keras.Model(inputs=standard_extractor.input, outputs=standard_extractor.layers[1].output)
        self._gray_model = tf.keras.Model(inputs=gray_extractor.input, outputs=gray_extractor.layers[1].output)
        self._blur_model = tf.keras.Model(inputs=blur_extractor.input, outputs=blur_extractor.layers[1].output)

    def generate_feature_maps(self):
        """
        Plots the feature maps from the first convolutional layer of each extractor in the aug model.
        """
        standard_features = self._standard_model.predict(np.expand_dims(preprocess_input(np.copy(self.standard_img)), axis=0))
        blur_features = self._blur_model.predict(np.expand_dims(preprocess_input(np.copy(self.blurred_img)), axis=0))
        gray_features = self._gray_model.predict(np.expand_dims(preprocess_input(np.copy(self.grayed_img)), axis=0))
        features = []
        for i in range(32):
            features.append(standard_features[0,:,:,i])
            features.append(blur_features[0,:,:,i])
            features.append(gray_features[0,:,:,i])
        names = ['standard', 'blur', 'gray']
        fig, ax = plt.subplots(8, 12, constrained_layout=True)
        for i in range(8):
            for j in range(12):
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])
                ax[i,j].imshow(features[12*i+j], cmap='gray')
                ax[i,j].title.set_text(f"kernel {((12*i+j)//3)+1}: {names[(12*i+j)%3]}")
        fig.show()

    def first_pass(self):
        """
        Passes a dummy image through each model to prepare them for fast predictions.
        """
        first_input = np.zeros((1, 299, 299, 3))
        self._model.predict([first_input, first_input, first_input])
        self._standard_model.predict(first_input)
        self._blur_model.predict(first_input)
        self._gray_model.predict(first_input)
        self._inception_model.predict(first_input)

    def initialise_labels(self):
        """
        Creates a labels array in which each label index corresponds to the output of a model's prediction.
        """
        self._labels = ["antelope",   
                        "badger",      
                        "bat",
                        "bear",        
                        "bee",
                        "beetle",      
                        "bison",       
                        "boar",        
                        "butterfly",   
                        "cat",
                        "caterpillar", 
                        "chimpanzee",  
                        "cockroach",   
                        "cow",
                        "coyote",      
                        "crab",        
                        "crow",        
                        "deer",        
                        "dog",
                        "dolphin",     
                        "donkey",      
                        "dragonfly",   
                        "duck",        
                        "eagle",       
                        "elephant",    
                        "flamingo",    
                        "fly",
                        "fox",
                        "goat",        
                        "goldfish",    
                        "goose",       
                        "gorilla",     
                        "grasshopper", 
                        "hamster",     
                        "hare",        
                        "hedgehog",    
                        "hippopotamus",
                        "hornbill",
                        "horse",
                        "hummingbird",
                        "hyena",
                        "jellyfish",
                        "kangaroo",
                        "koala",
                        "ladybugs",
                        "leopard",
                        "lion",
                        "lizard",
                        "lobster",
                        "mosquito",
                        "moth",
                        "mouse",
                        "octopus",
                        "okapi",
                        "orangutan",
                        "otter",
                        "owl",
                        "ox",
                        "oyster",
                        "panda",
                        "parrot",
                        "pelecaniformes",
                        "penguin",
                        "pig",
                        "pigeon",
                        "porcupine",
                        "possum",
                        "raccoon",
                        "rat",
                        "reindeer",
                        "rhinoceros",
                        "sandpiper",
                        "seahorse",
                        "seal",
                        "shark",
                        "sheep",
                        "snake",
                        "sparrow",
                        "squid",
                        "squirrel",
                        "starfish",
                        "swan",
                        "tiger",
                        "turkey",
                        "turtle",
                        "whale",
                        "wolf",
                        "wombat",
                        "woodpecker",
                        "zebra",]

    def prep_image(self, image):
        """
        Augments and preprocesses an image for prediction.
        """
        self.standard_img = image
        self.blurred_img = self._blur_aug(image=image)['image']
        self.grayed_img = self._gray_aug(image=image)['image']
        return [np.expand_dims(preprocess_input(np.copy(self.standard_img)), axis=0), np.expand_dims(preprocess_input(np.copy(self.grayed_img)), axis=0), 
                np.expand_dims(preprocess_input(np.copy(self.blurred_img)), axis=0)]

    def model_predict(self, image, model):
        """
        Uses the selected model to predict the top 5 classes for an image.
        """
        results = model.predict(image)[0]
        values, indices = tf.nn.top_k(results, k=5)
        labels = [(self._labels[indices[i]], values[i]) for i in range(len(indices))]
        return labels
                
    def classify(self, image_path):
        """
        Prepares an image from a selected filename and predicts its class with both models.
        """
        inputs = self.prep_image(img_to_array(load_img(image_path, target_size=(299, 299))))
        labels = self.model_predict(inputs, self._model)
        inception_labels = self.model_predict(inputs[0], self._inception_model)
        return labels, inception_labels
