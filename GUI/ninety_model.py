import tensorflow as tf
import albumentations as A
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class NinetyModel():
    def __init__(self):
        self._model = tf.keras.models.load_model("model-saves/multi-aug-classifiers/ninety/NINETY-ENSEMBLE-AUG-CLASSIFIER/classifier/savefile.hdf5")
        self._gray_aug = A.Compose([A.ToGray(p=1.0)])
        self._blur_aug = A.Compose([A.Blur(p=1.0)])
        self.initialise_labels()
        self.first_pass()

    def first_pass(self):
        first_input = np.zeros((1, 299, 299, 3))
        self._model.predict([first_input, first_input, first_input])

    def initialise_labels(self):
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
        standard_img = preprocess_input(image)
        blurred_img = preprocess_input(self._blur_aug(image=image)['image'])
        grayed_img = preprocess_input(self._gray_aug(image=image)['image'])
        return [np.expand_dims(standard_img, axis=0), np.expand_dims(grayed_img, axis=0), 
                np.expand_dims(blurred_img, axis=0)]
                
    def classify(self, image_path):
        inputs = self.prep_image(img_to_array(load_img(image_path, target_size=(299, 299))))
        results = self._model.predict(inputs)[0]
        values, indices = tf.nn.top_k(results, k=3)
        labels = [(self._labels[indices[i]], values[i]) for i in range(len(indices))]
        return labels
