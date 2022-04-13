from tensorflow.keras.applications import InceptionV3
from tensorflow import keras
from tensorflow.keras.layers import *
from prepare_data import *
from datetime import datetime
from tensorflow.keras.applications.inception_v3 import preprocess_input
import albumentations as A

dataset = Dataset.carabid
current_time = datetime.now().strftime("%d%m%Y-%H%M%S")
extractor_train, extractor_val = prep_data_single(dataset, 8)

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.Blur(p=0.5)
])

train_gen, val_gen = prep_data_augmented(dataset, extractor_train, extractor_val, 8, 2, augment=augment)
del extractor_train
del extractor_val

extractor_model_path = "models/carabid/CARABID-EXTRACTOR/extractor/savefile.hdf5"

feature_extractor = keras.models.load_model(extractor_model_path)
inception_model = feature_extractor.layers[0].layers[-1]
inception_model.trainable = False

classifier_model = keras.Sequential([
    InputLayer(input_shape=(None, 299, 299, 3)),
    TimeDistributed(inception_model),
    Bidirectional(LSTM(1000)),
    Dropout(0.5),
    Dense(1000, activation='relu'),
    Dense(train_gen.num_classes(), activation='softmax')
])

classifier_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(classifier_model.summary())

logdir = "logs/unfiltered/{0}_{1}/classifier".format(str(dataset), current_time)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model_path = "models/unfiltered/{0}_{1}/classifier/savefile.hdf5".format(str(dataset), current_time)
model_save_callback = keras.callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

class ShuffleCallback(keras.callbacks.Callback):
    def __init__(self, generator):
        self._generator = generator
        
    def on_epoch_end(self, epoch, logs=None):
        self._generator.shuffle()
    
train_shuffle_callback = ShuffleCallback(train_gen)
val_shuffle_callback = ShuffleCallback(val_gen)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
classifier_model.fit(train_gen, validation_data=val_gen, callbacks=[tensorboard_callback, model_save_callback, train_shuffle_callback, val_shuffle_callback], epochs=20)