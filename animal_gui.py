import tkinter as tk
from tkinter import filedialog as fd
from PIL import ImageTk, Image
import tensorflow as tf
import albumentations as A
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from os import listdir
from os.path import isfile


class App():
    def __init__(self, master):
        self._master = master
        self._master.title("Animal Classifier")
        self._master.geometry("500x400")
        self._model = Model()
        self.choose_img_screen()
        
    def choose_img_screen(self):
        # text frame to instruct the selection of an image
        self._text_frame = tk.Frame(self._master)
        self._text_frame.pack(side=tk.TOP)
        self._text_label = tk.Label(self._text_frame, text="Select an animal image to classify!",
                font=("Calibri", 18))
        self._text_label.pack(side=tk.TOP)

        # image frame to select and display images
        self._image_frame = tk.Frame(self._master)
        self._image_frame.pack(side=tk.LEFT, fill='y')
        self._image_canvas = tk.Canvas(self._image_frame, bg='white', height=299, width=299)
        self._image_canvas.pack(side=tk.TOP)
        self._selection_button = tk.Button(self._image_frame, command=self.open_img, text="Select Image", font=("Calibri", 12))
        self._selection_button.pack(side=tk.TOP)

        # predictions frame to display classification predictions
        self._pred_frame = tk.Frame(self._master)
        self._pred_frame.pack(side=tk.LEFT, fill='y')
        self._pred_label = tk.Label(self._pred_frame, text="Predictions:", font=("Calibri", 14), justify='left')
        self._pred_label.pack(side=tk.TOP, padx=10)

    def open_img(self):
        image_path = fd.askopenfilename()
        self._current_img = ImageTk.PhotoImage(Image.open(image_path).resize((299, 299)))
        self._image_canvas.create_image(0, 0, anchor="nw", image=self._current_img)
        results = self._model.classify(image_path)
        result_string = "Predictions:\n\n"
        for label, value in results:
            result_string += f"{label}: {(value.numpy() * 100.):.2f}%\n"
        self._pred_label.config(text=result_string)


class Model():
    def __init__(self):
        self._model = tf.keras.models.load_model("models/ninety/NINETY-NORMAL-GRAY-BLUR-CLASSIFIER/classifier/savefile.hdf5")
        self._gray_aug = A.Compose([A.ToGray(p=1.0)])
        self._blur_aug = A.Compose([A.Blur(p=1.0)])
        self.initialise_labels()
        self.first_pass()

    def first_pass(self):
        first_input = np.zeros((1, 299, 299, 3))
        self._model.predict([first_input, first_input, first_input])

    def initialise_labels(self):
        self._labels = []
        for label in listdir("datasets/90 animal dataset/animals/animals"):
            if not isfile(label):
                self._labels.append(label)

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


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()