import tkinter as tk
from tkinter import filedialog as fd
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ninety_model import NinetyModel
from carabid_model import CarabidModel
import matplotlib

matplotlib.rcParams.update({'font.size': 8})


class App():
    def __init__(self, master):
        self._master = master
        self._master.title("Classifier")
        self._master.geometry("800x400")

        # for animal classification, use the ninetymodel; for beetles, use the carabidmodel
        self._model = NinetyModel()
        # self._model = CarabidModel()
        self._results = []
        self._inception_results = []

        self.choose_img_screen()
        
    def choose_img_screen(self):
        # text frame to instruct the selection of an image
        self._text_frame = tk.Frame(self._master)
        self._text_frame.pack(side=tk.TOP)
        self._text_label = tk.Label(self._text_frame, text="Select an image to classify!",
                font=("Calibri", 18))
        self._text_label.pack(side=tk.TOP)

        # image frame to select and display images
        self._image_frame = tk.Frame(self._master)
        self._image_frame.pack(side=tk.LEFT, fill='both', padx=5)
        self._image_canvas = tk.Canvas(self._image_frame, bg='white', height=299, width=299)
        self._image_canvas.pack(side=tk.TOP)
        self._selection_button = tk.Button(self._image_frame, command=self.open_img, text="Select Image", font=("Calibri", 12))
        self._selection_button.pack(side=tk.TOP)

        # frame for info about image (predictions, features, augs, etc.)
        self._info_frame = tk.Frame(self._master)
        self._info_frame.pack(side=tk.LEFT, fill='y')
        self._pred_frame = tk.Frame(self._info_frame)
        self._pred_frame.pack(side=tk.TOP, anchor='w', fill='y')
        self._pred_label = tk.Label(self._pred_frame, text="Predictions:", font=("Calibri", 14), justify='left')
        self._pred_label.pack(side=tk.TOP, padx=10)

        # frame for buttons to choose the information shown on the info frame
        self._buttons_frame = tk.Frame(self._info_frame)
        self._buttons_frame.pack(side=tk.BOTTOM, anchor='w', padx=10)
        self._predictions_button = tk.Button(self._buttons_frame, text="Graph Predictions", font=("Calibri", 12), command=self.show_predictions)
        self._predictions_button.pack(side=tk.LEFT)
        self._features_button = tk.Button(self._buttons_frame, text="Show Extracted Features", font=("Calibri", 12), command=self._model.generate_feature_maps)
        self._features_button.pack(side=tk.LEFT)
        self._augs_button = tk.Button(self._buttons_frame, text="Show Image Augments", font=("Calibri", 12), command=self.show_augs)
        self._augs_button.pack(side=tk.LEFT)

    def show_predictions(self):
        """
        Plots bar graphs for the predictions (both aug model and Inception) of the most recent image.
        """
        fig, ax = plt.subplots(1, 2, constrained_layout=True)
        fig.set_size_inches(10, 5)
        result_x = [i[0] for i in self._results]
        result_height = [i[1].numpy() for i in self._results]
        incep_result_x = [i[0] for i in self._inception_results]
        incep_result_height = [i[1].numpy() for i in self._inception_results]
        ax[0].bar(result_x, result_height)
        ax[0].set_xlabel("Class")
        ax[0].set_ylabel("Probability Prediction")
        ax[0].set_ylim(bottom=0, top=1)
        ax[0].title.set_text("Augment Model Predictions")
        ax[1].bar(incep_result_x, incep_result_height)
        ax[1].set_xlabel("Class")
        ax[1].set_ylabel("Probability Prediction")
        ax[1].set_ylim(bottom=0, top=1)
        ax[1].title.set_text("Inception Model Predictions")
        fig.show()

    def show_augs(self):
        """
        Plots the standard, blurred and greyed versions of the image passed to the aug model.
        """
        fig, ax = plt.subplots(1, 3, constrained_layout=True)
        fig.set_size_inches(12, 5)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].title.set_text("Standard")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].title.set_text("Blurred")
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[2].title.set_text("Greyed")
        ax[0].imshow(self._model.standard_img / 255.0)
        ax[1].imshow(self._model.blurred_img / 255.0)
        ax[2].imshow(self._model.grayed_img / 255.0)
        fig.show()

    def open_img(self):
        """
        Open a file manager to select an image. The selected image is displayed
        on the screen, and the NN model produces its top 3 classification guesses.
        """
        image_path = fd.askopenfilename()
        self._current_img = ImageTk.PhotoImage(Image.open(image_path).resize((299, 299)))
        self._image_canvas.create_image(0, 0, anchor="nw", image=self._current_img)
        self._results, self._inception_results = self._model.classify(image_path)
        result_string = "Predictions:\n"
        for label, value in self._results:
            result_string += f"{label}: {(value.numpy() * 100.):.2f}%\n"
        result_string += "\nInception Results:\n"
        for label, value in self._inception_results:
            result_string += f"{label}: {(value.numpy() * 100.):.2f}%\n"
        self._pred_label.config(text=result_string)


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
