import tkinter as tk
from tkinter import filedialog as fd
from PIL import ImageTk, Image
from ninety_model import NinetyModel
from carabid_model import CarabidModel


class App():
    def __init__(self, master):
        self._master = master
        self._master.title("Classifier")
        self._master.geometry("700x400")

        # for animal classification, use the ninetymodel; for beetles, use the carabidmodel
        self._model = NinetyModel()
        # self._model = CarabidModel()

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
        """
        Open a file manager to select an image. The selected image is displayed
        on the screen, and the NN model produces its top 3 classification guesses.
        """
        image_path = fd.askopenfilename()
        self._current_img = ImageTk.PhotoImage(Image.open(image_path).resize((299, 299)))
        self._image_canvas.create_image(0, 0, anchor="nw", image=self._current_img)
        results = self._model.classify(image_path)
        result_string = "Predictions:\n\n"
        for label, value in results:
            result_string += f"{label}: {(value.numpy() * 100.):.2f}%\n"
        self._pred_label.config(text=result_string)


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
