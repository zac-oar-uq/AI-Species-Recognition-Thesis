# GUI

## classification_gui.py

Python file which runs a Tkinter GUI, allowing users to classify images using a selected backend model.  The GUI also allows a user to graph image predictions, show model feature map outputs and view the augmented images passed to the backend model.

## carabid_model.py & ninety_model.py

Python files which create a useable, trained model for the GUI.  Depending on the chosen model with classification_gui.py, this model can be used for classifying images from the carabid dataset or from the ninety dataset.
