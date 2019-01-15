"""It manages the states of the model layer.
"""
#Keras model
from keras.engine.training import Model
from keras.models import load_model

#File manipulation
from os import path

#Pickle objects
from pickle import dump as pickle_dump
from pickle import load as pickle_load

class ModelState:
    """It is a container for a model's trainable state.
    """
    def __init__(self, model):
        """It sets up the model object.
        
        Arguments:
            model {A keras model object} -- A keras model object.
        """
        #Internal parameters
        self._layers = {layer.name: layer.trainable for layer in model.layers if not isinstance(layer, Model)}
        self._sub_models = [ModelState(layer) for layer in model.layers if isinstance(layer, Model)]

    @property
    def layers(self):
        """It is a getter for _layers property
        
        Returns:
            {dict} -- A dictionary of layer names to their trainable state.
        """

        return self._layers

    @property
    def sub_models(self):
        """It is a getter for sub_models property.
        
        Returns:
            [The ModelState objects] -- A list of ModelState objects.
        """

        return self._sub_models

    def save(self, location, filename):
        """It write layer's trainable configuration to the disk.

        Arguments:
            model {A keras model object} -- A keras model object.
            location {string} -- A string representing the output location of the model state.
            filename {string} -- A string representing the name of the output file.
        """
        filepath = path.join(location, filename)

        with open(filepath, 'wb') as handle:
            pickle_dump(self, handle)

    @classmethod
    def load(cls, location, filename):
        """It loads model state from the disk.

        Arguments:
            location {string} -- A string representing the location of the model state.
            filename {string} -- A string representing the name of the model state file.
        """
        filepath = path.join(location, filename)
        model_state = None

        with open(filepath, 'rb') as handle:
            model_state = pickle_load(handle)

        return model_state

    @classmethod
    def load_model(cls, location, model_filename):
        """It creates model state from a model on the disk.

        Arguments:
            location {string} -- A string representing the location of the model file.
            model_filename {string} -- A string representing the name of the model file.
        """
        filepath = path.join(location, model_filename)
        model = load_model(filepath)

        return cls(model)