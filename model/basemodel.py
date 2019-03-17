#Keras support
from keras.applications.resnet50 import ResNet50 as ResNet
from keras.applications import InceptionV3
from keras.applications import MobileNetV2
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model

#Model specification
from model.definition import ModelSpecification, LayerSpecification, LayerType

#Randomize model names
from random import random

#Constants
from common import constants

#Logging
from common import logging

class BaseModel(object):
    """It defines the base models for the network.
    """
    #Base models
    base_models = dict({
                            'inceptionv3' : InceptionV3,
                            'resnet' : ResNet,
                            'mobilenet' : MobileNetV2
                        })

    def __init__(self, base_model_name, input_shape):
        """It initializes the base model parameters.

        Arguments:
            base_model_name {string} -- A string containing the name of a base model.
            input_shape {(int, int)} -- A tuple indicating the dimensions of model input.
        """
        #Input parameters
        self._base_model_name = base_model_name
        self._input_shape = input_shape

        #Validation
        if BaseModel.base_models.get(base_model_name) is None:
            raise ValueError(
                    'Base model: {} is invalid. Supported models are: {}'.format(
                                                                            base_model_name,
                                                                            BaseModel.base_models.keys()))

        #Logging
        self._logger = logging.get_logger(__name__)

        #Log input parameters
        self._logger.info(
                        "Input parameters:: base_model_name: %s input_shape: %s",
                        self._base_model_name,
                        self._input_shape)

    def base_model(self, name_guidance = 'Model'):
        """It creates a base model object
        """
        #Base model placeholder to be updated in the if/else clause
        base_model_params = dict(include_top = False, weights = None, input_shape = self._input_shape, pooling = 'max')

        #Base model object
        base_model = BaseModel.base_models[self._base_model_name](**base_model_params)

        #Mangle the base model name
        mangled_model_name = '{}_{}_{}_{}'.format(name_guidance, self._base_model_name, self._input_shape[0], int(random() * 1000))
        base_model.name = mangled_model_name

        return base_model

    def cnn(self, dimensions):
        """It creates a base model based on the input parameters

        Arguments:
            dimensions {int} -- An integer indicating the size of feature vector.

        Raises:
            ValueError -- It raise ValueError if the input base model is not supported.

        Returns:
            [A Model object] -- A model object.
        """

        self._logger.info(
                        "Creating a cnn model with base_model: %s dimensions: %d",
                        self._base_model_name,
                        dimensions)

        #Base model
        base_model = self.base_model()

        #Additional layer specification
        kwargs = dict(kernel_initializer = 'he_normal')  

        #Layer specifications
        additional_layer_spec = [
                                    #Dense units
                                    LayerSpecification(LayerType.Dense, 128, activation = 'relu', **kwargs),
                                    LayerSpecification(LayerType.Dropout, 0.5),

                                    #Output unit
                                    LayerSpecification(LayerType.Dense, dimensions, activation = 'softmax', **kwargs)
                                ]

        #Model specification
        model_specification = ModelSpecification(additional_layer_spec)

        #Model
        model = self._prepare_model(base_model, model_specification)

        return model

    def _prepare_model(self, base_model, additional_layers_spec):
        """It creates a model based on the base model and the additional layer specification

        Arguments:
            base_model {A Model object} -- A base model.
            additional_layers_spec {A ModelSpecification object} -- A model specification that defines the attachments to the base model.

        Returns:
            [A Model object] -- A model object.
        """
        #Output
        predictions = additional_layers_spec.get_specification(base_model.output)

        #Model object
        model = Model(inputs = base_model.input, outputs = predictions)

        return model
