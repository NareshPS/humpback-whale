#Keras support
from keras.applications.resnet50 import ResNet50 as ResNet
from keras.applications import InceptionV3
from keras.applications import MobileNetV2
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model

#Model specification
from model.definition import ModelSpecification, LayerSpecification, LayerType

#Model operation
from model.operation import Operation

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
    
    def __init__(self, base_model_name, input_shape, dimensions, num_unfrozen_base_layers = 0):
        """It initializes the base model parameters.

        Arguments:
            base_model_name {string} -- A string containing the name of a base model.
            input_shape {(int, int)} -- A tuple indicating the dimensions of model input.
            dimensions {int} -- An integer indicating the size of feature vector.
            num_unfrozen_base_layers {int} -- The number of bottom layers of base model to train.
        """
        #Input parameters
        self._base_model_name = base_model_name
        self._input_shape = input_shape
        self._dimensions = dimensions
        self._num_unfrozen_base_layers = num_unfrozen_base_layers

        #Derived parameters
        self._operation = Operation(self._num_unfrozen_base_layers, configure_base = False)

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
                        "Input parameters:: base_model_name: %s input_shape: %s dimensions: %d num_unfrozen_base_layers: %d",
                        self._base_model_name,
                        self._input_shape,
                        self._dimensions,
                        self._num_unfrozen_base_layers)

    def _get_base_model(self):
        """It creates a base model object
        """
        #Base model placeholder to be updated in the if/else clause
        base_model_params = dict(include_top=False, weights='imagenet', input_shape = self._input_shape, pooling = 'max')

        #Base model object
        base_model = BaseModel.base_models[self._base_model_name](**base_model_params)

        return base_model

    def cnn(self):
        """It creates a base model based on the input parameters
        
        Raises:
            ValueError -- It raise ValueError if the input base model is not supported.
        
        Returns:
            [A Model object] -- A model object.
        """

        self._logger.info("Creating a cnn model with base_model: %s", self._base_model_name)

        #Base model
        base_model = self._get_base_model()

        #Unfreeze base layers
        base_model = self._operation.configure(base_model)

        #Additional layer specification
        additional_layer_spec = self._prepare_specification()

        #Model
        model = self._prepare_model(base_model, additional_layer_spec)

        return model

    def _prepare_model(self, base_model, additional_layers_spec):
        """It creates a model based on the base model and the additional layer specification
        
        Arguments:
            base_model {A Model object} -- A base model.
            additional_layers_spec {A ModelSpecification object} -- A model specification that defines the attachments to the base model.

        Returns:
            [A Model object] -- A model object.
        """
        #Model placeholder
        model = base_model
        
        #Output
        predictions = additional_layers_spec.get_specification(base_model.output)

        #Model object
        model = Model(inputs = base_model.input, outputs = predictions)

        return model

    def _prepare_specification(self):
        """It creates a model specification to add to the base model.
        
        Returns:
            [A ModelSpecification object] -- A model specification objects that defines attachments to the base model.
        """
        #Layer args
        kwargs = dict(kernel_initializer = 'he_normal')

        #Layer specifications
        layer_specifications = [
                                    #Dense units
                                    LayerSpecification(LayerType.Dense, 128, activation = 'relu', **kwargs),
                                    LayerSpecification(LayerType.Dropout, 0.5),

                                    #Output unit
                                    LayerSpecification(LayerType.Dense, self._dimensions, activation = 'softmax', **kwargs)
                                ]

        #Model specification
        model_specification = ModelSpecification(layer_specifications)

        return model_specification