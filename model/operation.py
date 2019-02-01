#Model type
from keras.engine.training import Model as Type_Model

#Logging
from common import logging

class Operation(object):
    """It defines useful methods to operate on the model objects
    """
    #Set the maximum base level for validation
    max_base_level = 2

    def __init__(self, num_unfrozen_layers, configure_base = False, base_level = 1):
        """It initializes the input parametes.
        
        Arguments:
            num_unfrozen_layers {int} -- The number of bottom layers to unfreeze for training
            configure_base {boolean} -- It indicates if the model or the base models are configuration candidates
            base_model_level {int} -- It indicates the level of the base models.
        """
        #Input parameters
        self._num_unfrozen_layers = num_unfrozen_layers
        self._configure_base = configure_base
        self._base_level = base_level
        
        #Logging
        self._logger = logging.get_logger(__name__)

        #Validation
        if self._base_level < 0 or self._base_level > self.max_base_level:
            raise ValueError("The valid values for base level are [0, {}]".format(self.max_base_level))

        #Log input parameters
        self._logger.info(
                        "Input parameters:: num_unfrozen_layers: %d configure_base: %s base_level: %d",
                        self._num_unfrozen_layers,
                        self._configure_base,
                        self._base_level)

    def configure(self, model):
        """It selects and applies the appropiate configuration behavior.

        Arguments:
            model {A Model object} -- The configuration candidate

        Returns:
            [A Model object] -- A model object.
        """
        #Configured model placeholder
        configured_model = None

        #Configure
        if self._configure_base:
            configured_model = self._configure_base_model_layers(model)
            self._logger.info("Configured base models of the model: %s", model.name)
        else:
            configured_model = self._configure_layers(model)
            self._logger.info("Configured the model: %s", model.name)

        return configured_model

    def _configure_base_model_layers(self, model):
        """It configures the base models of the input model

        Arguments:
            model {A Model object} -- The configuration candidate

        Returns:
            [A Model object] -- A model object.
        """
        #Set the current dept level
        current_depth = 0
        current_depth_models = [model]

        #Find the base models at the input base level.
        while current_depth < self._base_level:
            current_depth_models = [layer for input_model in current_depth_models for layer in input_model.layers if type(layer) is Type_Model ]

            #Log status
            self._logger.info(
                            'Current depth: %d num model layers: %d %s',
                            current_depth,
                            len(current_depth_models),
                            current_depth_models)

            #No more model layers to be iterated
            if len(current_depth_models) == 0:
                break

            #Move to the next level
            current_depth += 1

        #Notify if no base model found at the intended level.
        if len(current_depth_models) == 0:
            raise ValueError('No configurable base model at level: {}'.format(self._base_level))

        #Configure the layers
        for layer in current_depth_models:
            self._configure_layers(layer)

        return model

    def _configure_layers(self, model):
        """It configures the model layers

        Arguments:
            model {A Model object} -- The configuration candidate

        Returns:
            [A Model object] -- A model object.
        """
        #Count layers
        num_layers = len(model.layers)

        #Freeze all layers except the ones indicated to stay unfrozen.
        for layer in model.layers[:num_layers - self._num_unfrozen_layers]:
            layer.trainable = False

        for layer in model.layers[num_layers - self._num_unfrozen_layers:]:
            layer.trainable = True

            #Log the name
            self._logger.info("Trainable layer: %s", layer.name)

        return model