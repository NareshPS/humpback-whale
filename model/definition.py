"""It defines model specifications.
"""
#Enum
from enum import Enum, unique

#Keras layers
from keras.layers import Input, Concatenate, Dense, BatchNormalization, Activation

@unique
class LayerType(Enum):
    Dense = 1
    Normalization = 2
    Dropout = 3
    Activation = 4
    Concatenate = 5

class LayerSpecification:
    """A container for layer specification.
    """
    layer_prefixes = {
                        LayerType.Dense : 'dense_',
                        LayerType.Normalization : 'batch_normalization_',
                        LayerType.Dropout : 'dropout_',
                        LayerType.Activation : 'activation_',
                        LayerType.Concatenate : 'concatenate_'
                    }

    def __init__(self, layer_type, *args, **kwargs):
        self._layer_type = layer_type
        self._args = args
        self._kwargs = kwargs

    def get_specification(self):
        #Dense layer
        if self._layer_type is LayerType.Dense:
            return Dense(*self._args, **self._kwargs)
        #Batch normalization
        elif self._layer_type is LayerType.Normalization:
            return BatchNormalization()
        #Activation
        elif self._layer_type == LayerType.Activation:
            return Activation(*self._args)
        #Concatenation
        elif self._layer_type == LayerType.Concatenate:
            return Concatenate(*self._args)

    @classmethod
    def get_prefix(cls, layer_type):
        return cls.layer_prefixes[layer_type]

class ModelSpecification:
    """A container for model specifications.
    """
    def __init__(self, layer_specifications):
        self._layer_specifications = layer_specifications

    def get_specification(self, X):
        for specification in self._layer_specifications:
            X = specification.get_specification()(X)

        return X