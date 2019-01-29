"""It defines model specifications.
"""
#Enum
from enum import Enum, unique

#Keras layers
from keras.layers import Input, Concatenate, Dense, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D

#Keras applications
from keras.applications.resnet50 import ResNet50 as ResNet
from keras.applications import InceptionV3

@unique
class LayerType(Enum):
    Dense = 1
    Normalization = 2
    Dropout = 3
    Activation = 4
    Concatenate = 5
    Inception = 6
    Resnet = 7
    GlobalAveragePooling2D = 8

class LayerSpecification:
    """A container for layer specification.
    """
    layer_prefixes = {
                        LayerType.Dense : (Dense, 'dense_'),
                        LayerType.Normalization : (BatchNormalization, 'batch_normalization_'),
                        LayerType.Dropout : (Dropout, 'dropout_'),
                        LayerType.Activation : (Activation, 'activation_'),
                        LayerType.Concatenate : (Concatenate, 'concatenate_'),
                        LayerType.GlobalAveragePooling2D : (GlobalAveragePooling2D, 'global_average_pool_2d'),
                        LayerType.Inception : (InceptionV3, 'inception_v3'),
                        LayerType.Resnet : (ResNet, 'resnet50')
                    }

    def __init__(self, layer_type, *args, **kwargs):
        #Required parameters
        self._layer_type = layer_type

        self._args = args
        self._kwargs = kwargs

        #Dependent parameters
        self._layer = LayerSpecification.get_layer_object(self._layer_type)

    def get_specification(self):
        #Layer object
        return self._layer(*self._args, **self._kwargs)

    @classmethod
    def get_prefix(cls, layer_type):
        return cls.layer_prefixes[layer_type][1]

    @classmethod
    def get_layer_object(cls, layer_type):
        return cls.layer_prefixes[layer_type][0]

class ModelSpecification:
    """A container for model specifications.
    """
    def __init__(self, layer_specifications):
        self._layer_specifications = layer_specifications

    def get_specification(self, X):
        for specification in self._layer_specifications:
            X = specification.get_specification()(X)

        return X