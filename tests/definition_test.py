#Unittest
import unittest as ut

#Definition
from model.definition import ModelSpecification, LayerSpecification, LayerType

#Keras
from keras.layers import Input

class TestLayerSpecification(ut.TestCase):
    def test_get_specification_dense(self):
        #Input parameters
        units = 5

        #Act
        inputs = Input(shape = (2, 2))
        layer_specification = LayerSpecification(LayerType.Dense, units)
        layer = layer_specification.get_specification()
        output = layer(inputs)

        #Assert
        self.assertEqual(layer.units, units)
        self.assertEqual(output.get_shape().as_list()[1: ], [2, 5])
        self.assertTrue(layer.name.startswith(LayerSpecification.get_prefix(LayerType.Dense)))

    def test_get_specification_normalization(self):
        #Input parameters
        shape = (2, 2)

        #Act
        inputs = Input(shape = shape)
        layer_specification = LayerSpecification(LayerType.Normalization)
        layer = layer_specification.get_specification()
        output = layer(inputs)

        #Assert
        self.assertEqual(output.get_shape().as_list()[1: ], list(shape))
        self.assertTrue(layer.name.startswith(LayerSpecification.get_prefix(LayerType.Normalization)))

    def test_get_specification_activation(self):
        #Input parameters
        shape = (2, 2)
        activation = 'relu'

        #Act
        inputs = Input(shape = shape)
        layer_specification = LayerSpecification(LayerType.Activation, activation)
        layer = layer_specification.get_specification()
        output = layer(inputs)

        #Assert
        self.assertEqual(output.get_shape().as_list()[1: ], list(shape))
        self.assertTrue(layer.name.startswith(LayerSpecification.get_prefix(LayerType.Activation)))

class TestModelSpecification(ut.TestCase):
    def test_get_specification(self):
        #Input parameters
        inputs = Input(shape = (2, 2))
        layer_specifications = [
                                    LayerSpecification(LayerType.Dense, 5),
                                    LayerSpecification(LayerType.Activation, 'relu')
                                ]

        #Act
        model_specification = ModelSpecification(layer_specifications)
        output = model_specification.get_specification(inputs)

        #Assert
        self.assertEqual(output.get_shape().as_list()[1: ], [2, 5])