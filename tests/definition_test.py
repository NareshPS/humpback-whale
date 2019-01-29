#Unittest
import unittest as ut

#Definition
from model.definition import ModelSpecification, LayerSpecification, LayerType

#Keras
from keras.layers import Input

class TestLayerSpecification(ut.TestCase):
    def get_specification(self, layer_type, parameter, results):
        #Input parameters
        shape = (2, 2)

        #Act
        inputs = Input(shape = shape)
        layer_specification = LayerSpecification(layer_type, parameter) if parameter is not None else LayerSpecification(layer_type)
        layer = layer_specification.get_specification()
        output = layer(inputs)

        #Assert
        self.assertEqual(output.get_shape().as_list()[1: ], results)
        self.assertTrue(layer.name.startswith(LayerSpecification.get_prefix(layer_type)))

    def test_get_specification_dense(self):
        self.get_specification(LayerType.Dense, 5, [2, 5])

    def test_get_specification_normalization(self):
        self.get_specification(LayerType.Normalization, None, [2, 2])

    def test_get_specification_activation(self):
        self.get_specification(LayerType.Activation, 'relu', [2, 2])

    def test_get_specification_dropout(self):
        self.get_specification(LayerType.Dropout, 0.5, [2, 2])

    def test_get_specification_concatenate(self):
        #Input parameters
        shape = (2, 2)

        #Act
        inputs = [Input(shape = shape), Input(shape = shape)]
        layer_specification = LayerSpecification(LayerType.Concatenate)
        layer = layer_specification.get_specification()
        output = layer(inputs)

        #Assert
        self.assertEqual(output.get_shape().as_list()[1: ], [2, 4])
        self.assertTrue(layer.name.startswith(LayerSpecification.get_prefix(LayerType.Concatenate)))

    def test_get_specification_resnet(self):
        #Input parameters
        shape = (2, 2, 3)
        resnet_params = dict(include_top=False, weights = None, input_shape = shape)

        #Act
        inputs = Input(shape = shape)
        layer_specification = LayerSpecification(LayerType.Resnet, resnet_params)
        layer = layer_specification.get_specification()
        output = layer(inputs)

        #Assert
        self.assertEqual(output.get_shape().as_list()[1: ], [1000])
        self.assertTrue(layer.name.startswith(LayerSpecification.get_prefix(LayerType.Resnet)))

    def test_get_specification_inception(self):
        #Input parameters
        shape = (100, 100, 3)
        inception_params = dict(include_top=False, weights = None, input_shape = shape)

        #Act
        inputs = Input(shape = shape)
        layer_specification = LayerSpecification(LayerType.Inception, inception_params)
        layer = layer_specification.get_specification()
        output = layer(inputs)

        #Assert
        print(layer.name)
        self.assertEqual(output.get_shape().as_list()[1: ], [1000])
        self.assertTrue(layer.name.startswith(LayerSpecification.get_prefix(LayerType.Inception)))

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