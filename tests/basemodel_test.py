#Unittests
import unittest as ut
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

#Operation
from model.basemodel import BaseModel

#Model specifications
from model.definition import LayerSpecification, LayerType

#Keras
from keras.models import load_model

#Test utils
import utils as test_utils

#Constants
from common import constants

#Basic parameters
num_unfrozen_layers = 2

class TestBaseModel(ut.TestCase):
    def test_cnn_invalid_base_model_name(self):
        #Arrange Act & Assert
        with self.assertRaises(ValueError):
            base_model = BaseModel('nanana', test_utils.input_shape, test_utils.dimensions)
            base_model.cnn()

    def cnn_validate_base_model_creation(self, feature_model_name, module_path):
        #Arrange
        base_model = BaseModel(feature_model_name, test_utils.input_shape, test_utils.dimensions)

        #Act & Assert
        with mock_patch(module_path) as resnet_mock:
            base_model._prepare_model = MagicMock()
            base_model._prepare_specification = MagicMock()
            base_model.cnn()
            resnet_mock.assert_called_once()

    def test_cnn_validate_base_model_creation(self):  
        self.cnn_validate_base_model_creation(constants.FEATURE_MODELS[0], 'keras_applications.resnet50.ResNet50')
        self.cnn_validate_base_model_creation(constants.FEATURE_MODELS[1], 'keras_applications.inception_v3.InceptionV3')

    def test_cnn_model(self):
        #Arrange
        base_model = BaseModel(constants.FEATURE_MODELS[1], test_utils.input_shape, test_utils.dimensions)

        #Act
        model = base_model.cnn()

        #Assert
        self.assertIsNotNone(model)
        self.assertTrue(model.layers[-1].name.startswith(LayerSpecification.layer_prefixes[LayerType.Dense][1]))
        self.assertTrue(model.layers[-2].name.startswith(LayerSpecification.layer_prefixes[LayerType.Dense][1]))
        self.assertTrue(model.layers[-3].name.startswith(LayerSpecification.layer_prefixes[LayerType.GlobalAveragePooling2D][1]))