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

#Test support
from tests.support.utils import input_shape, dimensions

#Constants
from common import constants

class TestBaseModel(ut.TestCase):
    def test_invalid_base_model_name(self):
        #Arrange Act & Assert
        with self.assertRaises(ValueError):
            base_model = BaseModel('nanana', input_shape)
            base_model.base_model()

    def validate_base_model_creation(self, feature_model_name, module_path):
        #Arrange
        base_model = BaseModel(feature_model_name, input_shape)

        #Act & Assert
        with mock_patch(module_path) as base_mock:
            base_model.base_model()
            base_mock.assert_called_once()

    def test_validate_base_model_creation(self):
        self.validate_base_model_creation('resnet', 'keras_applications.resnet50.ResNet50')
        self.validate_base_model_creation('inceptionv3', 'keras_applications.inception_v3.InceptionV3')
        self.validate_base_model_creation('mobilenet', 'keras_applications.mobilenet_v2.MobileNetV2')

    def cnn_validate_base_model_creation(self, feature_model_name, module_path):
        #Arrange
        base_model = BaseModel(feature_model_name, input_shape)

        #Act & Assert
        with mock_patch(module_path) as base_mock:
            base_model._prepare_model = MagicMock()
            base_model.cnn(dimensions)
            base_mock.assert_called_once()

    def test_cnn_validate_base_model_creation(self):  
        self.cnn_validate_base_model_creation('resnet', 'keras_applications.resnet50.ResNet50')
        self.cnn_validate_base_model_creation('inceptionv3', 'keras_applications.inception_v3.InceptionV3')
        self.cnn_validate_base_model_creation('mobilenet', 'keras_applications.mobilenet_v2.MobileNetV2')

    def test_cnn_model(self):
        #Arrange
        base_model = BaseModel('inceptionv3', input_shape)

        #Act
        model = base_model.cnn(dimensions)

        #Assert
        self.assertIsNotNone(model)
        self.assertTrue(model.layers[-1].name.startswith(LayerSpecification.layer_prefixes[LayerType.Dense][1]))
        self.assertTrue(model.layers[-2].name.startswith(LayerSpecification.layer_prefixes[LayerType.Dropout][1]))
