#Unittest
import unittest as ut

#Constants
from common import ut_constants

#Path manipulations
from os import path

#Test class
from model.state import ModelState

model_filename = 'cnn_model2d_1.h5'
model_state_output_file = 'model_state_output'
model_state_input_file = 'model_state_input'
model_layers_state = {
                        'conv2d_1': True,
                        'conv2d_2': True,
                        'max_pooling2d_1': True,
                        'dropout_1': True,
                        'conv2d_3': True,
                        'conv2d_4': True,
                        'max_pooling2d_2': True,
                        'dropout_2': True,
                        'flatten_1': True,
                        'dense_1': True,
                        'dropout_3': True,
                        'dense_2': True
                        }

class TestModelState(ut.TestCase):
    def test_load_model(self):
        model_state = ModelState.load_model(ut_constants.UT_DATA_STORE, model_filename)
        self.assertDictEqual(model_state.layers, model_layers_state)
        self.assertListEqual(model_state.sub_models, [])

    def test_save(self):
        model_state = ModelState.load_model(ut_constants.UT_DATA_STORE, model_filename)
        model_state.save(ut_constants.UT_DATA_STORE, model_state_output_file)
        model_state_output_path = path.join(ut_constants.UT_DATA_STORE, model_state_output_file)
        
        self.assertTrue(path.exists(model_state_output_path))

    def test_load(self):
        model_state = ModelState.load(ut_constants.UT_DATA_STORE, model_state_input_file)
        self.assertDictEqual(model_state.layers, model_layers_state)
        self.assertListEqual(model_state.sub_models, [])

if __name__ == "__main__":
    ut.main()
