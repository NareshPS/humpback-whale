#Unittests
import unittest as ut

#Operation
from model.operation import Operation

#Test utils
from utils import load_test_model

#Basic parameters
num_unfrozen_layers = 2

class TestOperation(ut.TestCase):
    def test_init_success(self):
        #Assert default construction
        _ = Operation(num_unfrozen_layers)

        #Assert configure base and its depth
        base_level = 2
        op = Operation(num_unfrozen_layers, configure_base = True, base_level = base_level)
        self.assertEqual(op._configure_base, True)
        self.assertEqual(op._base_level, base_level)

    def init_level(self, base_level, expect_expection):
        #Assert
        if expect_expection:
            with self.assertRaises(ValueError):
                _ = Operation(num_unfrozen_layers, configure_base = True, base_level = base_level)
        else:
            _ = Operation(num_unfrozen_layers, configure_base = True, base_level = base_level)

    def test_init_raises(self):
        #Lower than lower limit asserts
        self.init_level(-1, True)

        #Valid values
        self.init_level(0, False)
        self.init_level(2, False)

        #Higher than the allowed limit asserts
        self.init_level(3, True)

    def verify_unfrozen_layers(self, model, num_unfrozen_layers):
        #Arrange
        expected_results = [False]*(len(model.layers) - num_unfrozen_layers) + [True]*num_unfrozen_layers
        results = [layer.trainable for layer in model.layers]

        #Assert
        self.assertListEqual(expected_results, results)

    def test_configure_model(self):
        #Arrange
        op = Operation(num_unfrozen_layers)
        model = load_test_model()

        #Act
        model = op.configure(model)

        #Assert
        self.verify_unfrozen_layers(model, num_unfrozen_layers)

    def test_configure_base_model(self):
        #Arrange
        op = Operation(num_unfrozen_layers, configure_base = True, base_level = 0)
        model = load_test_model()

        #Act
        model = op.configure(model)

        #Assert
        self.verify_unfrozen_layers(model, num_unfrozen_layers)