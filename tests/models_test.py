#Unittest
import unittest as ut

from model.models import ModelParameters

all_valid_parameters = {
                        'n_classes': 5,
                        'l_rate': 0.001
                        }

valid_parameters = {'n_classes': 5}
invalid_parameters = {'x_classes': 5}
mixed_parameters = {
                        'n_classes': 5,
                        'x_classes': 0.001
                        }

class TestModelParameters(ut.TestCase):
    def test_all_valid_parameters(self):
        ModelParameters(all_valid_parameters)

    def test_valid_parameters(self):
        ModelParameters(valid_parameters)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            ModelParameters(invalid_parameters)

    def test_mixed_parameters(self):
        with self.assertRaises(ValueError):
            ModelParameters(mixed_parameters)

if __name__ == "__main__":
    ut.main()
