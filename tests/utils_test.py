#Unittests
import unittest as ut

#Data processing
import numpy as np

#Local imports
from operation.utils import convolve

class TestOperations(ut.TestCase):
    def test_convolve(self):
        image_size = (10, 10)
        images = [
                    np.random.randint(0, 255, size = image_size, dtype = np.uint8),
                    np.random.randint(0, 255, size = image_size, dtype = np.uint8)
                 ]
        kernel = np.random.randint(-2, 2, size = (3, 3), dtype = np.int16)
        convoluted_images = convolve(images, kernel, mode = 'same')
        l_input = len(images)
        l_convoluted = len(convoluted_images)

        self.assertEqual(l_input, l_convoluted, "Input images: {} convoluted images: {}".format(l_input, l_convoluted))