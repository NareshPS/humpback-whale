#Unittests
import unittest as ut

#Constants
from common import ut_constants

#Data processing
import numpy as np

#Local imports
from operation.utils import convolve, imload, imwrite

image_names = ['0000e88ab.jpg', '000f0f2bf.jpg', '3889d6902.jpg']

def get_images(image_size, num_images):
    return [np.random.randint(0, 255, size = image_size, dtype = np.uint8) for _ in range(num_images)]

class TestOperations(ut.TestCase):
    def test_convolve(self):
        #Arrange
        num_images = 2
        image_size = (10, 10)
        images = get_images(image_size, num_images)
        kernel = np.random.randint(-2, 2, size = (3, 3), dtype = np.int16)
        
        #Act
        convoluted_images = convolve(images, kernel, mode = 'same')
        l_input = len(images)
        l_convoluted = len(convoluted_images)

        #Assert
        self.assertEqual(l_input, l_convoluted, "Input images: {} convoluted images: {}".format(l_input, l_convoluted))

    def test_imload(self):
        #Arrange
        source_loc = ut_constants.TRAIN_STORE
        new_size = (100, 100)

        #Act
        resized_images = imload(source_loc, image_names, new_size)

        #Assert
        self.assertEqual(len(image_names), resized_images.shape[0])
        _ = [self.assertTupleEqual(resized_image.shape, new_size + (3, )) for resized_image in resized_images]

    def test_imwrite(self):
        #Arrange
        source_loc = ut_constants.TRAIN_STORE
        images = imload(source_loc, image_names)

        #Arrange target image names to write
        target_image_names = ["{}.1.jpg".format(image_name) for image_name in image_names]

        #Arrange image dictionary
        images_with_name = dict(zip(target_image_names, images))

        #Act
        imwrite(source_loc, images_with_name)

        #Assert
        _ = [self.assertTrue((source_loc / image_name).exists()) for image_name in target_image_names]

        



