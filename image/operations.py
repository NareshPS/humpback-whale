#Convolution
from scipy.signal import convolve

#Data processing
import numpy as np

class Operations:
    """Operations to process images.
    """
    @staticmethod
    def convolve(images, kernel, mode = 'same'):
        """It convolves the kernel on input images.
        
        Arguments:
            images {[numpy.array]} -- Images as numpy matrices.
            kernel {numpy.array} -- A numpy array to convolute.
            mode {string} -- A string indicating the mode to apply convolution. 
                The values accepted by scipy.signal.convolve() are valid.
        
        Returns:
            {[np.array]} -- Convoluted images.
        """
        input_images = np.asarray(images).astype(np.int16)
        convoluted_images = np.asarray([convolve(image, kernel, mode='same') for image in input_images])
        convoluted_images[convoluted_images > 255] = 255
        convoluted_images[convoluted_images < 0] = 0

        return convoluted_images.astype(np.uint8)