"""It augments the input dataset with random and intentional transformations.
"""

class ImageAugmentation(object):
    class Parameters(object):
        """It contains transformation parameters.
        """
        def __init__(self, transformation_params, num_output_images):
            """It initializes the augmentation parameters.
            
            Arguments:
                transformation_params {A ImageDataTransformation.Parameters object} -- The transformation parameters.
                num_output_images {int} -- The number of transformed images.
            """
            #Required parameters
            self.transformation_params = transformation_params
            self.num_output_images = num_output_images

        def __str__(self):
                return """Parameters::
                            transformation_params: {}
                            num_output_images: {}""".format(
                                                            self.transformation_params,
                                                            self.num_output_images)

    ### ImageAugmentation members ###    
    def __init__(self, augmentation_params):
        """It initializes the augmentation parameters
        
        Arguments:
            augmentation_params {[A ImageAugmentation.Parameters object]} -- The list of augmentation parameters.
        """
        self._augmentation_params = augmentation_params

        #


