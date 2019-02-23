#Path manipulations
from pathlib import Path

def update_params(obj, **kwargs):
    for key, value in kwargs.items():
        setattr(obj, key, value)

class TrainingParameters(object):
    def __init__(self, args):
        """It initializes the training parameters.
        
        Arguments:
            args {An argparse Argument} -- The argparse arguments
        """
        #Required parameters
        self.num_fit_images = args.num_fit_images
        self.number_of_epochs = args.number_of_epochs
        self.learning_rate = args.learning_rate

    def __str__(self):
            return """Parameters::
                        num_fit_images: {}
                        number_of_epochs: {} learning_rate: {}""".format(
                                                                                    self.num_fit_images,
                                                                                    self.number_of_epochs,
                                                                                    self.learning_rate)

class InputParameters(object):
    def __init__(self, args):
        """It initializes the training parameters.
        
        Arguments:
            args {An argparse Argument} -- The argparse arguments
        """
        #Required parameters
        self.model_name = args.model_name
        self.input_data = args.input_data
        self.input_shape = tuple(args.input_shape)

        #Validation
        if self.model_name is None:
            raise ValueError('The model name must be valid')

    def __str__(self):
            return """Parameters::
                        model_name: {} input_data: {} 
                        input_shape: {}""".format(
                                                self.model_name,
                                                self.input_data,
                                                self.input_shape)

class ImageGenerationParameters(object):
    def __init__(self, args):
        """It initializes the training parameters.
        
        Arguments:
            args {An argparse Argument} -- The argparse arguments
        """
        #Required parameters
        self.dataset_location = args.dataset_location
        self.image_cols = args.image_cols
        self.label_col = args.label_col
        self.image_transform_cols = args.image_transform_cols or []
        self.input_shape = tuple(args.input_shape)
        self.validation_split = args.validation_split if hasattr(args, 'validation_split') and args.validation_split is not None else 0
        self.image_cache_size = args.image_cache_size   
        self.batch_size = args.batch_size
        self.num_classes = 2
        self.normalize = True

        #Validation
        if self.dataset_location is None or not self.dataset_location.exists():
            raise ValueError('The dataset location must be valid')

    def __str__(self):
            return """Parameters::
                        dataset_location: {}
                        image_cols: {} label_col: {} image_transform_cols: {}
                        input_shape: {} image_cache_size: {} batch_size: {}
                        validation_split: {}
                        num_classes: {} normalize: {}""".format(
                                                                self.dataset_location,
                                                                self.image_cols,
                                                                self.label_col,
                                                                self.image_transform_cols,
                                                                self.input_shape,
                                                                self.image_cache_size,
                                                                self.batch_size,
                                                                self.validation_split,
                                                                self.num_classes,
                                                                self.normalize)

class SessionParameters(object):
    def __init__(self, args):
        """It initializes the training parameters.
        
        Arguments:
            args {An argparse Argument} -- The argparse arguments
        """
        #Required parameters 
        self.session_id = args.session_id
        self.input_data_training_set_size = args.input_data_training_set_size
        self.input_data_training_set_id = args.input_data_training_set_id
        self.num_df_sets = 1

    def __str__(self):
            return """Parameters::
                        session_id: {}
                        input_data_training_set_size: {} input_data_training_set_id: {}
                        num_df_sets: {}""".format(
                                                self.session_id,
                                                self.input_data_training_set_size,
                                                self.input_data_training_set_id,
                                                self.num_df_sets)