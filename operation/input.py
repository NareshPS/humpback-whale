#Path manipulations
from pathlib import Path

class TrainingParameters(object):
    def __init__(self, args):
        """It initializes the training parameters.
        
        Arguments:
            args {An argparse Argument} -- The argparse arguments
        """
        #Required parameters
        self.batch_size = args.batch_size
        self.image_cache_size = args.image_cache_size
        self.number_of_epochs = args.number_of_epochs
        self.learning_rate = args.learning_rate
        self.validation_split = args.validation_split
        self.num_prediction_steps = args.num_prediction_steps
        self.num_fit_images = args.num_fit_images

    def __str__(self):
            return """Parameters::
                        batch_size: {} image_cache_size: {}
                        number_of_epochs: {} learning_rate: {}
                        validation_split: {} num_prediction_steps: {}
                        num_fit_images: {}""".format(
                                                            self.batch_size,
                                                            self.image_cache_size,
                                                            self.number_of_epochs,
                                                            self.learning_rate,
                                                            self.validation_split,
                                                            self.num_prediction_steps,
                                                            self.num_fit_images)

class InputDataParameters(object):
    def __init__(self, args):
        """It initializes the training parameters.
        
        Arguments:
            args {An argparse Argument} -- The argparse arguments
        """
        #Required parameters
        self.model_name = args.model_name
        self.dataset_location = args.dataset_location
        self.input_data = args.input_data
        self.session_id = args.session_id
        self.input_data_training_set_size = args.input_data_training_set_size
        self.input_data_training_set_id = args.input_data_training_set_id
        self.image_cols = args.image_cols
        self.label_col = args.label_col
        self.input_shape = tuple(args.input_shape)

        #Optional parameters
        self.image_transform_cols = args.image_transform_cols

        #Validation
        if self.model_name is None:
            raise ValueError('The model name must be valid')
        
        if self.dataset_location is None or not self.dataset_location.exists():
            raise ValueError('The dataset location must be valid')

    def __str__(self):
            return """Parameters::
                        model_name: {} dataset_location: {}
                        input_data: {} 
                        input_data_training_set_size: {} input_data_training_set_id: {}
                        image_cols: {} label_col: {} image_transform_cols: {}""".format(
                                                                                    self.model_name,
                                                                                    self.dataset_location,
                                                                                    self.input_data,
                                                                                    self.input_data_training_set_size,
                                                                                    self.input_data_training_set_id,
                                                                                    self.image_cols,
                                                                                    self.label_col,
                                                                                    self.image_transform_cols)