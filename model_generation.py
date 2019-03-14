"""It creates customized models based on input parameters.
"""
#Logging
from common import logging

#Constants
from common import constants

#Argument parsing
from argparse import ArgumentParser

#Keras imports
from keras.models import load_model

#Model imports
from model.operation import Operation
from model.basemodel import BaseModel
from model import models

#Input parameters
from operation.input import TrainingParameters

#Inputs
from iofiles.input_file import ModelInput

#Logger
logger = None

def parse_args():
    parser = ArgumentParser(description = 'It creates a keras model.')

    parser.add_argument(
        '-n', '--name',
        required = True,
        help = 'It specifies the name of the model.')
    parser.add_argument(
        '--input_shape',
        default = [224, 224, 3],
        type = int, nargs = 3,
        help = 'It specifies the shape of the image input.')
    parser.add_argument(
        '-b', '--base_model_name',
        choices = list(BaseModel.base_models.keys()),
        help = 'It specifies a base model to use for the models.')
    parser.add_argument(
        '-d', '--dimensions',
        default = 90, type = int,
        help = 'It specifies the number of dimensions to output from base model.')
    parser.add_argument(
        '-r', '--learning_rate',
        default = 0.001, type = float,
        help = 'It specifies the learning rate of the optimization algorithm. It must be a float between 0 and 1')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')

    args = parser.parse_args()

    return args.name, args.input_shape, args.base_model_name, args.dimensions, args.learning_rate, args.log_to_console

if __name__ == "__main__":
    #Extract command line parameters
    name, input_shape, base_model_name, dimensions, learning_rate, log_to_console = parse_args()

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Log input parameters
    logger.info('Running with parameters:: %s input_shape: %s base_model_name: %s', name, input_shape, base_model_name)
    logger.info('Additional parameters:: dimensions: %d learning_rate: %f, log_to_console: %s', dimensions, learning_rate, log_to_console)

    #Output files
    model_name = "{}_{}".format(name, base_model_name)
    model_input = ModelInput(model_name)
    model_file_name = model_input.file_name(0, 0)

    logger.info('Output files model_file: %s', model_file_name)

    #Model function
    model_func = getattr(models, name)
    logger.info('Using model_func: %s', model_func)

    #Create the model
    model = model_func(base_model_name, input_shape, dimensions, learning_rate)
    logger.info("Created a new model using base_model_name: {}".format(base_model_name))

    #Save the trained model.
    model.save(str(model_file_name))
    print("Saved model to: {}".format(model_file_name))
