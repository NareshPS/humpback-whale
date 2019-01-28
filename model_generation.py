"""It creates customized models based on input parameters.
"""
#Logging
from common import logging

#Constants
from common import constants

#Argument parsing
from argparse import ArgumentParser

#Models
from model import models

#Load/Save model states
from model.state import ModelState

def parse_args():
    parser = ArgumentParser(description = 'It trains a siamese network for whale identification.')
    parser.add_argument(
        '-n', '--name',
        required = True,
        help = 'It specifies the name of the model.')
    parser.add_argument(
        '-b', '--base_model',
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
    parser.add_argument(
        '-a', '--train_all',
        action = 'store_true', default = False,
        help = 'It enables training the base model.')

    args = parser.parse_args()

    return args.name, args.base_model, args.dimensions, args.learning_rate, args.log_to_console, args.train_all

def process_create():
    pass

if __name__ == "__main__":
    #Extract command line parameters
    name, base_model, dimensions, learning_rate, log_to_console, train_all = parse_args()

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Log input parameters
    logger.info(
                'Running with parameters name: %s base_model: %s dimensions: %d',
                name,
                base_model,
                dimensions)

    #Additional parameters
    logger.info(
                'Additional parameters learning_rate: %d log_to_console: %s train_all: %s',
                learning_rate,
                log_to_console,
                train_all)

    #Model function
    model_func = getattr(models, name)

    #Output files
    model_file = "{}_{}.h5".format(name, base_model)
    model_state_file = "{}_{}.model_state".format(name, base_model)

    logger.info(
                'Output files model_file: %s model_state_file: %s',
                model_file,
                model_state_file)

    #Required parameters
    input_shape = constants.INPUT_SHAPE

    #Create the model
    model = model_func(base_model, input_shape, dimensions, learning_rate, train_all)
    logger.info("Created a new model using base_model: {}".format(base_model))

    #Write model trainable state
    model_state = ModelState(model)
    model_state.save(".", model_state_file)
    logger.info("Saved model state to: {}".format(model_state_file))

    #Save the trained model.
    model.save(model_file)
    logger.info("Saved model to: {}".format(model_file))