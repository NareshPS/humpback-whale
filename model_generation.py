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

#Logger
logger = None
program_actions = ['create', 'update']

def define_common_parameters(parser):
    parser.add_argument(
        '-t', '--train_all',
        action = 'store_true', default = False,
        help = 'It enables training the base model.')

    return parser

def parse_create_parameters(params):
    """It parses create parameters.
    
    Arguments:
        params {[string]} -- A list of the action parameters.
    """
    parser = ArgumentParser()
    
    parser.add_argument(
        '-d', '--dimensions',
        default = 90, type = int,
        help = 'It specifies the number of dimensions to output from base model.')
    parser.add_argument(
        '-r', '--learning_rate',
        default = 0.001, type = float,
        help = 'It specifies the learning rate of the optimization algorithm. It must be a float between 0 and 1')

    #Append common parameters
    parser = define_common_parameters(parser)

    return parser.parse_args(params)

def parse_update_parameters(params):
    """It parses update parameters.
    
    Arguments:
        params {[string]} -- A list of the action parameters.
    """
    
    parser = ArgumentParser()

    #Append common parameters
    parser = define_common_parameters(parser)

    return parser.parse_args(params)

def parse_action_parameters(action, raw_params):
    """It parses action specific input arguments.
    
    Arguments:
        action {string} -- A string to represent the name of the action.
        raw_params {string} -- A string containing the action parameters.
    """
    args = None
    params = raw_params.split() if raw_params is not None else []

    if action == program_actions[0]:
        args = parse_create_parameters(params)
    elif action == program_actions[1]:
        args = parse_update_parameters(params)

    return args

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
        '-a', '--action',
        choices = program_actions, default = program_actions[0],
        help = 'It specifies the action to be performed.')
    parser.add_argument(
        '-p', '--action_parameters',
        help = 'It specifies action specific parameters.')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')

    args = parser.parse_args()

    return args.name, args.base_model, args.action, args.action_parameters, args.log_to_console

def create(name, base_model, args):
    #Input parameters
    dimensions, learning_rate, train_all = args.dimensions, args.learning_rate, args.train_all
    
    #Model function
    model_func = getattr(models, name)

    #Required parameters
    input_shape = constants.INPUT_SHAPE

    #Create the model
    model = model_func(base_model, input_shape, dimensions, learning_rate, train_all)
    logger.info("Created a new model using base_model: {}".format(base_model))

    return model

def update(args):
    pass

def act(action, name, base_model, args):
    """It runs the input action with the input arguments.
    
    Arguments:
        name {string} -- A string to represent the name of the model.
        base_model {string} -- A string to represent the base model to use.
        action {string} -- A string to represent the name of the action. 
        args {An object} -- An argument object.
    """
    model = None

    if action == program_actions[0]:
        model = create(name, base_model, args)
    elif action == program_actions[1]:
        model = update(args)

    return model

if __name__ == "__main__":
    #Extract command line parameters
    name, base_model, action, action_parameters, log_to_console = parse_args()

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Log input parameters
    logger.info(
                'Running with parameters name: %s base_model: %s action: %s action_parameters: %s',
                name,
                base_model,
                action,
                action_parameters)

    #Additional parameters
    logger.info('Additional parameters log_to_console: %s', log_to_console)

    #Action parameters
    args = parse_action_parameters(action, action_parameters)

    #Output files
    model_file = "{}_{}.h5".format(name, base_model)
    model_state_file = "{}_{}.model_state".format(name, base_model)

    logger.info(
                'Output files model_file: %s model_state_file: %s',
                model_file,
                model_state_file)

    #Run action
    model = act(action, name, base_model, args)

    #Write model trainable state
    model_state = ModelState(model)
    model_state.save(".", model_state_file)
    logger.info("Saved model state to: {}".format(model_state_file))

    #Save the trained model.
    model.save(model_file)
    logger.info("Saved model to: {}".format(model_file))