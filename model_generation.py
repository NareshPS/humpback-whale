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

#Keras imports
from keras.models import load_model

#Model operation
from model.operation import Operation

#Base models
from model.basemodel import BaseModel

#Inputs
from operation.input_file import ModelInput

#Logger
logger = None
program_actions = ['create', 'update']

def define_common_parameters(parser):
    parser.add_argument(
        '-u', '--num_unfrozen_base_layers',
        type = int, default = 0,
        help = 'It unfreezes the specified number of bottom layers in the base model.')

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

    #Update parameters
    parser.add_argument(
        '--base_level',
        required = True,
        type = int, choices = [1, 2],
        help = 'It specifies the base level depth to update the trainable parameters.')

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
        '--input_shape',
        default = [224, 224, 3],
        type = int, nargs = 3,
        help = 'It specifies the shape of the image input.')
    parser.add_argument(
        '-b', '--base_model_name',
        choices = list(BaseModel.base_models.keys()),
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

    return args.name, args.input_shape, args.base_model_name, args.action, args.action_parameters, args.log_to_console

def create(name, base_model_name, input_shape, args):
    """It creates a model based on the input parameters.
    
    Arguments:
        name {string} -- A string to represent the name of the model.
        base_model_name {string} -- A string to represent the base model to use.
        input_shape {(W, H, C)} -- It specifies the shape of the image input.
        args {An ArgParse object} -- It provides access to the input parameters
    """
    #Input parameters
    dimensions, learning_rate, num_unfrozen_base_layers = args.dimensions, args.learning_rate, args.num_unfrozen_base_layers

    #Log input parameters
    logger.info(
                'Create:: Running with parameters dimensions: %d learning_rate: %f num_unfrozen_base_layers: %d',
                dimensions,
                learning_rate,
                num_unfrozen_base_layers)
    
    #Model function
    model_func = getattr(models, name)

    #Create the model
    model = model_func(base_model_name, input_shape, dimensions, learning_rate, num_unfrozen_base_layers)
    logger.info("Created a new model using base_model_name: {}".format(base_model_name))

    return model

def update(model_file_name, args):
    """It updates the input model based on the input arguments.
    
    Arguments:
        model_file_name {string} -- The name of the model file.
        args {An ArgParse object} -- It provides access to the input parameters
    """

    #Input parameters
    num_unfrozen_base_layers, base_level = args.num_unfrozen_base_layers, args.base_level

    #Log input parameters
    logger.info(
            'Update:: Running with parameters num_unfrozen_base_layers: %d base_level: %d',
            num_unfrozen_base_layers,
            base_level)

    #Load the model
    model = load_model(model_file_name)
    
    #Configure the model
    op = Operation(num_unfrozen_base_layers, configure_base = True, base_level = base_level)
    model = op.configure(model)

    ### Recompile the model for base layer changes ###
    #Extract compile parameters
    loss = model.loss
    optimizer = model.optimizer
    metrics = model.metrics

    #Compile the configured model
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    model.summary()

    return model

def act(action, name, base_model_name, model_file, input_shape, args):
    """It runs the input action with the input arguments.
    
    Arguments:
        action {string} -- A string to represent the name of the action. 
        name {string} -- A string to represent the name of the model.
        base_model_name {string} -- A string to represent the base model to use.
        model_file {string} -- The name of the model file.
        input_shape {(W, H, C)} -- It specifies the shape of the image input.
        args {An object} -- An argument object.
    """
    model = None

    if action == program_actions[0]:
        model = create(name, base_model_name, input_shape, args)
    elif action == program_actions[1]:
        model = update(model_file, args)

    return model

if __name__ == "__main__":
    #Extract command line parameters
    name, input_shape, base_model_name, action, action_parameters, log_to_console = parse_args()

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Log input parameters
    logger.info(
                'Running with parameters name: %s input_shape: %s base_model_name: %s action: %s action_parameters: %s',
                name,
                input_shape,
                base_model_name,
                action,
                action_parameters)

    #Additional parameters
    logger.info('Additional parameters log_to_console: %s', log_to_console)

    #Action parameters
    args = parse_action_parameters(action, action_parameters)

    #Output files
    model_name = "{}_{}".format(name, base_model_name)
    model_input = ModelInput(model_name, 1, 1, 2)

    logger.info('Output files model_file: %s', model_input.file_name())

    #Run action
    model = act(action, name, base_model_name, model_input.file_name(), input_shape, args)

    #Save the trained model.
    model.save(str(model_input.file_name()))
    logger.info("Saved model to: {}".format(model_input.file_name()))