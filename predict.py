"""It evaluates a model for the fidelity.
"""
#Dropbox
from client.dropbox import DropboxConnection

#Allow reproducible results
from numpy.random import seed as np_seed
from tensorflow import set_random_seed as tf_seed

#Load objects from the disk
from pandas import read_csv
from keras.models import load_model

#Image data generation
from generation.image import ImageDataGeneration

#Useful constants
from common import constants

#Commandline arguments
from argparse import ArgumentParser

#Path manipulations
from pathlib import Path

#Logging
from common import logging

def parse_args():
    parser = ArgumentParser(description = 'It produces insights on the dataset and the model.')
    parser.add_argument(
        '-m', '--model_name',
        required = True,
        help = 'It specifies the name of the model to evaluate.')
    parser.add_argument(
        '-d', '--dataset',
        default = 'test',
        choices = constants.DATASET_NAMES,
        help = 'It specifies the dataset to use for training.')
    parser.add_argument(
        '-i', '--input_tuples',
        required = True, type = Path,
        help = 'It specifies the path to the input tuples file.')
    parser.add_argument(
        '-p', '--dropbox_parameters',
        nargs = 2,
        help = 'It specifies dropbox parameters required to download the checkpoints.')
    parser.add_argument(
        '-n', '--num_steps',
        type = int,
        help = 'It specifies the number of predict steps.')
    parser.add_argument(
        '-b', '--batch_size',
        default = 32, type = int,
        help = 'It specifies the prediction batch size.')
    parser.add_argument(
        '-c', '--cache_size',
        default = 32, type = int,
        help = 'It specifies the image cache size.')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    #Parse commandline arguments
    args = parse_args()

    model_name = args.model_name
    dataset = args.dataset
    input_tuples = args.input_tuples
    dropbox_parameters = args.dropbox_parameters
    num_steps = args.num_steps
    batch_size = args.batch_size
    cache_size = args.cache_size
    log_to_console = args.log_to_console

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Log input parameters
    logger.info(
                'Running with parameters model_name: %s dataset: %s input_tuples: %s num_steps: %s',
                model_name,
                dataset,
                input_tuples,
                num_steps)

    #Log additional parameters
    logger.info(
                'Additional parameters: batch_size: %d cache_size: %d log_to_console: %s',
                batch_size,
                cache_size,
                log_to_console)

    #Predictable randomness
    seed = 3
    np_seed(seed)
    tf_seed(seed)

    #Required parameters
    input_shape = constants.INPUT_SHAPE
    dataset_loc = constants.DATASET_MAPPINGS[dataset]
    image_cols = constants.INPUT_TUPLE_HEADERS[:2]
    label_col = constants.INPUT_TUPLE_LABEL_COL

    #Dropbox
    dropbox = None

    if dropbox_parameters:
        dropbox_auth_token = dropbox_parameters[0]
        dropbox_remote_dir = constants.DROPBOX_APP_PATH_PREFIX / dropbox_parameters[1]

        dropbox = DropboxConnection(dropbox_auth_token, dropbox_remote_dir)

        logger.info('Dropbox parameters:: dropbox_remote_dir: %s', dropbox_remote_dir)

    #Model file
    model_file = Path("{}.h5".format(model_name))

    #Local file does not exist. Verify if dropbox parameters are provided to enable download.
    if not model_file.exists() and not dropbox:
        raise ValueError("File: {} does not exist locally. Please specify dropbox parameters to download.".format(model_file))

    #Download the file from the dropbox
    if not model_file.exists():
        dropbox.download(model_file)

    #Model file is successfully downloaded
    if not model_file.exists():
        raise ValueError("Model file: {} is not found.".format(model_file))

    #Load the model
    model = load_model(str(model_file))
    logger.info("Loaded model from: {}".format(model_file))

    input_tuples_df = read_csv(input_tuples)

    #Create a data generator to be used for fitting the model.
    datagen = ImageDataGeneration(
                    dataset_loc, input_tuples_df, 
                    input_shape[:2], batch_size,
                    image_cols, label_col,
                    cache_size = cache_size,
                    randomize = False)

    #Training flow
    predict_gen = datagen.flow(subset = 'prediction')

    #Fit the model the input.
    predictions = model.predict_generator(
                            generator = predict_gen,
                            steps = num_steps)

    print(input_tuples_df.loc[0:10, :])
    print(predictions)