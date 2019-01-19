"""It provides training insights using the history, and weight statistics generated during training.
"""
#Commandline arguments
from argparse import ArgumentParser

#Logging
from common import logging

#Load/Save objects from/to the disk
from pickle import load as pickle_load

#Analysis objects
from analysis.objects import History

#Plotting
from matplotlib import pyplot as plt

#Analysis choices
analysis_choices = ['history']

def parse_args():
    parser = ArgumentParser(description = 'It computes statistics from model and generated objects.')
    parser.add_argument(
        '-m', '--base_model',
        required = True,
        help = 'It specifies a base model to use for computation.')
    parser.add_argument(
        '-t', '--analysis_type',
        choices = analysis_choices,
        nargs = '+',
        required = True,
        help = 'It allows a choice of analysis type.')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')

    args = parser.parse_args()

    return args.base_model, args.analysis_type, args.log_to_console

if __name__ == '__main__':
    base_model, analysis_type, log_to_console = parse_args()

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #History file
    history_file = base_model + ".history"

    logger.info(
            'Running with parameters base_model: %s analysis_type: %s log_to_console: %s',
            base_model,
            analysis_type,
            log_to_console)

    #Placeholder objects
    training_history = None

    with open(history_file, 'rb') as handle:
        training_history = pickle_load(handle)

    logger.info("Loaded history object: %s", history_file)

    history = History(training_history)

    plt.plot(history.accuracy())