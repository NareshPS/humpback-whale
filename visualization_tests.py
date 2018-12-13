
#Command line parameters
from sys import argv

#Load training history
from pickle import load as pickle_load

#Load keras model
from keras.models import load_model

#Local imports
from visualization import HistoryInsights

if __name__ == "__main__":
    n_args = len(argv)

    if n_args != 2:
        print("Syntax error. Usage:: python visualization_tests.py <model_name>") 
        exit(-1)

    model_name = argv[1]
    history_file = "{model_name}.hist".format(model_name = model_name)
    model_file = "{model_name}.h5".format(model_name = model_name)

    print("Using model: {model_file} history: {history_file}".format(model_file = model_file, history_file = history_file))

    model = load_model(model_file)

    history = None
    with open(history_file, 'rb') as handle:
        history = pickle_load(handle)
        
    insights = HistoryInsights(history)
    insights.accuracy()
    insights.loss()