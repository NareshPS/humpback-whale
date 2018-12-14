"""Trains a model and save it to the disk.

    Usage:: python train.py <output_model_name> <batch_size> <num_epochs> <learning_rate> <validation_split> <[int] Size of training set. If not specified, process all.>
"""
#Basic imports
from sys import argv, stdout

#To save training data
from pickle import dump as pickle_dump

#Local imports
from common import constants
from models import cnn_gray_model_1
from model_utils import get_input_labels, get_label_ids, model_fit
from utils import list_files, split_dataset

if __name__ == "__main__":
    """Entry point to train the model.

    Usage:: python train.py <output_model_name> <batch_size> <num_epochs> <learning_rate> <validation_split)> <[int] Size of training set. If not specified, process all.>
    """
    n_args = len(argv)
    if n_args not in [5, 7]:
        print("Syntax error. Usage:: python train.py <output_model_name> <batch_size> <num_epochs> <learning_rate> <validation_split> <[int] Size of training set. If not specified, process all.>") 
        print("Example:: python train.py \"model_1\" 16 5 0.001 0.2 64")
        print("Example:: python train.py \"model_1\" 16 5 0.001")
        exit(-1)
    
    #Input parameters
    o_model = argv[1]
    batch_size = int(argv[2])
    n_epochs = int(argv[3])
    l_rate = float(argv[4])
    v_split = float(argv[5]) if len(argv) == 7 else 0.2
    n_images = int(argv[6]) if len(argv) == 7 else None

    #Fixed parameter
    dataset = "train"
    input_shape = constants.IMG_SHAPE
    source_loc = constants.PROCESSED_DATASET_MAPPINGS[dataset]
    
    #Input data parameters
    input_labels = get_input_labels()
    label_ids = get_label_ids()
    n_classes = len(label_ids)
    input_set = list_files(source_loc, n_images)
    n_images = len(input_set)

    #Training and validation sets
    train_set, validation_set = split_dataset(input_set, v_split)
    print("Training set: {t_size} validation set: {v_size}".format(t_size = len(train_set), v_size = len(validation_set)))

    #Initialize the model
    model = cnn_gray_model_1(input_shape, n_classes, l_rate)
    model.summary()

    #load_training_batch(requestor, source_loc, img_files, batch_size, input_labels, label_ids)

    #Train the model
    history = model_fit(model, source_loc, train_set, validation_set, input_labels, label_ids, batch_size, n_epochs)

    #Output model file
    model_file = "{o_model}.h5".format(o_model = o_model)
    history_file = "{o_model}.hist".format(o_model = o_model)

    #Save model data
    print("Saving model: {model_file} and history: {history_file}".format(model_file = model_file, history_file = history_file))
    model.save(model_file, overwrite=True)

    with open(history_file, 'wb') as handle:
        pickle_dump(history.history, handle)