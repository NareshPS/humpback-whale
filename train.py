"""Trains a model and save it to the disk.

    Usage:: python train.py <output_model_name> <batch_size> <num_epochs> <[int] Size of training set. If not specified, process all.>
"""
#Basic imports
from sys import argv, stdout

#Local imports
from common import constants
from models import model_1
from model_utils import get_input_labels, get_label_ids, model_fit
from utils import list_files

if __name__ == "__main__":
    """Entry point to train the model.

    Usage:: python train.py <output_model_name> <batch_size> <num_epochs> <[int] Size of training set. If not specified, process all.>
    """
    n_args = len(argv)
    if n_args not in [4, 5]:
        print("Syntax error. Usage:: python train.py <output_model_name> <batch_size> <num_epochs> <[int] Size of training set. If not specified, process all.>") 
        exit(-1)
    
    o_model = argv[1]
    batch_size = int(argv[2])
    n_epochs = int(argv[3])
    n_images = int(argv[4]) if len(argv) == 5 else None

    dataset = "train"
    validation_split = 0.2
    input_shape = constants.IMG_SHAPE
    source_loc = constants.PROCESSED_DATASET_MAPPINGS[dataset]
    
    input_labels = get_input_labels()
    label_ids = get_label_ids()
    num_classes = len(label_ids)
    input_set = list_files(source_loc, n_images)

    #Output model file
    model_file = "{o_model}.h5".format(o_model = o_model)

    #Initialize the model
    model = model_1(input_shape, num_classes)
    model.summary()

    #Train the model
    model_fit(model, source_loc, input_set, input_labels, label_ids, batch_size, n_epochs, validation_split)

    #Save model
    print("Saving model {model_file}".format(model_file = model_file))
    model.save(model_file, overwrite=True)