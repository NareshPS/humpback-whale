"""Entry point to train the model.

    Usage:: python train.py <batch_size> <num_epochs> <[int] Size of training set. If not specified, process all.>
"""
#Basic imports
from sys import argv, stdout

#Local imports
from common import constants
from models import model_1
from model_utils import get_image_labels, get_label_ids, model_fit_data_feeder
from utils import list_files

if __name__ == "__main__":
    """Entry point to train the model.

    Usage:: python train.py <batch_size> <num_epochs> <[int] Size of training set. If not specified, process all.>
    """
    n_args = len(argv)
    if n_args not in [3, 4]:
        print("Syntax error. Usage:: python train.py <batch_size> <num_epochs> <[int] Size of training set. If not specified, process all.>") 
        exit(-1)
    
    
    batch_size = int(argv[1])
    n_epochs = int(argv[2])
    n_images = int(argv[3]) if len(argv) == 4 else None

    dataset = "train"
    validation_split = 0.2
    input_shape = constants.IMG_SHAPE
    source_loc = constants.PROCESSED_DATASET_MAPPINGS[dataset]
    
    image_labels = get_image_labels()
    label_ids = get_label_ids()
    num_classes = len(label_ids)
    img_files = list_files(source_loc, n_images)
    n_images = len(img_files)
    
    #Training and validation sets
    split_marker = int(n_images*(1 - validation_split))
    train_set = img_files[:split_marker]
    validation_set = img_files[split_marker:]
    print("Training set: {t_size} validation set: {v_size}".format(t_size = len(train_set), v_size = len(validation_set)))

    #Initialize the model
    model = model_1(input_shape, num_classes)
    model.summary()

    #Train the model
    model.fit_generator(
        model_fit_data_feeder("training", source_loc, train_set, batch_size, image_labels, label_ids),
        steps_per_epoch = int((len(train_set) + batch_size - 1)/batch_size),
        epochs = n_epochs,
        validation_data=model_fit_data_feeder("validation", source_loc, validation_set, batch_size, image_labels, label_ids),
        validation_steps=int((len(validation_set) + batch_size - 1)/batch_size))