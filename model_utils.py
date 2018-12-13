"""Utility methods for the model.
"""
#To parse label data source
import csv

#To process image data
import numpy as np

#To generate one-hot vectors for labels.
from keras.utils import to_categorical

#To shuffle the dataset for each pass
from random import shuffle

#Local imports
from utils import load_images_batch
from common import constants

def get_input_labels():
    """Creates a mapping of image file names to  labels.

    Returns:
        {string : string} -- A dictionary with file name mapped to its label.
    """
    #Load labels
    input_labels = {}

    with open(constants.RAW_DATASET_MAPPINGS["labels"], 'r') as handle:
        label_reader = csv.reader(handle)
        next(label_reader, None)
        
        for row in label_reader:
            input_labels[row[0]] = row[1]

        return input_labels

def get_label_ids():
    """Assigns a unique index to a label.

    Returns:
        {string:int} -- A dictionary that maps a label to an index.
    """
    input_labels = get_input_labels()
    label_ids = {label : idx for idx, label in enumerate(set(input_labels.values()))}

    return label_ids

def get_all_labels():
    """Creates a list of training labels.

    Returns:
        [string] -- A list of training labels 
    """
    input_labels = get_input_labels()
    return set(input_labels.values())

def load_training_batch(requestor, source_loc, img_files, batch_size, input_labels, label_ids):
    """It loads training data in batches.
    
    Arguments:
        source_loc {string} -- Location of preprocessed image files.
        img_files {[type]} -- List of image file names.
        batch_size {[type]} -- Number of images to load in a batch.
        input_labels {{string:string}} -- A mapping of image file name to their labels.
        label_ids {{string:int}} -- A mapping of label to a unique index.
    """

    num_classes = len(label_ids)
    for batch_img_files, imgs in load_images_batch(source_loc, img_files, batch_size = batch_size):
        #Normalize image data
        x = np.asarray(imgs)/255

        #Assign label indices.
        y = [label_ids[input_labels[img_file]] for img_file in batch_img_files]
        y = to_categorical(y, num_classes = num_classes)

        yield [x], y

def _model_fit_data_feeder(requestor, source_loc, img_files, batch_size, image_labels, label_ids):
    """It is used in fit_generator() to supply the training and the validation data.
    
    Arguments:
        source_loc {string} -- Location of preprocessed image files.
        img_files {[type]} -- List of image file names.
        batch_size {[type]} -- Number of images to load in a batch.
        image_labels {{string:string}} -- A mapping of image file name to their labels.
        label_ids {{string:int}} -- A mapping of label to a unique index.
    """
    while True:
        shuffle(img_files)
        for x, y in load_training_batch(requestor, source_loc, img_files, batch_size, image_labels, label_ids):
            yield x, y

def model_fit(model, source_loc, input_set, input_labels, label_ids, batch_size, n_epochs, validation_split):
    """Splits the data into two sets. Training set is used for training the model. Validation set validates it.
    Training is conducted in batches.
    
    Arguments:
        model {keras model object} -- Keras model objects.
        source_loc {string} -- The location of input set.
        input_set {[string]} -- A list of items in the input set.
        input_labels {[string]} -- A mapping from input item name to its label.
        label_ids {[int]} -- A mapping from input label to its id.
        batch_size {int} -- The batch size.
        n_epochs {int} -- The number of epochs to train the model.
        validation_split {float} -- A float between [0, 1] indicating the split of train and validation sets.
    """

    n_images = len(input_set)

    #Training and validation sets
    split_marker = int(n_images*(1 - validation_split))
    train_set = input_set[:split_marker]
    validation_set = input_set[split_marker:]
    print("Training set: {t_size} validation set: {v_size}".format(t_size = len(train_set), v_size = len(validation_set)))

    model.fit_generator(
        _model_fit_data_feeder("training", source_loc, train_set, batch_size, input_labels, label_ids),
        steps_per_epoch = int((len(train_set) + batch_size - 1)/batch_size),
        epochs = n_epochs,
        validation_data=_model_fit_data_feeder("validation", source_loc, validation_set, batch_size, input_labels, label_ids),
        validation_steps=int((len(validation_set) + batch_size - 1)/batch_size))