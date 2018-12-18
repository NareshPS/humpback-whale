"""Utility methods for the model.
"""
#To parse label data source
import csv

#To process image data
import numpy as np

#Keras imports
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import TensorBoard

#To shuffle the dataset for each pass
from random import shuffle

#Load training history
from pickle import load as pickle_load

#Basic imports
from os import path
from time import time

#Local imports
from utils import load_images_batch, locate_file
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
        source_loc {string} -- The location of preprocessed image files.
        img_files {[string]} -- The list of image file names.
        batch_size {[int]} -- The number of images to load in a batch.
        input_labels {{string:string}} -- A mapping of image file name to their labels.
        label_ids {{string:int}} -- A mapping of label to a unique index.
    """

    num_classes = len(label_ids)
    for batch_img_files, imgs in load_images_batch(source_loc, img_files, batch_size = batch_size):
        #Normalize image data
        x = np.asarray(imgs)/255

        #Add additional dimension to make it compatible with Conv2D
        x = np.expand_dims(x, axis = -1)

        #Assign label indices.
        y = [label_ids[input_labels[img_file]] for img_file in batch_img_files]
        y = to_categorical(y, num_classes = num_classes)

        yield [x], y

def load_training_data(source_loc, input_set, input_labels, label_ids):
    """It loads the input set.
    
    Arguments:
        source_loc {string} -- Location of preprocessed image files.
        input_set {[string]} -- List of image file names.
        input_labels {{string:string}} -- A mapping of image file name to their labels.
        label_ids {{string:int}} -- A mapping of label to a unique index.
    """
    x = []
    y = []
    batch_size = 32

    for data, labels in list(load_training_batch("test", source_loc, input_set, batch_size, input_labels, label_ids)):
        x.append(data[0])
        y.append(labels)

    return np.vstack(x), np.vstack(y)

def _model_fit_data_feeder(requestor, source_loc, input_set, batch_size, image_labels, label_ids):
    """It is used in fit_generator() to supply the training and the validation data.
    
    Arguments:
        source_loc {string} -- The location of the image files.
        input_set {[type]} -- The list of input set file names.
        batch_size {[type]} -- Number of images to load in a batch.
        image_labels {{string:string}} -- A mapping of image file name to their labels.
        label_ids {{string:int}} -- A mapping of label to a unique index.
    """
    while True:
        shuffle(input_set)
        for x, y in load_training_batch(requestor, source_loc, input_set, batch_size, image_labels, label_ids):
            yield x, y

def _create_summary_callback(logs_loc, source_loc, validation_set, input_labels, label_ids):
    """Creates a callback to collect training statistics.

    Arguments:
        logs_loc {string} -- The output location of summary statistics.
        source_loc {string} -- The location of the validation set files.
        validation_set {[string]} -- Validation set file names.
        input_labels {[string]} -- A mapping from input item name to its label.
        label_ids {[int]} -- A mapping from input label to its id.
    
    Returns:
        A keras callback object -- A keras callback object configured to collect training insights.
    """
    logs_loc = path.join(logs_loc, str(time()))
    callback = TensorBoard(log_dir=logs_loc, histogram_freq = 1, write_graph=True, write_images=True)
    return callback

def load_pretrained_model(model_name, location = None):
    """It loads the model and the history objects from the disk.
    
    Arguments:
        model_name {string} -- The name of a model to be loaded.
        location {string} -- The location of the model file.
    
    Returns:
        (model, history) -- A tuple of a model and a history object loaded from the disk.
    """
    history_file = "{model_name}.hist".format(model_name = model_name)
    model_file = "{model_name}.h5".format(model_name = model_name)

    history_path = path.join(location, history_file) if location else history_file
    model_path = path.join(location, model_file) if location else model_name

    model = load_model(model_path)
    history = None
    with open(history_path, 'rb') as handle:
        history = pickle_load(handle)

    return (model, history)

def model_fit(model, source_loc, train_set, validation_set, input_labels, label_ids, batch_size, n_epochs):
    """Splits the data into two sets. Training set is used for training the model. Validation set validates it.
    Training is conducted in batches.
    
    Arguments:
        model {keras model object} -- Keras model objects.
        source_loc {string} -- The location of input set.
        train_set {[string]} -- A list of items in the training set.
        validation_set {[string]} -- A list of items in the validation set.
        input_labels {[string]} -- A mapping from input item name to its label.
        label_ids {[int]} -- A mapping from input label to its id.
        batch_size {int} -- The batch size.
        n_epochs {int} -- The number of epochs to train the model.

    Returns:
        A keras history object -- A keras history object returned by fit_generator()
    """
    #Generate validation data
    validation_data = load_training_data(source_loc, validation_set, input_labels, label_ids)

    #Training summary collection callback
    summary_callback = _create_summary_callback(constants.TENSORBOARD_LOGS_LOC, source_loc, validation_set, input_labels, label_ids)

    history = model.fit_generator(
                _model_fit_data_feeder("training", source_loc, train_set, batch_size, input_labels, label_ids),
                steps_per_epoch = int((len(train_set) + batch_size - 1)/batch_size),
                epochs = n_epochs,
                validation_data = validation_data,
                #validation_data=_model_fit_data_feeder("validation", source_loc, validation_set, batch_size, input_labels, label_ids),
                #validation_steps=int((len(validation_set) + batch_size - 1)/batch_size),
                callbacks = [summary_callback])
 
    return history