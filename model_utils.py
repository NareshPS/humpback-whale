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

def get_image_labels():
    """Creates a mapping of image file names to  labels.

    Returns:
        {string : string} -- A dictionary with file name mapped to its label.
    """
    #Load labels
    image_labels = {}

    with open(constants.RAW_DATASET_MAPPINGS["labels"], 'r') as handle:
        label_reader = csv.reader(handle)
        next(label_reader, None)
        
        for row in label_reader:
            image_labels[row[0]] = row[1]

        return image_labels

def get_label_ids():
    """Assigns a unique index to a label.

    Returns:
        {string:int} -- A dictionary that maps a label to an index.
    """
    image_labels = get_image_labels()
    label_ids = {label : idx for idx, label in enumerate(set(image_labels.values()))}

    return label_ids

def get_all_labels():
    """Creates a list of training labels.

    Returns:
        [string] -- A list of training labels 
    """
    image_labels = get_image_labels()
    return set(image_labels.values())

def load_training_batch(requestor, source_loc, img_files, batch_size, image_labels, label_ids):
    """It loads training data in batches.
    
    Arguments:
        source_loc {string} -- Location of preprocessed image files.
        img_files {[type]} -- List of image file names.
        batch_size {[type]} -- Number of images to load in a batch.
        image_labels {{string:string}} -- A mapping of image file name to their labels.
        label_ids {{string:int}} -- A mapping of label to a unique index.
    """

    num_classes = len(label_ids)
    for batch_img_files, imgs in load_images_batch(source_loc, img_files, batch_size = batch_size):
        #Normalize image data
        x = np.asarray(imgs)/255

        #Assign label indices.
        y = [label_ids[image_labels[img_file]] for img_file in batch_img_files]
        y = to_categorical(y, num_classes = num_classes)

        yield [x], y

def model_fit_data_feeder(requestor, source_loc, img_files, batch_size, image_labels, label_ids):
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