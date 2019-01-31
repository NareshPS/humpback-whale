#Plotting
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

#Data processing
import numpy as np
import pandas as pd
from collections import defaultdict
from random import sample as random_sample
from math import ceil
from funcy import without

#Load and save objects to disk
from pandas import read_csv
from os.path import isfile as is_a_file

#Useful constants
from common import constants

#Commandline arguments
from argparse import ArgumentParser

#Logging
from common import logging

def create_label_dict(label_df):
    label_dict = defaultdict(list)
    for _, row in label_df.iterrows():
        class_name = row[df_class_col]
        file_name = row[df_image_col]

        label_dict[class_name].append(file_name)

    return label_dict

def create_file_set(label_df, df_image_col):
    return set(label_df[df_image_col])

def filtered_random_sample(file_set, prohibited_file_set, n_samples):
    current_sample = random_sample(file_set, n_samples)
    filtered_sample = list(without(current_sample, prohibited_file_set))

    n_current_sample = len(current_sample)
    n_filtered_sample = len(filtered_sample)
    n_required_samples = n_current_sample - n_filtered_sample

    while n_current_sample != n_filtered_sample:
        current_sample = random_sample(file_set, n_required_samples)
        filtered_sample = list(without(current_sample, prohibited_file_set))

        n_current_sample = len(current_sample)
        n_filtered_sample = len(filtered_sample)
        n_required_samples = n_current_sample - n_filtered_sample
    
    return filtered_sample

def create_training_sample(label_dict, file_set, samples_per_image = 10, positive_sample_ratio = 0.5):
    #Ceiled to account for labels with small sample size. e.g. 1
    n_required_positive_samples = ceil(samples_per_image * positive_sample_ratio)
    n_required_negative_samples = samples_per_image - n_required_positive_samples

    samples = []
    for label, files in label_dict.items():
        for anchor in files:
            n_avail_positive_samples = min(n_required_positive_samples, len(files))
            n_avail_negative_samples = n_required_negative_samples

            #Create random positive and negative samples for each anchor file
            positive_samples = random_sample(files, n_avail_positive_samples)
            negative_samples = filtered_random_sample(file_set, files, n_avail_negative_samples)

            #Positive samples
            for p_sample in positive_samples:
                sample = (anchor, p_sample, label, 1)
                samples.append(sample)
            
            #Negative samples
            for n_sample in negative_samples:
                sample = (anchor, n_sample, label, 0)
                samples.append(sample)

    return samples

def parse_args():
    parser = ArgumentParser(description = 'It generates training samples for the siamese network model.')
    parser.add_argument(
        '-o', '--overwrite',
        action = 'store_true', default=False,
        help = 'If specified, overwrites the existing tuple file.')
    parser.add_argument(
        '-d', '--dataset',
        choices = constants.DATASET_NAMES, default = constants.DATASET_NAMES[0],
        help = 'It specifies the dataset to use.')
    parser.add_argument(
        '-s', '--samples_per_image',
        default = 10, type = int,
        help = 'It specified the number of samples to create for each image.')
    parser.add_argument(
        '-r', '--positive_sample_ratio',
        default = 0.5, type = float,
        help = 'It specified the ratio of the positive and the negative samples in the generated dataset.')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')

    args = parser.parse_args()
    
    return args.overwrite, args.dataset, args.samples_per_image, args.positive_sample_ratio, args.log_to_console

if __name__ == "__main__":
    #Parse commandline arguments
    overwrite_output_file, dataset, samples_per_image, positive_sample_ratio, log_to_console = parse_args()


    df_image_col = constants.IMAGE_HEADER_NAME
    df_class_col = constants.LABEL_HEADER_NAME
    batch_size = 32
    label_df = read_csv(constants.DATASET_MAPPINGS['labels'])
    train_set_loc = constants.DATASET_MAPPINGS[dataset]
    train_tuples_loc = constants.DATASET_MAPPINGS['train_tuples']
    train_tuples_columns = constants.TRAIN_TUPLE_HEADERS

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Log input parameters
    logger.info(
                'Running with parameters overwrite_output_file: %s dataset: %s samples_per_image: %d positive_sample_ratio: %f',
                overwrite_output_file,
                dataset,
                samples_per_image, 
                positive_sample_ratio)

    train_tuples_df = None

    if not is_a_file(train_tuples_loc) or overwrite_output_file:
        label_dict = create_label_dict(label_df)
        file_set = create_file_set(label_df, df_image_col)
        train_tuples =  create_training_sample(label_dict, file_set, samples_per_image, positive_sample_ratio)

        train_tuples_df = pd.DataFrame(train_tuples, columns = train_tuples_columns)
        train_tuples_df.to_csv(train_tuples_loc)

        logger.info("Wrote {} image tuples".format(len(train_tuples_df)))
    else:
        raise ValueError("File already exists. Please specify overwrite flag to regenerate.")