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

def plot_image_comparison(X, n_images, train_set_loc):
    n_images = 5
    figure, axes = plt.subplots(n_images, 2)

    for image_id in range(n_images):
        axes[image_id, 0].imshow(X[image_id])

        img = mpimg.imread(locate_file(train_set_loc, label_df.loc[image_id, df_image_col]))
        axes[image_id, 1].imshow(img)

    figure.tight_layout()

def create_label_dict(label_df):
    label_dict = defaultdict(list)
    for index, row in label_df.iterrows():
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
    parser = ArgumentParser(description = 'Usage:: python siamese_input_tuples.py [-o|--overwrite]')
    parser.add_argument(
        '-o', '--overwrite',
        action = 'store_true', default=False,
        help = 'If specified, overwrites the existing tuple file.')

    args = parser.parse_args()
    
    return args.overwrite

if __name__ == "__main__":
    df_image_col = constants.IMAGE_HEADER_NAME
    df_class_col = constants.LABEL_HEADER_NAME
    batch_size = 32
    label_df = read_csv(constants.DATASET_MAPPINGS['labels'])
    train_set_loc = constants.DATASET_MAPPINGS['train_preprocessed']
    train_tuples_loc = constants.DATASET_MAPPINGS['train_tuples']
    train_tuples_columns = constants.TRAIN_TUPLE_HEADERS

    #Parse commandline arguments
    overwrite_output_file = parse_args()

    train_tuples_df = None

    if not is_a_file(train_tuples_loc) or overwrite_output_file:
        label_dict = create_label_dict(label_df)
        file_set = create_file_set(label_df, df_image_col)
        train_tuples =  create_training_sample(label_dict, file_set)
        print(train_tuples[0])

        train_tuples_df = pd.DataFrame(train_tuples, columns = train_tuples_columns)
        train_tuples_df.to_csv(train_tuples_loc)

        print("Wrote {} image tuples".format(len(train_tuples_df)))
    else:
        print("File already exists. Please specify overwrite flag to regenerate.")



    