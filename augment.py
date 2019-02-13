"""It augments the dataset with image transformations.
"""
#Constants
from common import constants

#Allow reproducible results
from numpy.random import seed as np_seed

#Data processing
from pandas import DataFrame
import numpy as np

#Load objects from the disk
from pandas import read_csv

#Image data generation
from operation.transform import ImageDataTransformation
from operation.augmentation import ImageAugmentation
from operation.utils import imload, imwrite

#Commandline arguments
from argparse import ArgumentParser

#Path manipulations
from pathlib import Path

#Logging
from common import logging

#Progress bar
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser(description = 'It augments the dataset.')

    parser.add_argument(
        '-d', '--dataset_location',
        required = True, type = Path,
        help = 'It specifies the input dataset.')
    parser.add_argument(
        '-o', '--output_dataset_location',
        required = True, type = Path,
        help = 'It specifies the location to write the output files.')
    parser.add_argument(
        '-i', '--input_file',
        required = True, type = Path,
        help = 'It specifies the location of the input data in csv format.')
    parser.add_argument(
        '--output_file',
        required = True, type = Path,
        help = 'It specifies the location of the output data in csv format.')
    parser.add_argument(
        '-n', '--num_inputs',
        type = int, nargs = '?',
        help = 'It specifies the number of inputs to process')
    parser.add_argument(
        '-c', '--image_col',
        required = True,
        help = 'It specifies the name of the image column.')
    parser.add_argument(
        '-s', '--target_shape',
        nargs = 2, type = int,
        help = 'It specifies the target image size.')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')

    args = parser.parse_args()
    
    return args

def get_augmentation_executor():
    #Augmentation instances
    augmentation_instances = [
                                #Horizontal flip
                                ImageAugmentation.Instance(
                                        ImageDataTransformation.Parameters(horizontal_flip = True, horizontal_flip_prob = 1.0),
                                        num_output_images = 1),
                                #Rotation
                                ImageAugmentation.Instance(
                                        ImageDataTransformation.Parameters(rotation_range = 20),
                                        num_output_images = 5),
                                #Zoom
                                ImageAugmentation.Instance(
                                        ImageDataTransformation.Parameters(zoom_range = 0.25),
                                        num_output_images = 5),
                                #Shear
                                ImageAugmentation.Instance(
                                        ImageDataTransformation.Parameters(shear_range = 15),
                                        num_output_images = 5),
                                #Width shift
                                ImageAugmentation.Instance(
                                        ImageDataTransformation.Parameters(width_shift_range = .25),
                                        num_output_images = 5),
                                #Height shift
                                ImageAugmentation.Instance(
                                        ImageDataTransformation.Parameters(height_shift_range = .20),
                                        num_output_images = 5)
                            ]

    executor = ImageAugmentation(augmentation_instances)

    return executor

if __name__ == '__main__':
    #Parse commandline arguments
    args = parse_args()

    dataset_location = args.dataset_location
    output_dataset_location = args.output_dataset_location
    input_file = args.input_file
    output_file = args.output_file
    num_inputs = args.num_inputs
    image_col = args.image_col
    target_shape = tuple(args.target_shape)
    log_to_console = args.log_to_console

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Log input parameters
    logger.info('Parameters:: dataset_location: %s output_dataset_location: %s', dataset_location, output_dataset_location)
    logger.info('Parameters:: input_file: %s output_file: %s', input_file, output_file)
    logger.info('Parameters:: image_col: %s target_shape: %s', image_col, target_shape)
    logger.info('Parameters:: num_inputs: %s log_to_console: %s', num_inputs, log_to_console)

    ### Validation Start ###
    #Source location
    if not dataset_location.exists():
        raise ValueError('Dataset location: {} is not found.'.format(dataset_location))

    #Destination location
    if not output_dataset_location.exists():
        output_dataset_location.mkdir(parents = True, exist_ok = True)

    #Input labels
    if not input_file.exists():
        raise ValueError('Input file: {} is not found.'.format(input_file))
    ### Validation End ###

    #Required inputs
    input_df = read_csv(input_file)
    input_df = input_df.loc[:(num_inputs - 1), :] if num_inputs else input_df

    ####################################### Augment the dataset [Start]############################################
    #Image augmentation
    augmentation_executor = get_augmentation_executor()

    #Output dataframe
    output_df = DataFrame(columns = list(input_df))

    #Augment images
    for _, row in tqdm(input_df.iterrows(), total = len(input_df), desc = 'Augmenting images'):
        #Extract the image name
        image_name = row[image_col]

        #Image object
        image_objs = imload(dataset_location, [image_name], target_shape)

        #Augmented image objects
        augmented_objs = augmentation_executor.augmentations(image_objs)

        logger.debug('Augmented image objects: {}'.format(augmented_objs.shape))

        #Image path
        image_path = Path(image_name)

        #Create target image names
        target_image_names = ["{}-{}{}".format(image_path.stem, index, image_path.suffix) for index in range(len(augmented_objs) + 1)]
        target_image_objs = np.concatenate([image_objs, augmented_objs], axis = 0)

        logger.debug('Reshaped image objects: {}'.format(target_image_objs.shape))

        #Augmented image map
        augmented_image_name_and_objects = dict(zip(target_image_names, target_image_objs))

        #Write images to the disk
        imwrite(output_dataset_location, augmented_image_name_and_objects)

        #Add to output dataframe
        for name, _ in augmented_image_name_and_objects.items():
            #Augmented row
            augmented_row = row.copy()
            augmented_row[image_col] = name

            #Add to the dataframe
            output_df = output_df.append(augmented_row, ignore_index = True)

    ####################################### Augment the dataset [End]############################################

    #Shuffle the output dataframe
    output_df = output_df.sample(frac = 1).reset_index(drop = True)

    #Write the output dataframe
    output_df.to_csv(output_file)

    #Statistics
    input_size = len(input_df)
    generated_output_size = sum(1 for _ in output_dataset_location.iterdir())
    expected_output_size = input_size * (len(augmentation_executor) + 1)
    deviation = expected_output_size - generated_output_size
    output_size = len(output_df)

    completion_statistics = """
                                    Augmentation Summary
                                    ====================
                                    Input size: {}
                                    Generated output size: {}
                                    Expected output size: {}
                                    Error: {}
                                    Output Size: {}
                                    """.format(
                                            input_size,
                                            generated_output_size,
                                            expected_output_size,
                                            deviation,
                                            output_size)

    #Output the statistics
    print(completion_statistics)