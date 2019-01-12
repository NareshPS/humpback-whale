"""Generators to iterate over dataframes. The output of generators is formatted to support fit_generator in Keras.
"""

#Randomization
from random import shuffle as random_shuffle

#Caching
from cachetools import LRUCache

#Dataset processing
from funcy import chunks
from image import operations
import numpy as np

#Keras sequence
from keras.utils import Sequence

#Logging
from common import logging

class ImageDataIterator(Sequence):
        """It iterates over the dataframe to return a batch of input per next() call.
        """

        def __init__(self, image_data_generator, dataframe, batch_size, x_cols, y_col, transform_x_cols, subset):
            """It initializes the required and optional parameters
            
            Arguments:
                image_data_generator {An ImageDataGenerator object} -- A generator object that allows loading a data slice.
                dataframe {A pandas.DataFrame object} -- A data frame object containing the input data.
                batch_size {int} -- An integer value that indicates the size of a batch.
                x_cols {[string]} -- A list of names of columns that contain image file names.
                y_col {string} -- A string value of the name of column with labels.
                transform_x_cols {[string]} -- A list of names of columns that are eligible for image transformation.
                subset {string} -- A string to indicate the dataset name.
            """

            #Required parameters
            self._image_data_generator = image_data_generator
            self._dataframe = dataframe
            self._batch_size = batch_size
            self._x_cols = x_cols
            self._y_col = y_col
            self._transform_x_cols = transform_x_cols
            self._subset = subset

            #Internal parameters
            self._dataset_size = len(dataframe)

            #Randomize dataset
            self._shuffled_indices = list(range(self._dataset_size))

            #Logging
            self._logger = logging.get_logger(__name__)

        def __len__(self):
            """It calculates the number of batches per epoch
            
            Returns:
                {int} -- An integer indicating the number of batches.
            """

            batches_per_epoch = int((self._dataset_size + self._batch_size - 1)/self._batch_size)
            return batches_per_epoch

        def __getitem__(self, batch_id):
            """It loads the data for a given batch_id.
            
            Arguments:
                batch_id {int} -- An integer indicating the batch id.
            
            Returns:
                {(Numpy data, Numpy data)} -- A tuple of input data and labels for the input batch id.
            """

            #Mark start and end of the current slice
            start = batch_id*self._batch_size
            end = start + self._batch_size
            
            self._logger.info("Using dataset:{} slice [{}, {}] for batch_id: {}".format(self._subset, start, end, batch_id))

            #Make a data frame slice
            indices_slice = self._shuffled_indices[start:end]
            df_slice = self._dataframe.loc[indices_slice, :]

            return self._image_data_generator._load_slice(
                            df_slice,
                            self._x_cols,
                            self._y_col,
                            self._transform_x_cols)

        def on_epoch_end(self):
            self._logger.info("End of epoch. Shuffling the dataset:{}".format(self._subset))

            #Shuffle indices before iterating over the datset.
            random_shuffle(self._shuffled_indices)

class ImageDataGenerator:
    """It has functionality to create generators to feed data to keras.
    """
    valid_subsets = ['training', 'validation']

    def __init__(self,
                    source, dataframe, target_shape, batch_size,
                    validation_split = None,
                    normalize = True,
                    cache_size = 512):
        """It initializes the dataframe object.
        
        Arguments:
            source {string} -- A string to indicate the location of source images.
            dataframe {Pandas DataFrame} -- A pandas dataframe object with columnar data with image names and labels.
            target_shape {(width, height)} -- A tuple that indicates the target image dimensions.
            batch_size {int} -- A number indicating the size of each batch.
            validation_split {float} -- A float indicating the fraction of validation split of the dataset.
            normalize {bool} -- A boolean flag to enable img_obs normalization.
            cache_size {int} -- A integer value to determine the size of the cache.
        """
        #Required parameters
        self._source = source
        self._dataframe = dataframe
        self._batch_size = batch_size
        self._target_shape = target_shape

        #Optional parameters
        self._validation_split = validation_split
        self._normalize = normalize
        self._cache_size = cache_size

        #Caching
        self._image_cache = LRUCache(self._cache_size)

        #Logging
        self._logger = logging.get_logger(__name__)

        #Training and validation set boundaries
        if self._validation_split is not None:
            #Compute the training and validation boundary using the validation split.
            boundary = int(len(self._dataframe)*(1. - self._validation_split))

            self._logger.info("Validation split: {} Identified boundary: {}".format(self._validation_split, boundary))

            #Split the dataframe into training and validation.
            self._train_df = self._dataframe.loc[:(boundary - 1), :]
            self._validation_df = self._dataframe.loc[boundary:, :].reset_index(drop=True)
        else:
            self._train_df = self._dataframe
        
        n_dataframe = len(self._dataframe)
        n_train_df = len(self._train_df)
        n_validation_df = len(self._validation_df) if validation_split is not None else 0

        self._logger.info(
                "Dataframe size: {} training size: {} validation size: {}".format(n_dataframe, n_train_df, n_validation_df))

    def flow(self, x_cols, y_col, transform_x_cols = [], subset = 'training'):
        """It creates an iterator to the input dataframe.
        For y_col, only binary inputs are supported.
        
        Arguments:
            x_cols {[string]} -- A list of column names with image paths.
            y_col {[type]} -- The column name for labels.
            transform_x_cols {[string]} -- A list of column names to apply transformations.
            subset {string} -- A string to indicate select between training and validation splits.
        """
        #Validate parameters
        invalid_transform_x_cols = [col for col in transform_x_cols if col not in x_cols]

        if invalid_transform_x_cols:
            raise ValueError("Transform cols: {} not found in x_cols".format(invalid_transform_x_cols))

        self._logger.info("flow:: xcols: {} y_col: {}".format(x_cols, y_col))

        #Validate subset parameter
        if subset not in ImageDataGenerator.valid_subsets:
            raise ValueError("Valid values of subset are: {}".format(ImageDataGenerator.valid_subsets))

        dataframe = self._train_df if subset == 'training' else self._validation_df

        self._logger.info("flow:: subset: {} dataset size: {}".format(subset, len(dataframe)))

        return ImageDataIterator(
                    self,
                    dataframe,
                    self._batch_size,
                    x_cols,
                    y_col,
                    transform_x_cols,
                    subset)

    def _load_slice(self, df_slice, x_cols, y_col, transform_x_cols):
        """It loads the image objects for the data frame slice.
        
        Arguments:
            df_slice {A pandas.DataFrame object} -- A pandas DataFrame object containing input data and labels.
            x_cols {[string]} -- A list of dataframe columns that contain the image names.
            y_col {string} -- A string value indicating the name of the label column in the dataframe.
            transform_x_cols {[string]} -- A list of dataframe columns that have transformation enabled.
        
        Returns:
            (Numpy object, Numpy object) -- A tuple of input data and labels.
        """
        #Process labels
        df_slice_y = np.asarray(df_slice[y_col].tolist())
        df_slice_y = np.expand_dims(df_slice_y, axis = -1)

        #Process image columns
        df_slice_x = []  
        for x_col in x_cols:
            images = df_slice[x_col].tolist()

            #Load images
            img_objs_map = self._get_image_objects(images)

            #Arrange them in the input order
            img_objs = [img_objs_map[image] for image in images]
            img_objs = np.asarray(img_objs)

            if x_col in transform_x_cols:
                img_objs = self._apply_random_transformation(img_objs)

            df_slice_x.append(img_objs)

        return (df_slice_x, df_slice_y)
            
    def _get_image_objects(self, images):
        """It loads the image objects for the list of images.
        If the image is available, it is loaded from the cache.
        Otherwise, it is loaded from the disk.
        
        Arguments:
            images {[string]} -- A list of image names.
        """
        img_objs = {}
        candidate_images = set(images)
        for image in candidate_images:
            #Get the image object for the current image from the cache.
            #Add to the dictionary, if it is not None.
            img_obj = self._image_cache.get(image)

            if img_obj is not None:
                img_objs[image] = img_obj

        #Create a list of missing images.
        cached_images = set(img_objs.keys())
        missing_images = [image for image in candidate_images if not image in cached_images]

        self._logger.info("Cached images: {} missing images: {}".format(cached_images, missing_images))

        #Load the missing image objects, and apply parameters.
        missing_img_objs = operations.imload(self._source, missing_images, self._target_shape)
        missing_img_objs = self._apply_parameters(missing_img_objs)

        #Update the cache
        self._image_cache.update(zip(missing_images, missing_img_objs))

        #Update the image object dictionary with the missing image objects.
        for image, img_obj in zip(missing_images, missing_img_objs):
            img_objs[image] = img_obj

        return img_objs

    def _apply_parameters(self, img_objs):
        """It processes image objects based on the input parameters.
        e.g. normalization, reshaping etc.
        
        Arguments:
            img_objs {numpy.ndarray} -- A numpy array of image objects.
        """
        if self._normalize:
            img_objs = operations.normalize(img_objs)

        return img_objs

    def _apply_random_transformation(self, img_objs):
        None