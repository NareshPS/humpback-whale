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

#Logging
from common import logging

class ImageDataGenerator:
    """It has functionality to create generators to feed data to keras.
    """

    def __init__(self,
                    source, dataframe, target_shape, batch_size,
                    normalize = True,
                    cache_size = 512):
        """It initializes the dataframe object.
        
        Arguments:
            source {string} -- A string to indicate the location of source images.
            dataframe {Pandas DataFrame} -- A pandas dataframe object with columnar data with image names and labels.
            target_shape {(width, height)} -- A tuple that indicates the target image dimensions.
            batch_size {int} -- A number indicating the size of each batch.
            normalize {bool} -- A boolean flag to enable img_obs normalization.
            cache_size {int} -- A integer value to determine the size of the cache.
        """
        #Required parameters
        self._source = source
        self._dataframe = dataframe
        self._batch_size = batch_size
        self._dataset_size = len(dataframe)
        self._target_shape = target_shape

        #Optional parameters
        self._normalize = normalize
        self._cache_size = cache_size

        #Initialize shuffled indices
        self._shuffled_indices = list(range(self._dataset_size))

        #Caching
        self._image_cache = LRUCache(self._cache_size)

        #Logging
        self._logger = logging.get_logger(__name__)

    def flow(self, x_cols, y_col, transform_x_cols = []):
        """It creates an iterator to the input dataframe.
        For y_col, only binary inputs are supported.
        
        Arguments:
            x_cols {[string]} -- A list of column names with image paths.
            y_col {[type]} -- The column name for labels.
            transform_x_cols {[string]} -- A list of column names to apply transformations.
        """
        
        #Validate parameters
        invalid_transform_x_cols = [col for col in transform_x_cols if col not in x_cols]

        if invalid_transform_x_cols:
            raise ValueError("Transform cols: {} not found in x_cols".format(invalid_transform_x_cols))

        self._logger.info("flow:: xcols: {} y_col: {}".format(x_cols, y_col))

        while True:
            #Shuffle indices before iterating over the datset.
            random_shuffle(self._shuffled_indices)

            for pos in range(0, self._dataset_size, self._batch_size):
                #Mark start and end of the current slice
                start = pos
                end = pos + self._batch_size
                
                self._logger.info("Using dataset slice [{}, {}]".format(start, end))

                #Make a data frame slice
                indices_slice = self._shuffled_indices[start:end]
                df_slice = self._dataframe.loc[indices_slice, :]

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

                yield (df_slice_x, df_slice_y)

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