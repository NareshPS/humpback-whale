"""Generators to iterate over dataframes. The output of generators is formatted to support fit_generator in Keras.
"""

#Randomization
from random import shuffle as random_shuffle

#Dataset processing
from funcy import chunks
from image import operations
import numpy as np

class ImageDataGenerator:
    """It has functionality to create generators to feed data to keras.
    """

    def __init__(self,
                    source, dataframe, target_shape, batch_size,
                    normalize = True):
        """It initializes the dataframe object.
        
        Arguments:
            source {string} -- A string to indicate the location of source images.
            dataframe {Pandas DataFrame} -- A pandas dataframe object with columnar data with image names and labels.
            target_shape {(width, height)} -- A tuple that indicates the target image dimensions.
            batch_size {int} -- A number indicating the size of each batch.
            normalize {bool} -- A boolean flag to enable img_obs normalization.
        """
        #Required parameters
        self._source = source
        self._dataframe = dataframe
        self._batch_size = batch_size
        self._dataset_size = len(dataframe)
        self._target_shape = target_shape

        #Optional parameters
        self._normalize = normalize

        #Initialize shuffled indices
        self._shuffled_indices = list(range(self._dataset_size))

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

        while True:
            #Shuffle indices before iterating over the datset.
            random_shuffle(self._shuffled_indices)

            for pos in range(0, self._dataset_size, self._batch_size):
                #Make a data frame slice
                indices_slice = self._shuffled_indices[pos:pos + self._batch_size]
                df_slice = self._dataframe.loc[indices_slice, :]

                #Process labels
                df_slice_y = np.asarray(df_slice[y_col].tolist())
                df_slice_y = np.expand_dims(df_slice_y, axis = -1)

                #Process image columns
                df_slice_x = []  
                for x_col in x_cols:
                    img_objs = operations.imload(self._source, df_slice[x_col].tolist(), self._target_shape)
                    img_objs = self._apply_parameters(img_objs)

                    if x_col in transform_x_cols:
                        img_objs = self._apply_random_transformation(img_objs)

                    df_slice_x.append(img_objs)

                yield (df_slice_x, df_slice_y)

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
    

