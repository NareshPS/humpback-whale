"""Generators to iterate over dataframes. The output of generators is formatted to support fit_generator in Keras.
"""

#Randomization
from random import shuffle as random_shuffle
from random import randrange

#Caching
from cachetools import LRUCache

#Dataset processing
from funcy import chunks
from operation import utils
import numpy as np

#Keras sequence
from keras.utils import Sequence

#To categorical
from keras.utils import to_categorical

#Enum
from enum import Enum, unique
from bidict import frozenbidict

#Logging
from common import logging

#Metric
from common.metric import Metric

#Math functions
from math import ceil

class ImageDataIterator(Sequence):
    """It iterates over the dataframe to return a batch of input per next() call.
    """

    def __init__(self, image_data_generator, dataframe, batch_size, subset, randomize = True):
        """It initializes the required and optional parameters

        Arguments:
            image_data_generator {An ImageDataGenerator object} -- A generator object that allows loading a data slice.
            dataframe {A pandas.DataFrame object} -- A data frame object containing the input data.
            batch_size {int} -- An integer value that indicates the size of a batch.
            subset {A ImageDataSubset object} -- A ImageDataSubset value to indicate the dataset subset.
            randomize {boolean} -- It indicates to randomize the dataframe.
        """

        #Required parameters
        self._image_data_generator = image_data_generator
        self._dataframe = dataframe
        self._batch_size = batch_size
        self._subset = subset
        self._randomize = randomize

        #Internal parameters
        self._dataset_size = len(dataframe)

        #Randomization
        self._shuffled_indices = list(range(self._dataset_size))

        #Logging
        self._logger = logging.get_logger(__name__)

        #Pre-randomization
        if self._randomize:
            random_shuffle(self._shuffled_indices)

    def __len__(self):
        """It calculates the number of batches per epoch

        Returns:
            {int} -- An integer indicating the number of batches.
        """
        #Compute steps per epoch
        steps_per_epoch = ceil(self._dataset_size / self._batch_size)

        self._logger.info('steps_per_epoch: %d', steps_per_epoch)

        return steps_per_epoch

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

        self._logger.info(
                        "Using dataset:{} slice [{}, {}] for batch_id: {}".format(
                                                                            ImageDataGeneration.valid_subsets.inv[self._subset],
                                                                            start,
                                                                            end,
                                                                            batch_id))

        #Make a data frame slice
        indices_slice = self._shuffled_indices[start:end]
        df_slice = self._dataframe.loc[indices_slice, :]

        return self._image_data_generator._load_subset_slice(df_slice, self._subset)

    def on_epoch_end(self):
        self._logger.info("End of epoch. Shuffling the dataset:{}".format(ImageDataGeneration.valid_subsets.inv[self._subset]))

        #Shuffle indices before iterating over the datset.
        if self._randomize:
            random_shuffle(self._shuffled_indices)

@unique
class ImageDataSubset(Enum):
    """The enum values for data generator subsets.
    """
    Training = 1
    Validation = 2
    Prediction = 3

class ImageDataGeneration:
    """It has functionality to create generators to feed data to keras.
    """
    valid_subsets = frozenbidict({
                                    'training' : ImageDataSubset.Training,
                                    'validation' : ImageDataSubset.Validation,
                                    'prediction' : ImageDataSubset.Prediction
                                })

    def __init__(self, dataframe, input_params, image_generation_params, transformer = None, randomize = True):
        """It initializes the dataframe object.

        Arguments:
            dataframe {Pandas DataFrame} -- A pandas dataframe object with columnar data with image names and labels.
            input_params {A InputDataParameter object} -- An input parameter object.
            image_generation_params {A ImageGenerationParameters object} -- A training data parameter object.
            transformer {A ImageDataTransformation object} -- It is used to transform the image objects.
            randomize {boolean} -- It indicates randomization of the input dataframe.
        """
        #Required parameters
        self._dataframe = dataframe
        self._input_params = input_params
        self._image_generation_params = image_generation_params

        #Optional parameters
        self._transformer = transformer
        self._randomize = randomize

        #Caching
        self._image_cache = LRUCache(self._image_generation_params.image_cache_size)

        #Logging
        self._logger = logging.get_logger(__name__)

        #Metrics
        self._load_slice_metric = 'get_image_objects'

        #Create metrics
        Metric.create(self._load_slice_metric)

        #Compute the training and validation boundary using the validation split.
        boundary = int(ceil(len(self._dataframe)*(1. - self._image_generation_params.validation_split)))
        self._logger.info("Validation split: {} Identified boundary: {}".format(self._image_generation_params.validation_split, boundary))

        #Split the dataframe into training and validation.
        self._main_df = self._dataframe.loc[:(boundary - 1), :]
        self._validation_df = self._dataframe.loc[boundary:, :].reset_index(drop = True)

        n_dataframe = len(self._dataframe)
        n_main_df = len(self._main_df)
        n_validation_df = len(self._validation_df)

        self._logger.info(
                "Dataframe size: {} main set size: {} validation size: {}".format(n_dataframe, n_main_df, n_validation_df))

    def _get_images(self, n_images):
        """It extracts the image names from the dataframe.

        Arguments:
            n_images {An numpy.array object} -- It is a 4-D numpy array containing image data.
        """
        df_size = len(self._main_df)
        loop_count = 0
        images = set()

        while len(images) <= n_images and loop_count < df_size:
            random_index = randrange(df_size)

            for image_col in self._image_generation_params.image_cols:
                images.add(self._main_df.loc[random_index, image_col])

            loop_count += 1

        return list(images)

    def fit(self, n_images):
        """It calculates statistics on the input dataset. These are used to perform transformation.

        Arguments:
            n_images {An numpy.array object} -- It is a 4-D numpy array containing image data.
        """
        if n_images <= 0:
            ValueError("Expected a positive integer for n_images. Got: {}".format(n_images))

        #Input list for data fitting
        images = self._get_images(n_images)

        self._logger.info("%d images to use for data fitting", len(images))

        #Image objects
        img_objs_map = self._get_image_objects(images)
        img_objs = np.asarray(list(img_objs_map.values()))

        self._logger.info("fit:: images: {} to the transformer to compute statistics".format(img_objs.shape))

        #Fit the data in the transformer
        self._transformer.fit(img_objs)

    def flow(self, subset = 'training'):
        """It creates an iterator to the input dataframe.

        Arguments:
            subset {string} -- A string to indicate select between training and validation splits.
        """
        #Validate subset parameter
        if not ImageDataGeneration.valid_subsets.get(subset):
            raise ValueError("Valid values of subset are: {}".format(list(ImageDataGeneration.valid_subsets.keys())))

        #Qualified subset
        q_subset = ImageDataGeneration.valid_subsets[subset]

        #Dataframe placeholder
        dataframe = None

        #Pick the correct dataframe
        if q_subset == ImageDataSubset.Training or q_subset == ImageDataSubset.Prediction:
            dataframe = self._main_df
        elif q_subset == ImageDataSubset.Validation:
            dataframe = self._validation_df

        self._logger.info("flow:: subset: {} dataset size: {}".format(subset, len(dataframe)))

        return ImageDataIterator(
                    self,
                    dataframe,
                    self._image_generation_params.batch_size,
                    q_subset,
                    randomize = self._randomize)

    def _load_subset_slice(self, df_slice, subset):
        """It loads the image objects and the labels for the data frame slice.

        Arguments:
            df_slice {A pandas.DataFrame object} -- A pandas DataFrame object containing input data and labels.

        Returns:
            {An object} -- A list of image objects in prediction phase. A tuple of image objects and their labels in training phase.
        """
        self._logger.info('Using subset: %s', subset)

        #Results placeholder
        results = None

        #Load the slice
        if subset == ImageDataSubset.Training or subset == ImageDataSubset.Validation:
            results = self._load_train_phase_slice(df_slice)
        elif subset == ImageDataSubset.Prediction:
            results = self._load_predict_phase_slice(df_slice)

        return results

    def _load_train_phase_slice(self, df_slice):
        """It loads the image objects and the labels for the data frame slice.

        Arguments:
            df_slice {A pandas.DataFrame object} -- A pandas DataFrame object containing input data and labels.

        Returns:
            (Numpy object, Numpy object) -- A tuple of input data and labels.
        """
        return self._load_slice(df_slice)

    def _load_predict_phase_slice(self, df_slice):
        """It loads the image objects for the data frame slice.

        Arguments:
            df_slice {A pandas.DataFrame object} -- A pandas DataFrame object containing input data and labels.

        Returns:
            (Numpy object, Numpy object) -- A tuple of input data and labels.
        """
        images, _ = self._load_slice(df_slice)

        return images

    def _load_slice(self, df_slice):
        """It loads the image objects for the data frame slice.

        Arguments:
            df_slice {A pandas.DataFrame object} -- A pandas DataFrame object containing input data and labels.

        Returns:
            (Numpy object, Numpy object) -- A tuple of input data and labels.
        """
        #Calculate the number of classes
        num_classes = self._image_generation_params.num_classes

        #Process labels
        df_slice_y = df_slice[self._image_generation_params.label_col].values
        df_slice_y_categorical = to_categorical(df_slice_y, num_classes = num_classes) if num_classes > 2 else df_slice_y

        #Process image columns
        df_slice_x = []

        for x_col in self._image_generation_params.image_cols:
            images = df_slice[x_col].tolist()

            #Load images
            img_objs_map = self._get_image_objects(images)

            #Arrange them in the input order
            img_objs = [img_objs_map[image] for image in images]
            img_objs = np.asarray(img_objs)

            if x_col in self._image_generation_params.image_transform_cols:
                img_objs = self._apply_transformation(img_objs)

            df_slice_x.append(img_objs)

        return (df_slice_x, df_slice_y_categorical)

    def _get_image_objects(self, images):
        """It loads the image objects for the list of images.
        If the image is available, it is loaded from the cache.
        Otherwise, it is loaded from the disk.

        Arguments:
            images {[string]} -- A list of image names.
        """
        #Start recording time
        record_handle = Metric.start(self._load_slice_metric)

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

        self._logger.debug("Cached images: {} missing images: {}".format(cached_images, missing_images))

        #Load the missing image objects, and apply parameters.
        missing_img_objs = utils.imload(
                                    self._image_generation_params.dataset_location,
                                    missing_images,
                                    self._image_generation_params.input_shape[:2])
        missing_img_objs = self._apply_parameters(missing_img_objs)

        #Update the cache
        self._image_cache.update(zip(missing_images, missing_img_objs))

        #Update the image object dictionary with the missing image objects.
        for image, img_obj in zip(missing_images, missing_img_objs):
            img_objs[image] = img_obj

        #End recording time
        Metric.stop(record_handle, self._load_slice_metric)

        return img_objs

    def _apply_parameters(self, img_objs):
        """It processes image objects based on the input parameters.
        e.g. normalization, reshaping etc.

        Arguments:
            img_objs {numpy.ndarray} -- A numpy array of image objects.
        """
        if self._image_generation_params.normalize:
            img_objs = utils.normalize(img_objs)

        return img_objs

    def _apply_transformation(self, img_objs):
        transformed_objects = img_objs

        if self._transformer:
            transformed_objects = self._transformer.transform(img_objs)

        return transformed_objects
