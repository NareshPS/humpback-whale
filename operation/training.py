"""It provides classes to perform training.
"""
#Logging
from common import logging

#Keras imports
from keras import backend as K

#Image data generation, transformation, and prediction
from operation.image import ImageDataGeneration
from operation.transform import ImageDataTransformation
from operation.prediction import Prediction

#Save checkpoints
from model.callback import ModelDropboxCheckpoint

#Constants
from common import constants

#Numpy imports
from numpy import nonzero

#Math operations
from math import ceil

#Progress bar
from tqdm import tqdm

class ImageTraining(object):
    def __init__(
            self,
            input_params,
            training_params,
            image_generation_params,
            transformation_params,
            dropbox_auth,
            dropbox_dir):
        """It initializes the training parameters.
        
        Arguments:
            input_params {operation.input.InputParameters} -- The input parameters for the training.    
            training_params {operation.input.TrainingParameters} -- The parameter to start training.
            image_generation_params {operation.input.ImageGenerationParameters} -- The parameters required for image data generation.
            dropbox_auth {string} -- The authentication token to access dropbox store.
            dropbox_dir {string} -- The dropbox directory to store the generated data.
        """
        #Required parameters
        self._input_params = input_params
        self._training_params = training_params
        self._image_generation_params = image_generation_params
        self._transformation_params = transformation_params

        #Optional parameters
        self._dropbox_dir = dropbox_dir
        self._dropbox_auth = dropbox_auth

        #Derived parameters
        self._transformer = ImageDataTransformation(parameters = self._transformation_params)

        #Logging
        self._logger = logging.get_logger(__name__)

    def _prepare(self, model):
        """It performs initializations before stating training.

        Arguments:
            model {keras.Model} -- The keras model object.
        """
        if self._training_params.learning_rate:
            #Update the learning rate
            self._logger.info("Switching learning rate from: {} to: {}".format(
                                                                            K.get_value(model.optimizer.lr),
                                                                            self._training_params.learning_rate))

            K.set_value(model.optimizer.lr, self._training_params.learning_rate)

    def _generators(self, input_data, randomize = True):
        """It creates training and validation generators for the training

        Arguments:
            input_data {pandas.DataFrame} -- The input dataframe.
            randomize {boolean} -- It indicates randomization of the input data. (default: {True})
        
        Returns:
            (operation.image.ImageDataIterator, operation.image.ImageDataIterator) -- The training and validation data iterators.
        """

         #Create a data generator to be used for fitting the model.
        datagen = ImageDataGeneration(
                        input_data,
                        self._input_params,
                        self._image_generation_params,
                        self._transformer,
                        randomize = randomize)

        #Fit the data generator
        datagen.fit(n_images = self._training_params.num_fit_images)

        #Training flow
        train_gen = datagen.flow(subset = 'training')

        #Validation flow
        validation_gen = datagen.flow(subset = 'validation') if self._image_generation_params.validation_split else None

        return train_gen, validation_gen

    def train(self, model, input_data):
        """It executes the training.
        
        Arguments:
            model {keras.Model} -- The keras model object.
            input_data {pandas.DataFrame} -- The input dataframe.      
        """
        self._logger.info('Training session:: training_params: %s', self._training_params)

        #Prepare for training
        self._prepare(model)

        #The generators to vend training and validation batches
        train_gen, validation_gen = self._generators(input_data)
            
        #Training callbacks
        dropbox_callback = None

        #Fit the model the input.
        model.fit_generator(
                        generator = train_gen,
                        validation_data = validation_gen,
                        epochs = self._training_params.number_of_epochs,
                        callbacks = [dropbox_callback])

        self._logger.info('Training finished. Trained: %d epochs', self._training_params.number_of_epochs)

        #Compute predictions
        predictor = Prediction(model, self._input_params, self._image_generation_params)
        result = predictor.predict(input_data, self._training_params.number_prediction_steps)

        return model, result

    def batch_train(self, model, input_data):
        """It trains the model with input data starting from the input batch id.
        
        Arguments:
            model {keras.Model} -- The keras model object.
            input_data {pandas.DataFrame} -- The input dataframe.

        Keyword Arguments:
            summary {boolean} -- It enables printing prediction summary. (Default: {True})
        """
        self._logger.info(
                        'Batch training session:: batch_id: %d epoch_id: %d',
                        self._training_params.batch_id,
                        self._training_params.epoch_id)

        #Prepare for training
        self._prepare(model)

        #Complete training the current epoch
        model = self._run_epoch(model, input_data, self._training_params.batch_id, self._training_params.epoch_id)

        #Training over the remaining epochs
        for epoch_id in range(self._training_params.epoch_id + 1, self._training_params.number_of_epochs):
            model = self._run_epoch(model, input_data, 0, epoch_id)

        #Compute predictions
        predictor = Prediction(model, self._input_params, self._image_generation_params)
        result = predictor.predict(input_data, self._training_params.number_prediction_steps)

        return model, result

    def _run_epoch(self, model, input_data, start_batch_id, epoch_id):
        """It run one epoch of training
        
        Arguments:
            model {keras.Model} -- The keras model object.
            input_data {pandas.DataFrame} -- The input dataframe.
            batch_id {int} -- The batch to start the training for the epoch id.
            epoch_id {int} -- The epoch id to train.
        """
        #Iterate over all the remaining training batches
        train_gen, validation_gen = self._generators(input_data)

        for batch_id in range(start_batch_id, len(train_gen)):
            self._logger.info('Training the batch_id: %d', batch_id)

            #Batch training data from the generator
            X, Y = train_gen.__getitem__(batch_id)

            #Feed the batch data for training
            model.train_on_batch(X, Y)

        #Validation phase
        if validation_gen:
            #Compute predictions
            predictor = Prediction(model, self._input_params, self._image_generation_params)
            result = predictor.predict(validation_gen._dataframe, len(validation_gen._dataframe))

            #Print summary
            self._print_summary(result)

        return model

    def _print_summary(self, result):
        """It prints the prediction summary.

        Arguments:
            result {pandas.DataFrame} -- The results data frame containing the predictions and the matches.
        """
        #Compute accuracy
        num_matches = nonzero(result[constants.PANDAS_MATCH_COLUMN])[0].shape[0]
        num_mismatches = len(result[constants.PANDAS_MATCH_COLUMN]) - num_matches
        accuracy = (num_matches/len(result[constants.PANDAS_MATCH_COLUMN])) * 100.

        summary = """
                    Result Dataframe: {}
                    Total predictions: {}
                    Correct predictions: {}
                    Wrong predictions: {}
                    Accuracy: {}
                """.format(
                        result,
                        len(result),
                        num_matches,
                        num_mismatches,
                        accuracy)

        #Print summary
        print(summary)