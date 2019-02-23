"""It provides classes to perform training.
"""
#Logging
from common import logging

#Keras imports
from keras import backend as K

#Image data generation, transformation
from operation.image import ImageDataGeneration
from operation.transform import ImageDataTransformation

#Save checkpoints
from model.callback import ModelDropboxCheckpoint

class ImageTraining(object):
    def __init__(
            self,
            input_params,
            image_generation_params,
            training_params,
            transformation_params,
            dropbox_auth,
            dropbox_dir):
        """It initializes the training parameters.
        
        Arguments:
            input_params {operation.input.InputParameters} -- The input parameters for the training.
            image_generation_params {operation.input.ImageGenerationParameters} -- The parameters required for image data generation.
            training_params {operation.input.TrainingParameters} -- The parameter for running a training session.
            session_params {operation.input.SessionParameters} -- The session parameters
            dropbox_auth {string} -- The authentication token to access dropbox store.
            dropbox_dir {string} -- The dropbox directory to store the generated data.
        """
        #Required parameters
        self._input_params = input_params
        self._image_generation_params = image_generation_params
        self._training_params = training_params
        self._transformation_params = transformation_params

        #Optional parameters
        self._dropbox_dir = dropbox_dir
        self._dropbox_auth = dropbox_auth

        #Derived parameters
        self._transformer = ImageDataTransformation(parameters = self._transformation_params)

        #Logging
        self._logger = logging.get_logger(__name__)

    def train(self, model, input_data, session_params):
        """It executes the training.
        
        Arguments:
            model {keras.Model} -- The keras model object.
            input_data {pandas.DataFrame} -- The input dataframe.
            session_params {operation.input.SessionParameters} -- The session parameters
            
        """
        self._logger.info('Training session:: session_params: %s', session_params)

        #Create a data generator to be used for fitting the model.
        datagen = ImageDataGeneration(
                        input_data,
                        self._input_params,
                        self._image_generation_params,
                        self._transformer)

        #Fit the data generator
        datagen.fit(n_images = self._training_params.num_fit_images)

        #Training flow
        train_gen = datagen.flow(subset = 'training')

        #Validation flow
        validation_gen = datagen.flow(subset = 'validation') if self._image_generation_params.validation_split else None

        if self._training_params.learning_rate:
            #Update the learning rate
            self._logger.info("Switching learning rate from: {} to: {}".format(
                                                                    K.get_value(model.optimizer.lr),
                                                                    self._training_params.learning_rate))

            K.set_value(model.optimizer.lr, self._training_params.learning_rate)
            
        #Training callbacks
        dropbox_callback = ModelDropboxCheckpoint(
                                    self._input_params.model_name,
                                    session_params,
                                    dropbox_auth = self._dropbox_auth,
                                    dropbox_dir = self._dropbox_dir)

        #Fit the model the input.
        model.fit_generator(
                        generator = train_gen,
                        validation_data = validation_gen,
                        epochs = self._training_params.number_of_epochs,
                        callbacks = [dropbox_callback])

        self._logger.info('Training finished. Trained: %d epochs', self._training_params.number_of_epochs)

        return model