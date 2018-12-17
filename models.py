"""Various NN models.
"""

#Keras imports
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPool1D, Conv1D, Dropout, Conv2D, MaxPool2D
from keras.optimizers import SGD

#Local imports
from common import constants

class ModelParameters:
    """Encapsulates model parameters
    """
    valid_names = ['n_classes', 'l_rate']
    def __init__(self, parameters):
        """Initializes the model parameters
        
        Arguments:
            parameters {dict} -- A dictionary of model parameters.
        """
        ModelParameters.validate_names(parameters)
        self._parameters = parameters
    
    @classmethod
    def validate_names(cls, names):
        invalid_names = [name for name in names if name not in ModelParameters.valid_names]
        if invalid_names:
            ValueError("Invalid parameters: {}".format(invalid_names))
    
    def parameters(self):
        return self._parameters

def cnn_model_1d_1(input_shape, n_classes, l_rate = 0.01):
    """A grayscale CNN model based on VGG.
    
    Arguments:
        input_shape {(int, int)} -- A tuple of input image dimensions.
        n_classes {int} -- The number of classification classes.
        l_rate {float} -- The learning rate of the gradient descent algorithm.
    
    Returns:
        A keras model object -- A trained keras model object.
    """

    model = Sequential()

    model.add(Conv1D(32, kernel_size = 3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(32, kernel_size = 3, activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv1D(64, kernel_size = 3, activation='relu'))
    model.add(Conv1D(64, kernel_size = 3, activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    #Compile the model
    print("Using learning rate: {l_rate}".format(l_rate = l_rate))
    sgd = SGD(lr=l_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def cnn_model_2d_1(input_shape, n_classes, l_rate = 0.01):
    """A grayscale CNN model based on VGG.
    
    Arguments:
        input_shape {(int, int, int)} -- A tuple of input image dimensions.
        n_classes {int} -- The number of classification classes.
        l_rate {float} -- The learning rate of the gradient descent algorithm.
    
    Returns:
        A keras model object -- A trained keras model object.
    """
    model = Sequential()

    model.add(Conv2D(32, kernel_size = 3, strides = 3, activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size = 3, strides = 3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size = 3, strides = 3, activation='relu'))
    model.add(Conv2D(64, kernel_size = 3, strides = 3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation='softmax'))

    sgd = SGD(lr=l_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])

    return model
