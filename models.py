"""Various NN models.
"""

#Keras imports
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPool1D, Conv1D, Dropout
from keras.optimizers import SGD

#Local imports
from common import constants

def cnn_gray_model_1(input_shape, num_classes, l_rate = 0.01):
    """A grayscale CNN model based on VGG.
    
    Arguments:
        input_shape {(int, int)} -- A tuple of input image dimensions.
        num_classes {int} -- The number of classification classes.
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
    model.add(Dense(num_classes, activation='softmax'))

    #Compile the model
    print("Using learning rate: {l_rate}".format(l_rate = l_rate))
    sgd = SGD(lr=l_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model