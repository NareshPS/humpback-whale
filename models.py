"""Various convnet models.
"""

#Keras imports
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPool1D, Conv1D, Dropout
from keras.optimizers import SGD

#Local imports
from common import constants

def model_1(input_shape, num_classes):
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
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model