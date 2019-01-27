"""Various NN models.
"""
#Keras imports
from keras.models import Sequential, Model
from keras.layers import Input, Concatenate, Dense, BatchNormalization, Activation
from keras.optimizers import SGD, Adam

#Base models
from model.base_models import cnn

#Local imports
from common import constants

def siamese_network(base_model, input_shape, feature_dims, learning_rate):
    """It creates a siamese network model using the input as a base model.
    
    Arguments:
        base_model {A Model object.} -- A base model to generate feature vector.
        input_shape {(int, int, int))} -- A tuple to indicate the shape of inputs.
        feature_dims {int} -- An integer indicating the dimensions of the feature vector.
        learning_rate {float} -- A float value to control speed of learning.
    
    Returns:
        {A Model object} -- A keras model.
    """

    anchor_input = Input(shape = input_shape, name = 'Anchor')
    sample_input = Input(shape = input_shape, name = 'Sample')

    anchor_features = cnn(base_model, input_shape, feature_dims)(anchor_input)
    sample_features = cnn(base_model, input_shape, feature_dims)(sample_input)

    X = Concatenate()([anchor_features, sample_features])
    X = Dense(16, activation = 'linear')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Dense(4, activation = 'linear')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Dense(1, activation = 'sigmoid')(X)
    
    #Create an optimizer object
    adam_optimizer = Adam(lr = learning_rate)

    network = Model(inputs = [anchor_input, sample_input], outputs = [X], name = 'Siamese Model')
    network.compile(loss='binary_crossentropy', optimizer = adam_optimizer, metrics = ['accuracy'])
    network.summary()

    return network
