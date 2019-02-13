"""Various NN models.
"""
#Keras imports
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam

#Base models
from model.basemodel import BaseModel

#Model specification
from model.definition import ModelSpecification, LayerSpecification, LayerType

#Local imports
from common import constants

def siamese_network(base_model_name, input_shape, feature_dims, learning_rate, num_unfrozen_base_layers = 0):
    """It creates a siamese network model using the input as a base model.
    
    Arguments:
        base_model_name {string} -- A string containing the name of a base model.
        input_shape {(int, int, int))} -- A tuple to indicate the shape of inputs.
        feature_dims {int} -- An integer indicating the dimensions of the feature vector.
        learning_rate {float} -- A float value to control speed of learning.
        num_unfrozen_base_layers {int} -- The number of bottom layers of base model to train.
    
    Returns:
        {A Model object} -- A keras model.
    """
    #Base model
    base_model = BaseModel(base_model_name, input_shape, feature_dims, num_unfrozen_base_layers)

    #Siamese inputs
    anchor_input = Input(shape = input_shape, name = 'Anchor')
    sample_input = Input(shape = input_shape, name = 'Sample')

    #Feature inputs
    anchor_features = base_model.cnn()(anchor_input)
    sample_features = base_model.cnn()(sample_input)

    #Layer arrgs
    kwargs = dict(kernel_initializer = 'he_normal')

    #Layer specifications
    layer_specifications = [
                                #Unit 1
                                LayerSpecification(LayerType.Concatenate),
                                LayerSpecification(LayerType.Dense, 16, activation = 'linear', **kwargs),
                                LayerSpecification(LayerType.Normalization),
                                LayerSpecification(LayerType.Activation, 'relu'),
                                LayerSpecification(LayerType.Dropout, 0.5),

                                #Unit 2
                                LayerSpecification(LayerType.Dense, 4, activation = 'linear', **kwargs),
                                LayerSpecification(LayerType.Normalization),
                                LayerSpecification(LayerType.Activation, 'relu'),
                                LayerSpecification(LayerType.Dropout, 0.5),

                                #Output unit
                                LayerSpecification(LayerType.Dense, 1, activation = 'sigmoid', **kwargs)
                            ]

    #Model specification
    model_specification = ModelSpecification(layer_specifications)

    #Output
    X = model_specification.get_specification([anchor_features, sample_features])

    #Create an optimizer object
    adam_optimizer = Adam(lr = learning_rate)

    network = Model(inputs = [anchor_input, sample_input], outputs = [X], name = 'Siamese Model')
    network.compile(loss='binary_crossentropy', optimizer = adam_optimizer, metrics = ['accuracy'])
    network.summary()

    return network

def cnn(base_model_name, input_shape, feature_dims, learning_rate, num_unfrozen_base_layers = 0):
    """It creates a convolutional network model using the input as a base model.
    
    Arguments:
        base_model_name {string} -- A string containing the name of a base model.
        input_shape {(int, int, int))} -- A tuple to indicate the shape of inputs.
        feature_dims {int} -- An integer indicating the dimensions of the feature vector.
        learning_rate {float} -- A float value to control speed of learning.
        num_unfrozen_base_layers {int} -- The number of bottom layers of base model to train.
    
    Returns:
        {A Model object} -- A keras model.
    """
    #Base model
    base_model = BaseModel(base_model_name, input_shape, feature_dims, num_unfrozen_base_layers)
    
    #Model
    model = base_model.cnn()

    #Create an optimizer object
    adam_optimizer = Adam(lr = learning_rate)

    #Compile the model
    model.compile(loss = 'categorical_crossentropy', optimizer = adam_optimizer, metrics = ['accuracy'])
    model.summary()

    return model