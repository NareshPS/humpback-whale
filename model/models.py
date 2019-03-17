"""Various NN models.
"""
#Keras imports
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import Adam
from keras import backend as K

#Base models
from model.basemodel import BaseModel

#Model specification
from model.definition import ModelSpecification, LayerSpecification, LayerType

#Local imports
from common import constants

base_name = 'Base'

def siamese_network(base_model_name, input_shape, learning_rate, feature_dims):
    """It creates a siamese network model using the input as a base model.

    Arguments:
        base_model_name {string} -- A string containing the name of a base model.
        input_shape {(int, int, int))} -- A tuple to indicate the shape of inputs.
        learning_rate {float} -- A float value to control speed of learning.
        feature_dims {int} -- An integer indicating the dimensions of the feature vector.

    Returns:
        {A Model object} -- A keras model.
    """
    #Layer name guidances
    anchor_name_guidance = 'Anchor'
    sample_name_guidance = 'Sample'

    #Layer arrgs
    kwargs = dict()

    #Base model handler
    base_model = BaseModel(base_model_name, input_shape)

    #Feature models
    feature_model = base_model.base_model()

    #Siamese inputs
    anchor_input = Input(shape = input_shape, name = anchor_name_guidance)
    sample_input = Input(shape = input_shape, name = sample_name_guidance)

    #Feature vectors
    anchor_features = feature_model(anchor_input)
    sample_features = feature_model(sample_input)

    lambda_product = Lambda(lambda x : x[0] * x[1], **kwargs)([anchor_features, sample_features])
    lambda_add = Lambda(lambda x : x[0] + x[1], **kwargs)([anchor_features, sample_features])
    lambda_abs = Lambda(lambda x : K.abs(x[0] - x[1]), **kwargs)([anchor_features, sample_features])
    lambda_eucledian_dist = Lambda(lambda x: K.square(x), **kwargs)(lambda_abs)

    #Layer specifications
    layer_specifications = [
                                #Concatenate and reshape lambda outputs
                                LayerSpecification(LayerType.Concatenate),
                                LayerSpecification(LayerType.Reshape, (4, feature_model.output_shape[1], 1)),

                                #Convolution layer #1
                                LayerSpecification(LayerType.Conv2D, 32, (4, 1), activation = 'relu', padding = 'valid'),
                                LayerSpecification(LayerType.Reshape, (feature_model.output_shape[1], 32, 1)),

                                #Convolution layer #2
                                LayerSpecification(LayerType.Conv2D, 1, (1, 32), activation = 'linear', padding = 'valid'),

                                #Flatten
                                LayerSpecification(LayerType.Flatten),

                                #Output unit
                                LayerSpecification(LayerType.Dense, 1, activation = 'sigmoid', use_bias = True)
                            ]

    #Model specification
    model_specification = ModelSpecification(layer_specifications)

    #Output
    X = model_specification.get_specification([lambda_product, lambda_add, lambda_abs, lambda_eucledian_dist])

    #Create an optimizer object
    adam_optimizer = Adam(lr = learning_rate)

    model = Model(inputs = [anchor_input, sample_input], outputs = [X], name = 'Siamese Model')
    model.compile(loss='binary_crossentropy', optimizer = adam_optimizer, metrics = ['accuracy'])
    model.summary()

    return model

def cnn(base_model_name, input_shape, learning_rate, feature_dims):
    """It creates a convolutional network model using the input as a base model.

    Arguments:
        base_model_name {string} -- A string containing the name of a base model.
        input_shape {(int, int, int))} -- A tuple to indicate the shape of inputs.
        learning_rate {float} -- A float value to control speed of learning.
        feature_dims {int} -- An integer indicating the dimensions of the feature vector.

    Returns:
        {A Model object} -- A keras model.
    """
    #Base model
    base_model = BaseModel(base_model_name, input_shape)

    #Model
    model = base_model.cnn(feature_dims)

    #Create an optimizer object
    adam_optimizer = Adam(lr = learning_rate)

    #Compile the model
    model.compile(loss = 'categorical_crossentropy', optimizer = adam_optimizer, metrics = ['categorical_accuracy'])
    model.summary()

    return model
