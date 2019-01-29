"""Various NN models.
"""
#Keras imports
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam

#Base models
from model.base_models import cnn

#Model specification
from model.definition import ModelSpecification, LayerSpecification, LayerType

#Local imports
from common import constants

def siamese_network(base_model, input_shape, feature_dims, learning_rate, train_base_layers = False):
    """It creates a siamese network model using the input as a base model.
    
    Arguments:
        base_model {A Model object.} -- A base model to generate feature vector.
        input_shape {(int, int, int))} -- A tuple to indicate the shape of inputs.
        feature_dims {int} -- An integer indicating the dimensions of the feature vector.
        learning_rate {float} -- A float value to control speed of learning.
        train_base_layers {boolean} -- A boolean value to indicate training of base layers.
    
    Returns:
        {A Model object} -- A keras model.
    """

    anchor_input = Input(shape = input_shape, name = 'Anchor')
    sample_input = Input(shape = input_shape, name = 'Sample')

    anchor_features = cnn(base_model, input_shape, feature_dims, train_base_layers)(anchor_input)
    sample_features = cnn(base_model, input_shape, feature_dims, train_base_layers)(sample_input)

    #Layer specifications
    layer_specifications = [
                                #Unit 1
                                LayerSpecification(LayerType.Concatenate),
                                LayerSpecification(LayerType.Dense, 16, activation = 'linear'),
                                LayerSpecification(LayerType.Normalization),
                                LayerSpecification(LayerType.Activation, 'relu'),

                                #Unit 2
                                LayerSpecification(LayerType.Dense, 4, activation = 'linear'),
                                LayerSpecification(LayerType.Normalization),
                                LayerSpecification(LayerType.Activation, 'relu'),

                                #Output unit
                                LayerSpecification(LayerType.Dense, 1, activation = 'sigmoid')
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
