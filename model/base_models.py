#Keras support
from keras.applications.resnet50 import ResNet50 as ResNet
from keras.applications import InceptionV3
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model

#Constants
from common import constants

def cnn(base_model_name, input_shape, dimensions, train_base_layers):
    """It selects a base model based on the input parameter.
    
    Arguments:
        base_model_name {string} -- A string containing the name of a base model.
        input_shape {(int, int)} -- A tuple indicating the dimensions of model input.
        dimensions {int} -- An integer indicating the size of feature vector.
        train_base_layers {boolean} -- A boolean value to indicate training of base layers.
    """
    #Base model placeholder to be updated in the if/else clause
    base_model = None
    base_model_params = dict(include_top=False, weights='imagenet', input_shape=input_shape)

    if base_model_name == constants.FEATURE_MODELS[0]:
        #Use ResNet to represent images.
        base_model = ResNet(**base_model_params)
    elif base_model_name == constants.FEATURE_MODELS[1]:
        #Use Inception V3
        base_model = InceptionV3(**base_model_params)
    else:
        raise ValueError("Invalid base model: {}".format(base_model_name))

    #Feature model
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(dimensions, activation='softmax')(x)

    #Model object
    model = Model(inputs=base_model.input, outputs=predictions)

    if not train_base_layers:
        #Disable base model training to make sure consistent image representation.
        for layer in base_model.layers:
            layer.trainable = False

    return model