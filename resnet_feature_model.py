#Keras support
from keras.applications.resnet50 import ResNet50 as ResNet
from keras.layers import Input, Dense, Flatten, BatchNormalization, Activation, Dropout
from keras.models import Model, Sequential

def resnet_feature_model(input_shape, feature_dims):
    #Use ResNet to represent images.
    base_model = ResNet(include_top=False, weights='imagenet', input_shape=input_shape)

    #Disable base model training to make sure consistent image representation.
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(feature_dims, activation="softmax"))

    return model
