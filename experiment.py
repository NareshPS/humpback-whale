from keras import backend as K
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Concatenate, Conv2D, Dense, Flatten, Lambda, Reshape
from keras.models import Model, load_model

from keras.applications.densenet import DenseNet121, preprocess_input

BOX_SIZE = 512

def preprocess_image(img):
    return preprocess_input(img)


def get_branch_model(inp_shape):
    model = DenseNet121(input_shape=inp_shape, include_top=False, weights=None, pooling='max')
    return model


def build_model(img_shape, activation='sigmoid'):
    optim = Adam(lr=0.0001)
    branch_model = get_branch_model(img_shape)

    mid = 32
    xa_inp = Input(shape=branch_model.output_shape[1:], name='hm_inp_a')
    xb_inp = Input(shape=branch_model.output_shape[1:], name='hm_inp_b')
    x1 = Lambda(lambda x: x[0] * x[1], name='lambda_1')([xa_inp, xb_inp])
    x2 = Lambda(lambda x: x[0] + x[1], name='lambda_2')([xa_inp, xb_inp])
    x3 = Lambda(lambda x: K.abs(x[0] - x[1]), name='lambda_3')([xa_inp, xb_inp])
    x4 = Lambda(lambda x: K.square(x), name='lambda_4')(x3)
    x = Concatenate(name='concat_1')([x1, x2, x3, x4])
    x = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x = Conv2D(mid, (4, 1), activation='relu', padding='valid', name='hm_conv_2d_1')(x)
    x = Reshape((branch_model.output_shape[1], mid, 1), name='hm_reshape_2')(x)
    x = Conv2D(1, (1, mid), activation='linear', padding='valid', name='hm_conv_2d_2')(x)
    x = Flatten(name='flatten')(x)

    # Weighted sum implemented as a Dense layer.
    x = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model(inputs = [xa_inp, xb_inp], outputs = x, name='head')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a = Input(shape=img_shape)
    img_b = Input(shape=img_shape)
    xa = branch_model(img_a)
    xb = branch_model(img_b)
    x = head_model([xa, xb])
    model = Model(inputs = [img_a, img_b], outputs = x, name='full_model')
    model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])

    return model, branch_model, head_model

img_shape = (512, 512, 3)

model, _, _ = build_model(img_shape)
model.save('siamese_densenet.batch.0.epoch.0.h5')
