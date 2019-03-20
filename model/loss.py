"""It defines the custom loss functions
"""
#Tensorflow imports
import tensorflow as tf

TRIPLET_LOSS_ALPHA = 0.2

def triplet_loss(X):
    #Unroll anchor, positive and negative features
    anchor, positive, negative = X

    #Compute the distance of positive sample from the anchor
    positive_dist = tf.reduce_sum(
                        tf.square(
                            tf.subtract(anchor, positive)),
                        1)

    #Compute the distance of negative sample from the anchor
    negative_dist = tf.reduce_sum(
                        tf.square(
                            tf.subtract(anchor, negative)),
                        1)

    #Compute the loss
    basic_loss = tf.add(
                    tf.subtract(positive_dist, negative_dist),
                    TRIPLET_LOSS_ALPHA)

    #Clip minimum loss value to 0
    loss = tf.reduce_mean(
                tf.maximum(basic_loss, 0.0),
                0)

    return loss
