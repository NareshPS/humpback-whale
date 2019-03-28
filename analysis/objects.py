"""It defines objects that encapsulate training statistics. These can be used to plot them.
"""

#Plotting
from matplotlib import pyplot as plt
from math import sqrt
import numpy as np

#Perform convolutions
from image import operations as img_ops

#Keras imports
from keras import backend as K

class History:
    """Support to analyze metrics to get a better understanding of the training process.
    """

    def __init__(self, history):
        #History object must exist to plot accuracy.
        if history is None:
            raise ValueError("History object must exist.")

        self._history = history.history

    def accuracy(self, acc_cols = None):
        """It pulls the accuracy metrics for the input columns.

        Keyword Arguments:
            cols {[string]} -- It is a list of column names. (default: {None})

        Raises:
            ValueError -- It raises a ValueError for invalid column names.

        Returns:
            ([], []) -- A tuple of metrics values and their names.
        """
        if acc_cols is None:
            acc_cols = self._acc_cols
        else:
            acc_cols = set(acc_cols)

        #Validate the input metrics set
        if not set(acc_cols).issubset(self._acc_cols):
            raise ValueError(
                    "acc_cols: {} must contain valid values. Allowed values are: {}".format(
                            acc_cols,
                            self._acc_cols))

        return {col:self._history[col] for col in acc_cols}

    def loss(self, loss_cols = None):
        """It pulls the loss metrics for the input columns.

        Keyword Arguments:
            cols {[string]} -- It is a list of column names. (default: {None})

        Raises:
            ValueError -- It raises a ValueError for invalid column names.

        Returns:
            ([], []) -- A tuple of metrics values and their names.
        """
        if loss_cols is None:
            loss_cols = self._loss_cols
        else:
            loss_cols = set(loss_cols)

        #Validate the input metrics set
        if not set(loss_cols).issubset(self._loss_cols):
            raise ValueError(
                    "loss_cols: {} must contain valid values. Allowed values are: {}".format(
                            loss_cols,
                            self._loss_cols))
        return {col:self._history[col] for col in loss_cols}

    def _plot(self, plot_id, params, legend, title = None, ylabel = None):    
        if len(params) == 0:
            raise ValueError("params must contain a list of history items.")

        #Initialize figure and axes variables.
        figure = self._figure
        axes = self._axes[plot_id]

        #Set axes data
        for param in params:
            axes.plot(self._history[param])

        #Set axes parameters
        axes.legend(legend, loc='lower right')
        axes.set_xlabel('Epoch')

        if title is not None:
            axes.set_title(title)

        if ylabel is not None:
            axes.set_ylabel(ylabel)

        #Beautify and display
        figure.tight_layout()

class WeightInsights:
    """Support to analyze the weights to gain insight.
    """
    def __init__(self, model):
        """Convolve the input image witht he 

        Arguments:
            model {A ModelInsights object} -- A model insights object.
        """
        #Weight object must exist to derive insights.
        if model is None:
            raise ValueError("A valid ModelInsights object is required to analyze.")

        self._model = model
        self._conv_weights = self._model.get_conv_weights()

    def convole(self, image, plot):
        """Convolve the input image with the convolutional weights

        Arguments:
            image {An image object} -- An image object to convolve.
            plot {A plotting object} -- The convolved image is plotted using this object.
        """
        convolutions = img_ops.convolve([image], self._model.get_conv_weights())

        print(convolutions.shape)

class ModelInsights:
    visual_layer_prefix = 'conv'
    def __init__(self, model):
        self._model = model

    def _get_conv_weights(self):
        """It extracts the trained weights from covolutional layers.

        Returns:
            A numpy array -- A numpy array containing the convolutional weights.
        """
        #Iterate through all layers to get the weights
        return {layer.name: layer.get_weights()[0] for layer in self._model.layers if self._is_conv_layer(layer.name)}

    def _get_layer_names(self):
        #Iterate through all layers to fetch their names.
        return [layer.name for layer in self._model.layers]

    def _is_conv_layer(self, layer_name):
        return True if layer_name.startswith(self.visual_layer_prefix) else False

    def visualize_layer(self, layer_name, images):
        """It applies activations to input images up to a convolution layer.

        Arguments:
            layer_name {string} -- The name of layer whose activations are to be visualized.
            images {[Image objects]} -- A list of images.

        Raises:
            ValueError -- It is raise if
            1- The input layer name is invalid.
            2- The input layer name is not a convolution layer.
        """

        layer_names = self._get_layer_names()
        r_count = len(images)

        if layer_name not in layer_names:
            raise ValueError("Invalid layer name: {}".format(layer_name))

        inputs = [self._model.layers[0].input]
        outputs = [self._model.get_layer(layer_name).output]

        #Create a keras function from input -> output.
        get_activations = K.function(inputs, outputs)

        #Compute activations
        activations = get_activations(images)

        #Plot a grid of activations through each filter.
        for image_id in range(r_count):
            plt.figure()
            image_act = activations[0][image_id, :, :, :]
            n_filters = image_act.shape[-1]

            cols = int(sqrt(n_filters))
            rows = int((n_filters + cols - 1)/cols)

            image = images[image_id, :, :, :]
            figure, axes = plt.subplots(rows, cols, figsize = (12, 12))

            for fid in range(n_filters):
                location = PlottingUtils.get_plot_axes((rows, cols), fid)
                image = np.asarray(image_act[:, :, fid])
                axes[location].imshow(image, aspect='auto')

            figure.tight_layout()
            figure.savefig("image_{}.png".format(image_id), dpi=100)

    def summary(self):
        #Useful objects
        model = self._model
        layers = model.layers

        print("Layers: {}".format(len(layers)))

        print("\nNames and shapes.")
        for layer in layers:
            print("({}, {})".format(layer.name, layer.output_shape))

        print("\nNames and weights.")
        for layer in layers:
            if layer.name.startswith(ModelInsights.visual_layer_prefix):
                weights = layer.get_weights()[0]
                print("({}, {})".format(layer.name, weights.shape))
