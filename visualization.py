"""Utility methods that help visualize the model training and accuracy.
"""
#Plotting
from matplotlib import pyplot as plt

from image import operations as img_ops

class HistoryInsights:
    """Support to analyze the training history to gain better insight.
    """
    def __init__(self, history):
        #History object must exist to plot accuracy.
        if history is None:
            raise ValueError("History object must exist to plot accuracy.")

        self._history = history

        #Create a figure to accomodate accuracy and loss plots.
        self._figure, self._axes = PlottingUtils.create_plot((1, 2))
        
    def accuracy(self):
        self._plot(
                0, #plot_id
                ['acc', 'val_acc'],
                ['Training', 'Validation'],
                title = 'Training and validation accuracies',
                ylabel = 'Accuracy')
    
    def loss(self):
        self._plot(
                1, #plot_id
                ['loss', 'val_loss'],
                ['Training', 'Validation'],
                title = 'Training and validation losses',
                ylabel = 'Loss')
        
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
    def __init__(self, weights):
        """Convolve the input image witht he 
        
        Arguments:
            weight {Numpy array} -- A numpy array with kernel weights.
        """
        #Weight object must exist to derive insights.
        if weights is None:
            raise ValueError("Valid weights are required to analyze.")

        self._weights = weights
        self._conv_weights = None

    def get_conv_weights(self):
        """Getter for _conv_weights.
        
        Returns:
            A numpy array -- A numpy array containing the convolutional weights.
        """
        return self._extract_conv_weights() if self._conv_weights is None else self._conv_weights

    def _extract_conv_weights(self):
        """Extracts the trained weights from covolutional layers.
        It saves them in an instance variable for subsequenct accesses.
        
        Returns:
            A numpy array -- A numpy array containing the convolutional weights.
        """
        #Calculate weights here
        conv_weights = None
        return conv_weights
    
    def convole(self, image, plot):
        """Convolve the input image witht he 
        
        Arguments:
            image {[type]} -- [description]
            kernel {[type]} -- [description]
            plot {[type]} -- [description]
        """
        convolutions = img_ops.convolve([image], kernel)
        plot.plot(convolutions[0])

class PlottingUtils:
    """A collection of utilities to supplement matplotlib.
    """
    @staticmethod
    def create_plot(dimensions):
        """Creates a grid of plots based on the input dimensions.
        
        Arguments:
            dimensions {(rows, cols)} -- A tuple to indicate the dimensions of the plot.
        """
        figure, axes = plt.subplots(dimensions[0], dimensions[1], figsize = dimensions*5)

        return figure, axes


    @staticmethod
    def get_plot_axes(grid_dimensions, plot_id):
        """It calculates the position for a plot based on the 0 indexed plot id.
        
        Arguments:
            grid_dimensions {(int, int)} -- A two dimensional tuple to indicate the number of rows and the columns.
            plot_id {int} -- An integer to represent the plot id.
            
        Returns:
            {(int, int)} -- A tuple to indicate the position of a plot on a two dimensional grid.
        """
        n_cols = grid_dimensions[1]
        
        row_id = int(plot_id / n_cols)
        col_id = int(plot_id - (row_id * n_cols))
        
        return row_id, col_id