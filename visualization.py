"""Utility methods that help visualize the model training and accuracy.
"""
#Plotting
from matplotlib import pyplot as plt

class HistoryInsights:
    """Support to analyze the training history to gain better insight.
    """
    def __init__(self, history):
        #History object must exist to plot accuracy.
        if history is None:
            raise ValueError("History object must exist to plot accuracy.")

        self._history = history

        #Create a figure to accomodate accuracy and loss plots.
        self._figure, self._axes = plt.subplots(1, 2, figsize = (10, 3))
        
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