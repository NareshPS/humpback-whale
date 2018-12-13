"""Utility methods that help visualize the model training and accuracy.
"""
#Plotting
from matplotlib import pyplot as plt

class TrainingInsights:
    """Support to analyze keras models, and history to gain better insight into the training.
    """
    def __init__(self, model, history = None):
        self._model = model
        self._history = history

        #Create a figure to accomodate accuracy and loss plots.
        self._history_figure, self._history_axes = plt.subplots(1, 2, figsize = (10, 3))
        
    def accuracy(self):
        self._plot_history(
                0, #plot_id
                ['acc', 'val_acc'],
                ['Training', 'Validation'],
                title = 'Training and validation accuracies',
                ylabel = 'Accuracy')
    
    def loss(self):
        self._plot_history(
                1, #plot_id
                ['loss', 'val_loss'],
                ['Training', 'Validation'],
                title = 'Training and validation losses',
                ylabel = 'Loss')
        
    def _plot_history(self, plot_id, params, legend, title = None, ylabel = None):
        #History object must exist to plot accuracy.
        if self._history is None:
            raise ValueError("History object must exist to plot accuracy.")
            
        if len(params) == 0:
            raise ValueError("params must contain a list of history items.")

        #Initialize figure and axes variables.
        figure = self._history_figure
        axes = self._history_axes[plot_id]
 
        #Set axes data
        for param in params:
            axes.plot(self._history[param])

        #Set axes parameters
        axes.legend(legend, loc='upper left')
        axes.set_xlabel('Epoch')
        
        if title is not None:
            axes.set_title(title)
            
        if ylabel is not None:
            axes.set_ylabel(ylabel)

        #Beautify and display
        figure.tight_layout()