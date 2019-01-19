"""It provides utilities to plot statistics.
"""

#Plotting
from matplotlib import pyplot as plt

class SinglePlot:
    """A collection of utilities to supplement matplotlib.
    """
    def __init__(self, size = (5,5)):
        """It initializes class parameters, and creates a plotting figure.
        
        Arguments:
            dimensions {(rows, cols)} -- A tuple to indicate the dimensions of the plot.
            size {(width, height)} -- A tuple to indicate the size of each plotted element.
        """
        
        #Required parameters
        self._size = size

        #Plot
        self._figure, self._axes = plt.subplots(figsize = size)

    def plot(self, data):
        keys = data.keys
        n_keys = len(data)

        key_id = 0