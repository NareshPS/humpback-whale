"""Utility to extract valuable information from a model.
"""

class ModelSummary:
    def __init__(self, model):
        self._model = model

    def summary(self):
        #Useful objects
        model = self._model
        layers = model.layers
        weights = model.weights

        print("Layers: {} Weights: {}".format(len(layers), len(weights)))
        print("Layer shapes: {}".format(layers))

