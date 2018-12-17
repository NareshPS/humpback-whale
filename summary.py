"""Utility to extract valuable information from a model.
"""
from matplotlib import pyplot as plt

class ModelCnnSummary:
    visual_layer_prefix = 'conv'
    def __init__(self, model):
        self._model = model

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
            if layer.name.startswith(ModelCnnSummary.visual_layer_prefix):
                weights = layer.get_weights()[0]
                print("({}, {})".format(layer.name, weights.shape))

