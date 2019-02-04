#Unittests
import unittest as ut

#Local imports
from model_utils import load_pretrained_model
from visualization import HistoryInsights
from visualization import ModelInsights
from visualization import WeightInsights
from visualization import PlottingUtils
from common import ut_constants

model_name = "cnn_model2d_1"
store = ut_constants.DATA_STORE

class TestHistoryInsights(ut.TestCase):
    def test_accuracy(self):
        _, history = load_pretrained_model(model_name, store)

        insights = HistoryInsights(history)
        insights.accuracy()
    
    def test_loss(self):
        _, history = load_pretrained_model(model_name, store)

        insights = HistoryInsights(history)
        insights.loss()

class TestPlottingUtils(ut.TestCase):
    grid_dimensions = (5, 3)

    def test_get_plot_axes(self):
        plot_id_locations = {
                                0: (0, 0),
                                2: (0, 2),
                                5: (1, 2),
                                14: (4, 2)
                            }
        
        for plot_id, expected_loc in plot_id_locations.items():
            location = PlottingUtils.get_plot_axes(TestPlottingUtils.grid_dimensions, plot_id)
            self.assertEqual(location, expected_loc, "Got unexpected location: {} for plot_id: {}".format(location, plot_id))

    def test_create_plot_d(self):
        grid_dimensions = (5, 3)

        figure, axes = PlottingUtils.create_plot_d(grid_dimensions)
        self.assertEqual(len(axes), 15, "Expected 15 axes for the grid with dimensions {}".format(grid_dimensions))
        self.assertEqual(len(figure.get_axes()), 15, "Expected 15 axes for the grid with dimensions {}".format(grid_dimensions))

    def test_create_plot_n(self):
        n_graphs = 6
        figure, axes = PlottingUtils.create_plot_n(n_graphs)

        self.assertEqual(len(axes), 6, "Expected 6 axes for {} graphs".format(n_graphs))
        self.assertEqual(len(figure.get_axes()), 6, "Expected 6 axes for {} graphs".format(n_graphs))

class TestWeightInsights(ut.TestCase):
    def test_get_conv_weights(self):
        model, _ = load_pretrained_model(model_name, store)
        model_insights = ModelInsights(model)

        weights = model_insights.get_conv_weights()
        for l_name, l_weights in weights.items():
            print(l_name)
            print(l_weights)

class TestModelInsights(ut.TestCase):
    def test_summary(self):
        model, _ = load_pretrained_model(model_name, store)

        summary = ModelInsights(model)
        summary.summary()

if __name__ == "__main__":
    ut.main()