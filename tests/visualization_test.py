#Unittests
import unittest as ut

#Local imports
from model_utils import load_pretrained_model
from visualization import HistoryInsights
from visualization import ModelSummary
from visualization import WeightInsights
from visualization import PlottingUtils
from common import ut_constants

model_name = "cnn_model2d_1"
store = ut_constants.UT_DATA_STORE

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

class TestWeightInsights(ut.TestCase):
    def test_get_conv_weights(self):
        weights = []
        model, _ = load_pretrained_model(model_name, store)
        for layer in model.layers:
            None

class TestModelSummary(ut.TestCase):
    def test_summary(self):
        model, _ = load_pretrained_model(model_name, store)

        summary = ModelSummary(model)
        summary.summary()

if __name__ == "__main__":
    ut.main()