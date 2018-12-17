#Unittests
import unittest as ut

#Local imports
from model_utils import load_pretrained_model
from visualization import HistoryInsights
from visualization import PlottingUtils

model_name = "model_1"

class TestHistoryInsights(ut.TestCase):
    def test_accuracy(self):
        _, history = load_pretrained_model(model_name)

        insights = HistoryInsights(history)
        insights.accuracy()
    
    def test_loss(self):
        _, history = load_pretrained_model(model_name)

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

if __name__ == "__main__":
    ut.main()