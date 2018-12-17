#Unittest
import unittest as ut

#Local imports
from model_utils import load_pretrained_model
from summary import ModelCnnSummary

model_name = "model_1"

class TestModelCnnSummary(ut.TestCase):
    def test_summary(self):
        model, _ = load_pretrained_model(model_name)

        summary = ModelCnnSummary(model)
        summary.summary()

if __name__ == "__main__":
    ut.main()