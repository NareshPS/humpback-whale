#Basic imports
from sys import argv, stdout

#To save training data
from pickle import dump as pickle_dump

#Data adjustment
import numpy as np

#Local imports
from common import constants
from model_utils import get_input_labels, get_label_ids, model_fit, load_training_batch, load_training_data, load_pretrained_model
from utils import list_files

batch_size = 2
n_images = 3
n_epochs = 30
model_name = "model_1"
l_rate = 0.001

dataset = "train"
validation_split = 0.2
input_shape = constants.IMG_SHAPE
source_loc = constants.PROCESSED_DATASET_MAPPINGS[dataset]

input_labels = get_input_labels()
label_ids = get_label_ids()
n_classes = len(label_ids)
input_set = list_files(source_loc, n_images)

x, y = load_training_data(source_loc, input_set, input_labels, label_ids)

print(x.shape)
print(y.shape)

model, history = load_pretrained_model(model_name)
