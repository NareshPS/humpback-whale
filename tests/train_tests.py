#Local imports
from common import constants
from models import cnn_model_2d_1
from model_utils import get_input_labels, get_label_ids, model_fit
from utils import list_files, split_dataset

dataset = "train"
v_split = 0.2
n_images = 6
batch_size = 2
l_rate = 0.001
n_epochs = 1
input_shape = constants.IMG_SHAPE
source_loc = constants.PROCESSED_DATASET_MAPPINGS[dataset]

#Input data parameters
input_labels = get_input_labels()
label_ids = get_label_ids()
n_classes = len(label_ids)
input_set = list_files(source_loc, n_images)
n_images = len(input_set)

#Training and validation sets
train_set, validation_set = split_dataset(input_set, v_split)
print("Training set: {t_size} validation set: {v_size}".format(t_size = len(train_set), v_size = len(validation_set)))

model = cnn_model_2d_1(input_shape, n_classes, l_rate)
model.summary()

#Train the model
history = model_fit(model, source_loc, train_set, validation_set, input_labels, label_ids, batch_size, n_epochs)

print(history.history.keys())