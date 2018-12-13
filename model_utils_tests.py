from common import constants
from utils import list_files
from model_utils import get_image_labels, get_label_ids, load_training_batch

import numpy as np

image_labels = get_image_labels()
label_ids = get_label_ids()
source_loc = constants.PROCESSED_DATASET_MAPPINGS["train"]
n_images = 50
batch_size = 32
img_files = list_files(source_loc, n_images)

for batch_id, train_data in enumerate(load_training_batch(source_loc, img_files, batch_size, image_labels, label_ids)):
    x, y = train_data
    l_index = 10
    g_index = batch_id*batch_size + l_index
    img_file = img_files[g_index]
    image_label = image_labels[img_file]
    print(x[0].shape)
    print(np.nonzero(y[l_index])[0][0])
    print(label_ids[image_label])
    print(img_file)
    print(image_label)