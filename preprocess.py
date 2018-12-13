"""Entry point for preprocessing image datasets.

    Usage:: python preprocess.py <train/test> <batch_size> <[int] Count of images to preprocess. If not specified, process all.>
"""

#Basic imports
from sys import argv, stdout

#Progress bar
from tqdm import tqdm

#Local imports
from common import constants
from utils import list_files, load_images_batch, resize_images, store_dataset

if __name__ == "__main__":
    """Preprocesses raw input images to generate the training and test datasets.

    Usage:: python preprocess.py <train/test> <batch_size> <[int] Count of images to preprocess. If not specified, process all.>
    """

    n_args = len(argv)
    if n_args not in [3, 4]:
        print("Syntax error. Usage:: python preprocess.py <train/test> <batch_size> <[int] Count of images to preprocess. If not specified, process all.>") 
        exit(-1)

    dataset = argv[1]
    batch_size = int(argv[2])
    n_images = int(argv[3]) if len(argv) == 4 else None
    source_loc = constants.RAW_DATASET_MAPPINGS[dataset]
    target_loc = constants.PROCESSED_DATASET_MAPPINGS[dataset]

    #Scan all images
    img_files = list_files(source_loc, n_images)
    print("Preprocessing {n_images} images.".format(n_images = n_images))
    
    with tqdm(total = len(img_files), file=stdout) as progress_bar:
        for batch_id, img_pairs in enumerate(load_images_batch(source_loc, img_files, batch_size = batch_size)):
            img_files, imgs = img_pairs
            progress_bar.set_description("Processing batch: {batch_id}".format(batch_id = batch_id))
            #Resize images to keep all images for a consistent size.
            resized_imgs = resize_images(imgs, constants.IMG_SHAPE)

            #Save training images to be readily available to be trained.
            store_dataset(target_loc, img_files, resized_imgs)

            progress_bar.update(len(img_files))