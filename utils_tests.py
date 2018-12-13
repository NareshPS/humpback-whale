from common import constants
import utils

files = utils.list_files(constants.RAW_DATASET_MAPPINGS["train"], 5)
print("Count: {count}".format(count = len(files)))
print(files)