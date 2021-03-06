#Data processing
from pandas import DataFrame
from collections import defaultdict
from random import sample as random_sample
from random import shuffle as random_shuffle
from common.pandas import to_dict

#Progress bar
from tqdm import tqdm

#Logging
from common import logging

class TupleGeneration(object):
    """It generates Siamese input tuples.
    """
    def __init__(self, label_df, image_col, label_col, output_df_cols):
        """It initializes the required and optional parameters.

        Arguments:
            label_df {A Pandas DataFrame} -- It contains the input names and labels.
            image_col {string} -- The image column name in the dataframe.
            label_col {string} -- The label column name in the dataframe.
            output_df_cols {(string, string, string)} -- The pandas DataFrame headers in the tuple dataframe.
        """

        #Required parameters
        self._label_df = label_df
        self._image_col = image_col
        self._label_col = label_col
        self._output_df_cols = output_df_cols

        #Derived parameters
        self._labelled_images = None

        #Logging
        self._logger = logging.get_logger(__name__)

        self._logger.info(
                        'label_df: %d image_col: %s label_col: %s output_df_cols: %s',
                        len(self._label_df),
                        self._image_col,
                        self._label_col,
                        self._output_df_cols)

    def _get_labelled_images(self):
        """It creates a dictionary of the label to its images.

        Returns:
            [A dict object] -- A mapping from the label to its images.
        """

        #Prepopulated
        if self._labelled_images:
            return self._labelled_images

        #Placeholder for label dictionary
        self._labelled_images = to_dict(self._label_df, self._label_col, self._image_col)

        self._logger.info(
                        'Created label to images dictionary for %d labels',
                        len(self._labelled_images))

        return self._labelled_images

    def _get_images(self):
        """It creates a set of all the images in the dataframe.
        """
        images = set(self._label_df[self._image_col])

        self._logger.info('Unique images: %d', len(images))

        return images

    def get_tuples(self, num_positive_samples, num_negative_samples):
        """It generates a list of tuples (Anchor, Sample, [0|1]) for each image.

        Arguments:
            num_positive_samples {int} -- The number of positive samples.
            num_negative_samples {int} -- The number of negative samples.

        Returns:
            [(string, string, int)] -- A list of sample tuples.
        """

        self._logger.info(
                        'Recieved parameters:: num_positive_samples: %d num_negative_samples: %d',
                        num_positive_samples,
                        num_negative_samples)

        #Images
        images = self._get_images()

        #Label images
        labelled_images = self._get_labelled_images()

        samples = []
        for label, label_images in tqdm(labelled_images.items(), desc = 'Generating input tuples', total = len(labelled_images)):
            #Available samples
            n_avail_positive_samples = min(num_positive_samples, len(label_images))
            n_avail_negative_samples = num_negative_samples

            #Candidates for negative sample
            negative_sample_candidates = images - set(label_images)

            for anchor in label_images:
                #Positive samples
                positive_samples = random_sample(label_images, n_avail_positive_samples)

                #Negative samples
                negative_samples = random_sample(negative_sample_candidates, n_avail_negative_samples)

                #Positive sample tuple
                for p_sample in positive_samples:
                    sample = (anchor, p_sample, 1)
                    samples.append(sample)

                #Negative sample tuple
                for n_sample in negative_samples:
                    sample = (anchor, n_sample, 0)
                    samples.append(sample)

        #Pandas DataFrame
        tuple_df = DataFrame(samples, columns = self._output_df_cols)

        return tuple_df

    def get_triplets(self, num_samples):
        """It generates a list of triplets (Anchor, Positive, Negative) for each image.

        Arguments:
            num_samples {int} -- The number of samples per image

        Returns:
            [(string, string, string)] -- A list of sample triplets
        """

        self._logger.info('Parameters:: num_samples: %d', num_samples)

        #Images
        images = self._get_images()

        #Label images
        labelled_images = self._get_labelled_images()

        #Triplet placeholder
        samples = []

        for _, label_images in tqdm(labelled_images.items(), desc = 'Generating input triplets', total = len(labelled_images)):
            #Available samples
            n_avail_samples = min(num_samples, len(label_images))

            #Candidates for negative sample
            negative_sample_candidates = images - label_images

            for anchor in label_images:
                #Positive samples
                positive_samples = random_sample(label_images, n_avail_samples)

                #Negative samples
                negative_samples = random_sample(negative_sample_candidates, n_avail_samples)

                #Shuffle samples
                random_shuffle(positive_samples)
                random_shuffle(negative_samples)

                for index in range(n_avail_samples):
                    #Create a sample
                    sample = (anchor, positive_samples[index], negative_samples[index])

                    #Append sample to the sample list
                    samples.append(sample)

        #Pandas DataFrame
        triplets = DataFrame(samples, columns = self._output_df_cols)

        return triplets
