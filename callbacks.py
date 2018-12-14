"""Custom callbacks to collect training summaries.
"""

#Keras imports for callbacks
from keras import backend as K
from keras.callbacks import Callback

class SummaryCallback(Callback):
    """Collects weight summaries per epoch.
    """
    def __init__(self):
        super(SummaryCallback, self).__init__()
        self.weights = []
        self.tf_session = K.get_session()
            
    def on_epoch_end(self, epoch, logs=None):
        None
    
    def on_train_end(self, logs=None):
        None