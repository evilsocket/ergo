import logging as log
import pandas as pd
import numpy as np

class Loader(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def load(self):
        log.info("loading %s ..." % self.dataset.train_path)
        self.dataset.train = pd.read_csv(self.dataset.train_path, sep = ',', header = None)
        log.info("loading %s ..." % self.dataset.test_path)
        self.dataset.test = pd.read_csv(self.dataset.test_path, sep = ',', header = None)
        log.info("loading %s ..." % self.dataset.valid_path)
        self.dataset.validation = pd.read_csv(self.dataset.valid_path, sep = ',', header = None)

        u = np.concatenate( \
                (self.dataset.validation.iloc[:,0].unique(), 
                self.dataset.test.iloc[:,0].unique(),
                self.dataset.train.iloc[:,0].unique()) )
        self.dataset.n_labels  = len(np.unique(u))

        log.info("detected %d distinct labels", self.dataset.n_labels)
