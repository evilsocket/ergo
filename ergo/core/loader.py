import logging as log
import pandas as pd
import numpy as np
import pickle
import threading

class Loader(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def _reader(self, path):
        log.info("loading %s ..." % path)
        if self.dataset.is_flat:
            return pd.read_csv(path, sep = ',', header = None)
        else:
            return pickle.load( open( path, "rb" ) )

    def _worker(self, idx):
        if idx == 0:
            self.dataset.train = self._reader(self.dataset.train_path)
        elif idx == 1:
            self.dataset.test = self._reader(self.dataset.test_path)
        elif idx == 2:
            self.dataset.validation = self._reader(self.dataset.valid_path)

    def load(self):
        threads = ( \
          threading.Thread(target=self._worker, args=(0,)),
          threading.Thread(target=self._worker, args=(1,)),
          threading.Thread(target=self._worker, args=(2,)) 
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join() 

        u = np.concatenate( \
                (self.dataset.validation.iloc[:,0].unique(), 
                self.dataset.test.iloc[:,0].unique(),
                self.dataset.train.iloc[:,0].unique()) )
        self.dataset.n_labels  = len(np.unique(u))

        log.info("detected %d distinct labels", self.dataset.n_labels)
