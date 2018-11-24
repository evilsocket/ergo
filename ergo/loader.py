import logging as log
import pandas as pd
import numpy as np
import threading

class Loader(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def _worker(self, idx):
        if idx == 0:
            log.info("loading %s ..." % self.dataset.train_path)
            self.dataset.train = pd.read_csv(self.dataset.train_path, sep = ',', header = None)
        elif idx == 1:
            log.info("loading %s ..." % self.dataset.test_path)
            self.dataset.test = pd.read_csv(self.dataset.test_path, sep = ',', header = None)
        elif idx == 2:
            log.info("loading %s ..." % self.dataset.valid_path)
            self.dataset.validation = pd.read_csv(self.dataset.valid_path, sep = ',', header = None)

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
