import os
import threading
import logging as log

import numpy as np
import pandas as pd

from keras.utils import to_categorical
from ergo.utils import clean_if_exist

class Dataset(object):
    @staticmethod 
    def clean(path):
        clean_if_exist(path, ('data-train.csv', 'data-test.csv', 'data-validation.csv'))

    @staticmethod
    def optimize(path, reuse = 0.15, output = None):
        log.info("optimizing dataset %s (reuse ratio is %.1f%%) ...", path, reuse * 100.0)

        data  = pd.read_csv(path, sep = ',', header = None)
        n_tot = len(data)

        log.info("loaded %d total samples", n_tot)

        unique  = data.drop_duplicates()
        n_uniq  = len(unique)
        n_reuse = int( n_uniq * reuse )
        reuse   = data.sample(n=n_reuse).reset_index(drop = True)

        log.info("found %d unique samples, reusing %d samples from the main dataset", n_uniq, n_reuse)

        out          = pd.concat([reuse, unique]).sample(frac=1).reset_index(drop=True)
        outpath      = output if output is not None else path
        n_out        = len(out)
        optimization = 100.0 - (n_out * 100.0) / float(n_tot)

        log.info("optimized dataset has %d records, optimization is %.2f%%", n_out, optimization)

        log.info("saving %s ...", outpath)
        out.to_csv( outpath, sep = ',', header = None, index = None)


    def __init__(self, path):
        self.path       = os.path.abspath(path)
        self.train_path = os.path.join(self.path, 'data-train.csv')
        self.test_path  = os.path.join(self.path, 'data-test.csv')
        self.valid_path = os.path.join(self.path, 'data-validation.csv')
        self.n_labels   = 0
        self.train      = None
        self.test       = None
        self.validation = None
        self.X_train    = None
        self.Y_train    = None
        self.X_test     = None
        self.Y_test     = None
        self.X_val      = None
        self.Y_val      = None

    def exists(self):
        return os.path.exists(self.train_path) and \
               os.path.exists(self.test_path) and \
               os.path.exists(self.valid_path)

    def _set_xys(self):
        self.X_train = self.train.values[:,1:]
        self.Y_train = to_categorical(self.train.values[:,0], self.n_labels)
        self.X_test  = self.test.values[:,1:]
        self.Y_test  = to_categorical(self.test.values[:,0], self.n_labels)
        self.X_val   = self.validation.values[:,1:]
        self.Y_val   = to_categorical(self.validation.values[:,0], self.n_labels)

    def _save_thread(self, v, filename):
        log.info("saving %s ..." % filename)
        v.to_csv(filename, sep = ',', header = None, index = None)

    def _save(self):
        threads = ( \
          threading.Thread(target=self._save_thread, args=( self.train, self.train_path, )),
          threading.Thread(target=self._save_thread, args=( self.test, self.test_path, )),
          threading.Thread(target=self._save_thread, args=( self.validation, self.valid_path, )) 
        )

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        self._set_xys()

    def load(self):
        log.info("loading %s ..." % self.train_path)
        self.train = pd.read_csv(self.train_path, sep = ',', header = None)

        log.info("loading %s ..." % self.test_path)
        self.test = pd.read_csv(self.test_path, sep = ',', header = None)

        log.info("loading %s ..." % self.valid_path)
        self.validation = pd.read_csv(self.valid_path, sep = ',', header = None)

        u = np.concatenate( \
                (self.validation.iloc[:,0].unique(), 
                self.test.iloc[:,0].unique(),
                self.train.iloc[:,0].unique()) )

        self.n_labels  = len(np.unique(u))
        self._set_xys()
    
    def source(self, data, p_test, p_val):
        log.info("generating train, test and validation datasets (test=%f validation=%f) ...", 
                p_test, 
                p_val)

        dataset = data.sample(frac = 1).reset_index(drop = True)
        n_tot   = len(dataset)
        n_train = int(n_tot * ( 1 - p_test - p_val))
        n_test  = int(n_tot * p_test)
        n_val   = int(n_tot * p_val)

        self.n_labels   = len(dataset.iloc[:,0].unique())
        self.train      = dataset.head(n_train)
        self.test       = dataset.head(n_train + n_test).tail(n_test)
        self.validation = dataset.tail(n_val)

        self._save()
