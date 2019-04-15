import threading
import logging as log
import pickle

class Saver(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.threads = None

    @staticmethod
    def _worker(v, filename, flat):
        log.info("saving %s ..." % filename)

        if not flat:
            pickle.dump( v, open( filename, "wb" ) )
        else:
            v.to_csv(filename, sep = ',', header = None, index = None, chunksize=512)

    def save(self):
        train_path = self.dataset.train_path
        test_path  = self.dataset.test_path
        valid_path = self.dataset.valid_path
        flat = self.dataset.is_flat

        if not self.dataset.is_flat:
            train_path = train_path.replace('.csv', '.pkl')
            test_path = test_path.replace('.csv', '.pkl')
            valid_path = valid_path.replace('.csv', '.pkl')

        self.threads = [ \
          threading.Thread(target=Saver._worker, args=( self.dataset.train, train_path, flat, )),
          threading.Thread(target=Saver._worker, args=( self.dataset.test, test_path, flat, )),
          threading.Thread(target=Saver._worker, args=( self.dataset.validation, valid_path, flat, )) 
        ]

        for t in self.threads:
            t.start()
    
    def wait(self):
        if self.threads is not None:
            log.info("waiting for datasets saving to complete ...")
            for t in self.threads:
                t.join()
