import threading
import logging as log

class Saver(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.threads = None

    @staticmethod
    def _worker(v, filename):
        log.info("saving %s ..." % filename)
        v.to_csv(filename, sep = ',', header = None, index = None, chunksize=512)

    def save(self):
        self.threads = [ \
          threading.Thread(target=Saver._worker, args=( self.dataset.train, self.dataset.train_path, )),
          threading.Thread(target=Saver._worker, args=( self.dataset.test, self.dataset.test_path, )),
          threading.Thread(target=Saver._worker, args=( self.dataset.validation, self.dataset.valid_path, )) 
        ]
        for t in self.threads:
            t.start()
    
    def wait(self):
        if self.threads is not None:
            log.info("waiting for datasets saving to complete ...")
            for t in self.threads:
                t.join()
