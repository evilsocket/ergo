import os
import shutil
import json
import logging as log
from importlib.machinery import SourceFileLoader

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import model_from_yaml, load_model
from keras.utils.vis_utils import plot_model
from keras.utils.training_utils import multi_gpu_model

from ergo.dataset import Dataset
from ergo.templates import Templates

class Project(object):
    @staticmethod
    def create(path):
        log.info("initializing project %s ...", path)
        for filename, data in Templates.items():
            log.info( "creating %s", filename)
            with open( os.path.join(path, filename), 'wt' ) as fp:
                fp.write(data)

    @staticmethod
    def clean(path, full):
        if not full:
            # clean dataset files only
            Dataset.clean(path)
        else:
            # clean everything
            keep = Templates.keys()
            for fname in os.listdir(path):
                if fname not in keep:
                    fname = os.path.join(path, fname)
                    log.info("removing %s", fname)
                    if os.path.isdir(fname):
                        shutil.rmtree(fname)
                    else:
                        os.remove(fname)

    def __init__(self, path):
        self.path           = os.path.abspath(path)
        self.builder_path   = os.path.join(self.path, 'model.py')
        self.model_path     = os.path.join(self.path, 'model.yml')
        self.model_img_path = os.path.join(self.path, 'model.png')
        self.weights_path   = os.path.join(self.path, 'model.h5')
        self.fdeep_path     = os.path.join(self.path, 'model.fdeep')
        self.stats_path     = os.path.join(self.path, 'model.stats')
        self.history_path   = os.path.join(self.path, 'history.json')
        self.trainer_path   = os.path.join(self.path, 'train.py')
        self.dataset        = Dataset(self.path)
        self.model          = None
        self.history        = None
    
    def exists(self):
        return os.path.exists(self.path)

    def is_trained(self):
        return os.path.exists(self.weights_path)

    def _load_file(self, name, symbol):
        log.debug("loading symbol %s from %s ..." % (symbol, name))
        doer = SourceFileLoader("",name).load_module()
        if symbol not in doer.__dict__:
            return None, "%s does not define a %s function" % (name, symbol)
        return doer.__dict__[symbol], None

    def load(self):
        log.info("loading project %s ..." % self.path)

        if not self.exists():
            return "%s does not exist" % self.path

        self.builder, err = self._load_file(self.builder_path, 'build_model')
        if err is not None:
            return err

        self.trainer, err = self._load_file(self.trainer_path, 'train_model')
        if err is not None:
            return err

        if os.path.exists(self.weights_path):
            log.debug("loading model from %s ...", self.weights_path)
            self.model = load_model(self.weights_path)
            # https://github.com/keras-team/keras/issues/6462
            self.model._make_predict_function()

        elif os.path.exists(self.model_path):
            log.debug("loading model from %s ...", self.model_path)
            with open(self.model_path, 'r') as fp:
                self.model = model_from_yaml(fp.read())

        else:
            self.model = self.builder(True)
                
        if os.path.exists(self.history_path):
            log.debug("loading history from %s ...", self.history_path)
            with open(self.history_path, 'r') as fp:
                self.history = json.loads(fp.read())
        
        return None
    
    def _accuracy_for(self, X, Y):
        Y_tpred = np.argmax(self.model.predict(X), axis = 1)
        repo    = classification_report(np.argmax(Y, axis = 1), Y_tpred)
        cm      = confusion_matrix(np.argmax(Y, axis = 1), Y_tpred)
        return repo, cm

    def accuracy(self):
        train, tr_cm = self._accuracy_for(self.dataset.X_train, self.dataset.Y_train)
        test,  ts_cm = self._accuracy_for(self.dataset.X_test, self.dataset.Y_test)
        val,  val_cm = self._accuracy_for(self.dataset.X_val, self.dataset.Y_val)
        return {'train': (train, tr_cm), 
                'test': (test, ts_cm), 
                'val': (val, val_cm)}

    def _save_model(self):
        log.info("updating %s ...", self.model_path)
        with open( self.model_path, 'w' ) as fp:
            fp.write(self.model.to_yaml())

        log.info("updating %s ...", self.weights_path)
        self.model.save(self.weights_path)

    def _save_history(self):
        log.info("updating %s ...", self.history_path)
        with open(self.history_path, 'w') as fp:
            json.dump(self.history, fp) 

    def _save_stats(self):
        log.info("updating %s ...", self.stats_path)
        acc = self.accuracy()
        with open(self.stats_path, 'w') as out:
            out.write("[*] Training stats:\n")
            out.write(acc['train'][0]+"\n")
            out.write("%s\n" % acc['train'][1])
            out.write("\n\n[*] Validation stats:\n")
            out.write(acc['val'][0]+"\n")
            out.write("%s\n" % acc['val'][1])
            out.write("\n\n[*] Test stats:\n")
            out.write(acc['test'][0]+"\n")
            out.write("%s\n" % acc['test'][1])

    def train(self, gpus):
        # train
        if self.model is None:
            self.model = self.builder(True)

        to_train = self.model
        if gpus > 1:
            log.info("training with %d GPUs", gpus)
            to_train = multi_gpu_model(self.model, gpus=gpus)

        self.history = self.trainer(to_train, self.dataset).history
        # save model structure and weights
        self._save_model()
        # save training history
        self._save_history()
        # save model accuracy statistics
        self._save_stats() 

    def view(self):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        if self.model is not None:
            self.model.summary()
            log.info("saving model to %s ...", self.model_img_path)
            plot_model(self.model, to_file=self.model_img_path)
            img = mpimg.imread(self.model_img_path)
            plt.figure()
            plt.imshow(img)

        if self.history is not None:
            plt.figure()
            # Plot training & validation accuracy values
            plt.plot(self.history['acc'])
            plt.plot(self.history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.figure()
            # Plot training & validation loss values
            plt.plot(self.history['loss'])
            plt.plot(self.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
        
        plt.show()



