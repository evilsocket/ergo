import os
import sys
import json
import logging as log

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import model_from_yaml, load_model
from keras.utils.vis_utils import plot_model
from keras.utils.training_utils import multi_gpu_model

from ergo.core.logic import Logic
from ergo.core.utils import clean_if_exist

from ergo.dataset import Dataset
from ergo.templates import Templates

class Project(object):
    @staticmethod
    def create(path):
        log.info("initializing project %s ...", path)
        for filename, data in Templates.items():
            log.info( "creating %s", filename)
            with open( os.path.join(path, filename), 'wt' ) as fp:
                fp.write(data.strip())

    @staticmethod
    def clean(path, full):
        Dataset.clean(path)
        if full:
            # clean everything
            clean_if_exist(path, ( \
                '__pycache__',
                'logs',
                'model.yml',
                'model.png',
                'model.h5',
                'model.fdeep',
                'model.stats',
                'history.json'))

    def __init__(self, path):
        # base info
        self.path  = os.path.abspath(path)
        self.logic = Logic(self.path)
        # model related data
        self.model          = None
        self.accu           = None
        self.model_path     = os.path.join(self.path, 'model.yml')
        self.model_img_path = os.path.join(self.path, 'model.png')
        self.weights_path   = os.path.join(self.path, 'model.h5')
        self.fdeep_path     = os.path.join(self.path, 'model.fdeep')
        # training related data
        self.dataset        = Dataset(self.path)
        self.stats_path     = os.path.join(self.path, 'model.stats')
        self.history_path   = os.path.join(self.path, 'history.json')
        self.history        = None
        self.what           = {
            'train' : "Training --------------------------------------------\n",
            'val'   : "Validation ------------------------------------------\n",
            'test'  : "Test ------------------------------------------------\n"
        }

    def exists(self):
        return os.path.exists(self.path)

    def is_trained(self):
        return os.path.exists(self.weights_path)

    def load(self):
        log.info("loading project %s ..." % self.path)

        if not self.exists():
            return "%s does not exist" % self.path

        err = self.logic.load()
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
            self.model = self.logic.builder(True)
                
        if os.path.exists(self.history_path):
            log.debug("loading history from %s ...", self.history_path)
            with open(self.history_path, 'r') as fp:
                self.history = json.loads(fp.read())
        
        return None
    
    def accuracy_for(self, X, Y, repo_as_dict = False):
        Y_tpred = np.argmax(self.model.predict(X), axis = 1)
        repo    = classification_report(np.argmax(Y, axis = 1), Y_tpred, output_dict = repo_as_dict)
        cm      = confusion_matrix(np.argmax(Y, axis = 1), Y_tpred)
        return repo, cm

    def accuracy(self):
        train, tr_cm = self.accuracy_for(self.dataset.X_train, self.dataset.Y_train)
        test,  ts_cm = self.accuracy_for(self.dataset.X_test, self.dataset.Y_test)
        val,  val_cm = self.accuracy_for(self.dataset.X_val, self.dataset.Y_val)
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

    def _out_stats(self, where):
        for who, header in self.what.items():
            vals = self.accu[who]
            where.write( header )
            where.write( vals[0] )
            where.write("\n\n")
            where.write("confusion matrix:")
            where.write("\n\n")
            where.write("%s\n" % vals[1])
            where.write("\n")

    def _save_stats(self):
        log.info("updating %s ...", self.stats_path)
        with open(self.stats_path, 'w') as fp:
            self._out_stats(fp)

    def prepare(self, filename, p_test, p_val):
        log.info("preparing data from %s ...", filename)
        data = self.logic.prepare_dataset(filename)
        return self.dataset.source(data, p_test, p_val)

    def train(self, gpus):
        # async datasets saver might be running, wait before training
        self.dataset.saver.wait()
        
        # train
        if self.model is None:
            self.model = self.logic.builder(True)

        to_train = self.model
        if gpus > 1:
            log.info("training with %d GPUs", gpus)
            to_train = multi_gpu_model(self.model, gpus=gpus)

        self.history  = self.logic.trainer(to_train, self.dataset).history
        self.accu     = self.accuracy() 

        print("")
        self._out_stats(sys.stdout)

        # save model structure and weights
        self._save_model()
        # save training history
        self._save_history()
        # save model accuracy statistics
        self._save_stats() 

    def null_feature(self, idx):
        input_layer = self.model.layers[0]
        # get_weights returns a copy!
        input_weights = input_layer.get_weights()
        backup_w = input_weights[0][idx,:].copy()
        backup_b = input_weights[1][idx].copy()
        # set to 0 the weights and bias of this input column
        input_weights[0][idx,:] = 0
        input_weights[1][idx] = 0
        input_layer.set_weights(input_weights)
        return backup_w, backup_b

    def restore_feature(self, idx, backup_w, backup_b):
        input_layer = self.model.layers[0]
        input_weights = input_layer.get_weights()
        input_weights[0][idx,:] = backup_w
        input_weights[1][idx] = backup_b
        input_layer.set_weights(input_weights)

    def view(self):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        if self.model is not None:
            self.model.summary()
            log.info("saving model to %s ...", self.model_img_path)
            plot_model( self.model, 
                    to_file = self.model_img_path,
                    show_shapes = True, 
                    show_layer_names = True)
            img = mpimg.imread(self.model_img_path)
            plt.figure()
            plt.imshow(img)

        if self.history is not None:
            plt.figure()
            # Plot training & validation accuracy values
            plt.subplot(2,1,1)
            plt.plot(self.history['acc'])
            plt.plot(self.history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='lower right')
            
            # Plot training & validation loss values
            plt.subplot(2,1,2)
            plt.plot(self.history['loss'])
            plt.plot(self.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper right')

            plt.tight_layout()
        
        plt.show()



