import os
import sys
import json
import re
import logging as log

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

from keras.models import model_from_yaml, load_model
from keras.utils.training_utils import multi_gpu_model

from ergo.core.utils import serialize_classification_report, serialize_cm
from ergo.core.logic import Logic
from ergo.dataset import Dataset

class Project(object):
    def __init__(self, path):
        # base info
        self.path  = os.path.abspath(path)
        self.logic = Logic(self.path)
        # model related data
        self.model          = None
        self.accu           = None
        self.model_path     = os.path.join(self.path, 'model.yml')
        self.weights_path   = os.path.join(self.path, 'model.h5')
        self.fdeep_path     = os.path.join(self.path, 'model.fdeep')
        # training related data
        self.dataset         = Dataset(self.path)
        self.txt_stats_path  = os.path.join(self.path, 'stats.txt')
        self.json_stats_path = os.path.join(self.path, 'stats.json')
        self.history_path    = os.path.join(self.path, 'history.json')
        self.history         = None
        self.what            = {
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

    def _emit_txt_stats(self, where):
        for who, header in self.what.items():
            vals = self.accu[who]
            where.write( header )
            where.write( vals[0] )
            where.write("\n\n")
            where.write("confusion matrix:")
            where.write("\n\n")
            where.write("%s\n" % vals[1])
            where.write("\n")

    def _emit_json_stats(self, where):
        stats = {}
        for who in self.what:
            report, cm = self.accu[who]
            stats[who] = { 
                'accuracy': serialize_classification_report(report), 
                'cm': serialize_cm(cm)
            }
        json.dump(stats, where)

    def _save_stats(self):
        log.info("updating %s ...", self.txt_stats_path)
        with open(self.txt_stats_path, 'w') as fp:
            self._emit_txt_stats(fp)

        log.info("updating %s ...", self.json_stats_path)
        with open(self.json_stats_path, 'wt') as fp:
            self._emit_json_stats(fp)

    def _from_file(self, filename):
        log.info("preparing data from %s ...", filename)
        return self.logic.prepare_dataset(filename)

    def prepare(self, source, p_test, p_val, shuffle = True):
        data = self._from_file(source)
        log.info("data shape: %s", data.shape)
        return self.dataset.source(data, p_test, p_val, shuffle)

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

        past = self.history.copy() if self.history is not None else None
        present = self.logic.trainer(to_train, self.dataset).history

        if past is None:
            self.history = present
        else:
            self.history = {}
            for name, past_values in past.items():
                self.history[name] = past_values + present[name]

        self.accu = self.accuracy()

        print("")
        self._emit_txt_stats(sys.stdout)

        # save model structure and weights
        self._save_model()
        # save training history
        self._save_history()
        # save model accuracy statistics
        self._save_stats()
          
    def view(self, img_only = False):
        import ergo.views as views

        views.model(self, img_only)
        views.roc(self, img_only)
        views.stats(self, img_only)
        views.history(self, img_only)
        views.show(img_only)
