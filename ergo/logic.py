import os
import logging as log
from importlib.machinery import SourceFileLoader

class Logic(object):
    @staticmethod
    def get_symbol(name, symbol):
        log.debug("loading symbol %s from %s ..." % (symbol, name))
        doer = SourceFileLoader("",name).load_module()
        if symbol not in doer.__dict__:
            return None, "%s does not define a %s function" % (name, symbol)
        return doer.__dict__[symbol], None

    def __init__(self, path):
        self.path           = os.path.abspath(path)
        self.preparer_path  = os.path.join(self.path, 'prepare.py')
        self.preparer       = None
        self.builder_path   = os.path.join(self.path, 'model.py')
        self.builder        = None
        self.trainer_path   = os.path.join(self.path, 'train.py')
        self.trainer        = None

    def load(self):
        self.preparer, err = Logic.get_symbol(self.preparer_path, 'prepare_dataset')
        if err is not None:
            return err

        self.builder, err = Logic.get_symbol(self.builder_path, 'build_model')
        if err is not None:
            return err

        self.trainer, err = Logic.get_symbol(self.trainer_path, 'train_model')
        if err is not None:
            return err

        return None






