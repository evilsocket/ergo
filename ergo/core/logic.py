import os
import logging as log
from importlib.machinery import SourceFileLoader

class Logic(object):
    @staticmethod
    def get_doer(name):
        log.debug("loading %s ..." % name)
        return SourceFileLoader("",name).load_module()

    @staticmethod
    def get_symbol(name, symbol):
        doer = Logic.get_doer(name)
        if symbol not in doer.__dict__:
            return None, "%s does not define a %s function" % (name, symbol)
        return doer.__dict__[symbol], None

    @staticmethod
    def get_symbols(name, symbols):
        doer = Logic.get_doer(name)
        ret  = []
        for symbol in symbols:
            if symbol not in doer.__dict__:
                return None, "%s does not define a %s function" % (name, symbol)
            else:
                ret.append(doer.__dict__[symbol])
        return ret, None

    def __init__(self, path):
        self.path             = os.path.abspath(path)
        self.preparer_path    = os.path.join(self.path, 'prepare.py')
        self.dataset_preparer = None
        self.input_preparer   = None
        self.builder_path     = os.path.join(self.path, 'model.py')
        self.builder          = None
        self.trainer_path     = os.path.join(self.path, 'train.py')
        self.trainer          = None

    def load(self):
        preparers, err = Logic.get_symbols(self.preparer_path, ('prepare_dataset', 'prepare_input')) 
        if err is not None:
            return err
        else:
            self.dataset_preparer, self.input_preparer = preparers

        self.builder, err = Logic.get_symbol(self.builder_path, 'build_model')
        if err is not None:
            return err

        self.trainer, err = Logic.get_symbol(self.trainer_path, 'train_model')
        if err is not None:
            return err

        return None

    def prepare_dataset(self, filename):
        return self.dataset_preparer(filename)

    def prepare_input(self, x):
        return self.input_preparer(x)





