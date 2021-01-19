from ergo.core.template import Template

prepare = \
"""
import pandas as pd

# this function is called whenever the `ergo train <project> --dataset file.csv`
# command is executed, the first argument is the dataset and it must return
# a pandas.DataFrame object.
def prepare_dataset(filename):
    # simply read as csv
    return pd.read_csv(filename, sep = ',', header = None)

# this function is called to process a single input into a vector
# that can be used for training or to run an inference.
# it is called from `ergo encode ...` with `is_encoding` set to True,
# in which case you can add additional metadata to the vector, or
# from the `ergo serve ...` API, in which case the vector will be
# used for inference and can't contain metadata.
def prepare_input(x, is_encoding = False):
    # simply read as csv
    return pd.read_csv( pd.compat.StringIO(x), sep = ',', header = None)
"""

model = \
"""
import logging as log

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

# build the model
def build_model(is_train):
    n_inputs       = {NUM_INPUTS}
    n_hidden       = [{HIDDEN}]
    dropout        = 0.4
    activation     = 'relu'
    out_activation = 'softmax'

    log.info("building model for %s ..." % 'training' if is_train else 'evaluation')

    model = Sequential()
    for i, n_neurons in enumerate(n_hidden):
        # setup the input layer
        if i == 0:
            model.add(Dense(n_neurons, input_shape = (n_inputs,), activation = activation))
        else:
            model.add(Dense(n_neurons, activation = activation))
        # add dropout
        if is_train:
            model.add(Dropout(dropout))
    # setup output layer
    model.add(Dense({NUM_OUTPUTS}, activation = out_activation))

    return model
"""

train = \
"""
import logging as log

from tensorflow.keras.callbacks import EarlyStopping

# define training strategy
def train_model(model, dataset):
    log.info("training model (train on %d samples, validate on %d) ..." % ( \\
            len(dataset.Y_train),
            len(dataset.Y_val) ) )

    loss      = 'categorical_crossentropy'
    optimizer = 'adam'
    metrics   = ['accuracy']

    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    earlyStop = EarlyStopping(monitor = 'val_acc', min_delta=0.0001, patience = 5, mode = 'auto')
    return model.fit( dataset.X_train, dataset.Y_train,
            batch_size = {BATCH_SIZE},
            epochs = {MAX_EPOCHS},
            verbose = 2,
            validation_data = (dataset.X_val, dataset.Y_val),
            callbacks = [earlyStop])
"""

deps = \
"""
ergo-ai
"""

gitignore = \
"""
*.pyc
*.csv
__pycache__
"""

Templates = [
    Template('prepare.py', prepare),
    Template('model.py', model),
    Template('train.py', train),
    Template('requirements.txt', deps),
    Template('.gitignore', gitignore)
]
