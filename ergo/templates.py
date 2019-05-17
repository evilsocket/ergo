prepare = \
"""
import pandas as pd

# this function is called whenever the `ergo train <project> --dataset file.csv`
# command is executed, the first argument is the dataset and it must return
# a pandas.DataFrame object.
def prepare_dataset(filename):
    # simply read as csv
    return pd.read_csv(filename, sep = ',', header = None)

# called during `ergo serve <project>` for each `x` input parameter, use this 
# function to convert, for instance, a file name in a vector of scalars for 
# your model input layer.
def prepare_input(x):
    # simply read as csv
    return pd.read_csv( pd.compat.StringIO(x), sep = ',', header = None)
"""

model = \
"""
import logging as log

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

# build the model
def build_model(is_train):  
    n_inputs       = 10
    n_hidden       = (30, 30,)
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
    model.add(Dense(2, activation = out_activation))
    
    return model
"""

train = \
"""
import logging as log

from keras.callbacks import EarlyStopping

# define training strategy
def train_model(model, dataset):
    log.info("training model (train on %d samples, validate on %d) ..." % ( \\
            len(dataset.Y_train), 
            len(dataset.Y_val) ) )
    
    loss      = 'binary_crossentropy'
    optimizer = 'adam'
    metrics   = ['accuracy']
    
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    earlyStop = EarlyStopping(monitor = 'val_acc', min_delta=0.0001, patience = 5, mode = 'auto')
    return model.fit( dataset.X_train, dataset.Y_train,
            batch_size = 64,
            epochs = 50,
            verbose = 2,
            validation_data = (dataset.X_val, dataset.Y_val),
            callbacks = [earlyStop])
"""

deps = \
"""
ergo
"""

gitignore = \
"""
*.pyc
*.csv
__pycache__
"""

Templates = { 
    'prepare.py' : prepare,
    'model.py'   : model,
    'train.py'   : train,
    'requirements.txt': deps,
    '.gitignore' : gitignore
}
