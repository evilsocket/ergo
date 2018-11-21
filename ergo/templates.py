model = \
"""
import logging as log

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

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
            model.add(Dense(600, input_shape = (n_inputs,), activation = activation))
        else:
            model.add(Dense(600, activation = activation))
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

def train_model(model, dataset):
    log.info("training model is-malware (train on %d samples, validate on %d) ..." % ( \\
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

Templates = { 
    'model.py': model,
    'train.py': train
}
