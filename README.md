# Ergo

ergo (from the Latin sentence *["Cogito ergo sum"](https://en.wikipedia.org/wiki/Cogito,_ergo_sum)*) is a tool that makes machine learning and deep learning with [Keras](https://keras.io/) easier. 

**It can be used to**: 

* scaffold new projects in seconds and customize only a minimum amount of code.
* import and optimize CSV datasets and train the model with them.
* visualize the model structure, loss and accuracy functions during training.
* export a simple REST API to use your models from a server.

**WORK IN PROGRESS, WAIT FOR A STABLE RELEASE**

#### Installation

Requires `python3` and `pip3`, to install:

    sudo pip3 install -r requirements.txt
    python3 setup.py build
    sudo python3 setup.py install

##### Enable GPU support

Make sure you have [CUDA 9.0 and cuDNN 7.0 installed](https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e) and then:

    sudo pip3 uninstall tensorflow
    sudo pip3 install tensorflow-gpu

#### Usage

Print available actions:

    ergo help

Print the software version (ergo, keras and tensorflow versions) and some hardware info:

    ergo info

Create a new project:

    ergo new project-name

Customize the model creation logic in the `project-name/model.py` file:

```python
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
```

Customize the training logic in the `project-name/train.py` file:

```python
def train_model(model, dataset):
    log.info("training model is-malware (train on %d samples, validate on %d) ..." % ( \
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
```

Optimize a dataset (get unique rows and reuse 15% of the total samples, customize ratio with `--reuse-ratio` argument, customize output with `--output`):

    ergo optimize-dataset /some/path/data.csv

Import a dataset (data format is csv, first column is the label) and start training (use `--test` and `--validation` optional arguments to size the datasets) using 2 GPUs:

    ergo train project-name --dataset /some/path/data.csv --gpus 2

Train with a previously imported dataset (and update the model) using all available GPUs (or CPU cores if no GPU support is found):

    ergo train project-name

Remove the train, test and validation temporary datasets:

    ergo clean project-name

Display a model and its training history (requires `sudo apt-get install graphviz python3-tk`):

    ergo view project-name

Load the model and start a REST API for evaluation (can be customized with `--host`, `--port` and `--debug` options, default to `http://127.0.0.1:8080/?x=0.345,1.0,0.9,...`): 

    ergo serve project-name

Convert the Keras model to [fdeep](https://github.com/Dobiasd/frugally-deep) format:

    ergo to-fdeep project-name

Reset the state of a project (**WARNING**: this will remove the datasets, the model files and all training statistics):

    ergo clean project-name --all
