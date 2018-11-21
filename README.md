<p align="center">
  <img alt="ergo view" src="https://raw.githubusercontent.com/evilsocket/ergo/master/docs/banner.jpg"/>
</p>

`ergo` (from the Latin sentence *["Cogito ergo sum"](https://en.wikipedia.org/wiki/Cogito,_ergo_sum)*) is a tool that makes machine learning and deep learning with [Keras](https://keras.io/) easier. 

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

Start by printing the available actions by running `ergo help`, you can also print the software version (ergo, keras 
and tensorflow versions) and some hardware info with `ergo info` to verify your installation. 

Once ready, create a new project named `example`:

    ergo new example

Inside the newly created `example` folder, there are two files: `model.py`, that you can change to customize the model 
creation logic:

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

and `train.py`, to customize the training algorithm:

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

After defining the model structure and the training process, you can import a CSV dataset (first column must be the label) and start training using 2 GPUs:

    ergo train example --dataset /some/path/data.csv --gpus 2

This will split the dataset into a train, validation and test sets (partitioned with the `--test` and `--validation` arguments) and start the training.

If you want to update a model and/or train it on already imported data, you can simply:

    ergo train example --gpus 2

Now it's time to visualize the model structure and how the the `accuracy` and `loss` metrics changed during training (requires `sudo apt-get install graphviz python3-tk`):
    
    ergo view example

<p align="center">
  <img alt="ergo view" src="https://raw.githubusercontent.com/evilsocket/ergo/master/docs/view.png"/>
</p>

Once you're done, you can remove the train, test and validation temporary datasets with:

    ergo clean example

To load the model and start a REST API for evaluation (can be customized with `--host`, `--port` and `--debug` options): 

    ergo serve example

You'll be able to access the model for evaluation via `http://127.0.0.1:8080/?x=0.345,1.0,0.9,...`.

To reset the state of a project (**WARNING**: this will remove the datasets, the model files and all training statistics):

    ergo clean example --all

##### Other commands

Optimize a dataset (get unique rows and reuse 15% of the total samples, customize ratio with the `--reuse-ratio` argument, customize output with `--output`):

    ergo optimize-dataset /some/path/data.csv

Convert the Keras model to [frugally-deep](https://github.com/Dobiasd/frugally-deep) format:

    ergo to-fdeep example

#### License

`ergo` was made with ♥  by [Simone Margaritelli](https://www.evilsocket.net/) and it is released under the GPL 3 license.

