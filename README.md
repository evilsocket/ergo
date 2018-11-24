<p align="center">
  <img alt="ergo" src="https://i.imgur.com/EO9PdNp.jpg"/>
</p>

`ergo` (from the Latin sentence *["Cogito ergo sum"](https://en.wikipedia.org/wiki/Cogito,_ergo_sum)*) is a tool that makes deep learning with [Keras](https://keras.io/) easier. 

**It can be used to**: 

* scaffold new projects in seconds and customize only a minimum amount of code.
* import and optimize CSV datasets and train the model with them.
* visualize the model structure, loss and accuracy functions during training.
* export a simple REST API to use your models from a server.

#### Installation

Download the [latest stable release](https://github.com/evilsocket/ergo/releases), extract and:

    cd /path/to/ergo
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

    ergo create example

Inside the newly created `example` folder, there will be three files: 

1. `prepare.py`, used to preprocess your dataset and inputs (if, for instance, you're using pictures instead of a csv file).
2. `model.py`, that you can change to customize the model.
3. `train.py`, for the training algorithm.

By default, ergo will simply read the dataset as a CSV file, build a small neural network with 10 inputs, two hidden layers of 30 neurons 
each and 2 outputs and use a pretty standard training algorithm. You can see a complete (and more complex) example on the [planes-detector](https://github.com/evilsocket/ergo-planes-detector) 
project repository.

After defining the model structure and the training process, you can import a CSV dataset (first column must be the label) and start training using 2 GPUs:

    ergo train example --dataset /some/path/data.csv --gpus 2

This will split the dataset into a train, validation and test sets (partitioned with the `--test` and `--validation` arguments) and start the training.

If you want to update a model and/or train it on already imported data, you can simply:

    ergo train example --gpus 2

Now it's time to visualize the model structure and how the the `accuracy` and `loss` metrics changed during training (requires `sudo apt-get install graphviz python3-tk`):
    
    ergo view example

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

