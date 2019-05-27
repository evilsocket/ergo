<p align="center">
  <img alt="ergo" src="https://i.imgur.com/EO9PdNp.jpg"/>
  <p align="center">
    <a href="https://github.com/evilsocket/ergo/releases/latest"><img alt="Release" src="https://img.shields.io/github/release/evilsocket/ergo.svg?style=flat-square"></a>
    <a href="https://github.com/evilsocket/ergo/blob/master/LICENSE.md"><img alt="Software License" src="https://img.shields.io/badge/license-GPL3-brightgreen.svg?style=flat-square"></a>
  </p>
</p>

`ergo` (from the Latin sentence *["Cogito ergo sum"](https://en.wikipedia.org/wiki/Cogito,_ergo_sum)*) is a command line tool that makes machine learning with [Keras](https://keras.io/) easier. 

**It can be used to**: 

* scaffold new projects in seconds and customize only a minimum amount of code.
* encode samples, import and optimize CSV datasets and train the model with them.
* visualize the model structure, loss and accuracy functions during training.
* determine how each of the input features affects the accuracy by differential inference.
* export a simple REST API to use your models from a server.

### Installing

    sudo pip3 install ergo-ai

### Installing from Sources

    git clone https://github.com/evilsocket/ergo.git
    cd ergo
    sudo pip3 install -r requirements.txt
    python3 setup.py build
    sudo python3 setup.py install

### Enable GPU support (optional)

Make sure you have [CUDA 9.0 and cuDNN 7.0 installed](https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e) and then:

    sudo pip3 uninstall tensorflow
    sudo pip3 install tensorflow-gpu

### Example Projects

- A [planes detector](https://github.com/evilsocket/ergo-planes-detector) from satellite imagery.
- An [anti malware API](https://github.com/evilsocket/ergo-pe-av) for Windows.

### Usage

To print the general help menu:

    ergo help

To print action specific help:

    ergo <action> -h

Start by printing the available actions by running `ergo help`, you can also print the software version (ergo, keras and tensorflow versions) and some hardware info with `ergo info` to verify your installation. 

#### Creating a Project

Once ready, create a new project named `example` (`ergo create -h` to see how to customize the initial model):

    ergo create example

Inside the newly created `example` folder, there will be three files: 

1. `prepare.py`, used to preprocess your dataset and inputs (if, for instance, you're using pictures instead of a csv file).
2. `model.py`, that you can change to customize the model.
3. `train.py`, for the training algorithm.

By default, ergo will simply read the dataset as a CSV file, build a small neural network with 10 inputs, two hidden layers of 30 neurons each and 2 outputs and use a pretty standard training algorithm.

#### Exploration (optional) 

Explore properties of the dataset. Ergo can generate graphs and tables that can be useful for the feature engineering of the problem. 

Example with a dataset `some/path/data.csv`: 
    
    ergo explore example --dataset some/path/data.csv 

This will generate a table with the correlation of each feature with the target feature and the [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) decomposition of the dataset, saving (and optionally showing) the explained variance vs the number of principal component vectors used and the 2D projection of the dataset (colored by labels). 

#### Encoding (optional)

In case you implemented the `prepare_input` function in the `prepare.py` script, ergo can be used to encode raw samples, being them executables, images, strings or whatever, into vectors of scalars that are then saved into a `dataset.csv` file suitable for training

Example with a folder `/path/to/data` which contains a `pos` and `neg` subfolders, in auto labeling mode each group of sample is labeled with its parent directory name:

    ergo encode example /path/to/data

Example with a single folder and manual labeling:

    ergo encode example /path/to/data --label 'some-label'

Example with a single text file containing multiple inputs, one per line:

    ergo encode example /path/to/data --label 'some-label' -m

#### Training

After defining the model structure and the training process, you can import a CSV dataset (first column must be the label) and start training using 2 GPUs:

    ergo train example --dataset /some/path/data.csv --gpus 2

This will split the dataset into a train, validation and test sets (partitioned with the `--test` and `--validation` arguments), start the training and once finished show the model statistics.

If you want to update a model and/or train it on already imported data, you can simply:

    ergo train example --gpus 2

#### Testing

Now it's time to visualize the model structure and how the the `accuracy` and `loss` metrics changed during training (requires `sudo apt-get install graphviz python3-tk`):
    
    ergo view example

If the `data-test.csv` file is still present in the project folder (`ergo clean` has not been called yet), `ergo view` will also show the ROC curve.

You can use the `relevance` command to evaluate the model on a given set (or a subset of it, see `--ratio 0.1`) by nulling one attribute at a time and measuring how that influenced the accuracy (`feature.names` is an optional file with the names of the attributes, one per line):

    ergo relevance example --dataset /some/path/data.csv --attributes /some/path/feature.names --ratio 0.1

Once you're done, you can remove the train, test and validation temporary datasets with:

    ergo clean example

#### Inference

To load the model and start a REST API for evaluation (can be customized with `--address`, `--port`, `--classes` and `--debug` options): 

    ergo serve example

To run an inference on a vector of scalars:

    curl "http://localhost:8080/?x=0.345,1.0,0.9,..."

If you customized the `prepare_input` function in `prepare.py` (see the `Encoding` section), you can run an inference on a raw sample:

    curl "http://localhost:8080/?x=/path/to/sample"

The input `x` can also be passed as a POST request:

    curl --data 'x=...' "http://localhost:8080/"

Or as a file upload:

    curl -F 'x=@/path/to/file' "http://localhost:8080/"

The API can also be used to perform encoding only:

    curl -F 'x=@/path/to/file' "http://localhost:8080/encode"

This will return the raw features vector that can be used for inference later.

#### Other commands

To reset the state of a project (**WARNING**: this will remove the datasets, the model files and all training statistics):

    ergo clean example --all

Evaluate and compare the performances of two trained models on a given dataset and (optionally) output the differences to a json file:

    ergo cmp example_a example_b --dataset /path/to/data.csv --to-json diffs.json

Freeze the graph and convert the model to the [TensorFlow](https://www.tensorflow.org/) protobuf format:

    ergo to-tf example

Convert the Keras model to [frugally-deep](https://github.com/Dobiasd/frugally-deep) format:

    ergo to-fdeep example

Optimize a dataset (get unique rows and reuse 15% of the total samples, customize ratio with the `--reuse-ratio` argument, customize output with `--output`):

    ergo optimize-dataset /some/path/data.csv

### License

`ergo` was made with ♥  by [the dev team](https://github.com/evilsocket/ergo/graphs/contributors) and it is released under the GPL 3 license.

