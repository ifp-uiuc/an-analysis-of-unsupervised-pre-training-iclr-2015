# Running the STL-10 experiments

# Contents
+ [Introduction](#introduction)
  + [Folder contents](#folder-contents)
+ [Running experiments](#running-experiments)
  + [Unsupervised training](#unsupervised-training)
  + [Supervised training](#supervised-training)

# Introduction

This folder contains the code used to obtain our results on the [STL-10][STL-10] dataset. This involves training a convolutional autoencoder (cae), and then training convolutional neural network (cnn) with four types of regularization which are denoted as follows:

+ a = Data Augmentation
+ d = Dropout
+ c = Color Augmentation
+ u = Unsupervised Pre-training (with cae)

We will first describe the contents of this folder, and then walk you through
how to run the experiments.

## Folder contents
The folder contains:
``` shell
/cae/layer1/train.py
/cae/layer2/train.py
/cae/layer3/train.py
/cnn_ad/train.py
/cnn_adc/train.py
/cnn_adcu/train.py
/cnn_adu/train.py
checkpoint_checker.py
checkpoints.py
models.py
```

#### `train.py`
As you can see, there are several `train.py` files. Each one trains either a cae model (with 1, 2, or 3 layers), or a cnn model with various regulization methods turned on. Basically the `train.py` files do all the heavy lifting of running the individual experiments. They output a directory of model checkpoint files, and a log of the training process.

#### `checkpoint_checker.py`
The file `checkpoint_checker.py` is a script used to examine all the checkpoints created by a single experiment, and choose the best one.

#### `checkpoints.py`
Some of the experiments involve loading in pre-trained model checkpoints. The paths to those checkpoints will be inserted in `checkpoints.py`. The `train.py`
files will use these paths to load the checkpoints, and take care of the rest.

#### `models.py`
There are 4 neural network models used in these experiments. They include a 1, 2, and 3 layer cae, and a cnn with 3 convolutional layers. They code to construct these models is in `models.py`. They are used by the `train.py` files.

Next we will walk you through running the experiments.

# Running experiments
The experiments involve 1) training an unsupervised model, a stacked convolutional auto encoder as described in the paper, and 2) training supervised convolutional neural networks, with various regularization techniques.

We will first describe how to train the unsupervised model, and then how to run the supervised experiments.

## Unsupervised training

We train the cae in a greedy fashion. First we train a cae with 1 convolutional layer (and 1 deconvolutional layer), which we call the 1 layer model. We then take the weights from that 1 layer model to initialize the layer 1 weights for a 2 layer model. The layer 1 weights are fixed, and then we train layer 2. Similarly, we then take the weights from that 2 layer model to initialize the layer 1 and layer 2 weights for a 3 layer model. The layer 1 and layer 2 weights are fixed, and then we train layer 3.

We describe how to run those steps below.

### How to train layer 1

In order to train the 1 layer cae, first navigate to `./cae/unsupervised_layer1/`, and in the terminal, run the `train.py` script by typing:

``` shell
# Snippet: unsupervised training
$ THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True' \
python -u train.py \
> log.txt & 
```

Please take note of the command above as it will be reused to train the 
second layer as well as the third layer. 

The `log.txt` file will output the mean squared error (MSE) of a minibatch 
after every 10 iterations. The code will also generate a folder called 
`checkpoints` where it will save a .pkl file containing the weights of the 
first convolutional layer. 

Once the code has run to completion, open the `checkpoints.py` file in the 
top-level directory and set `unsupervised_layer1` equal to the path of the 
desired checkpoint in `./cae/unsupervised_layer1/checkpoints` folder. 


### How to train layers 2 and 3

When training the second and third layer, the process is almost identical
to training the first. The process can be broken down into three repeatable 
steps. 

For a given layer L:  

1.  Navigate to the `unsupervised_layer_L` directory in the `cae` directory  
2.  Run the terminal command above to train layer L  
3.  Set the `unsupervised_layer_L` variable in `checkpoints.py` to the 
	appropriate `.pkl` file in `./cae/unsupervised_layer_L/checkpoints/`  


### How to select a checkpoint

One factor that must be considered when doing unsupervised pre-training, is 
when to stop training a given layer and start training the next. We recommend 
finding the point where the mean squared error (MSE) changes only by small 
amounts (i.e. flattens out). This is colloquially known as finding the "elbow" 
of the curve. Our library, anna, provides a means of visualizing the MSE over 
time in a script called `log_plotter.py`, found in the scripts directory. 

Suppose we are training the first layer of our cae. The MSE values in the 
`log.txt` file can be visualized using the following command:

``` shell
# Snippet: plot mse
$ python /path/to/anna/anna/script/log_plotter.py log.txt
```
This will yield something like this:

![elbow_curve](./docs/elbow_curve_stl10.png)

In the case of this layer1 MSE plot, we would suggest choosing a checkpoint 
after tick 60, which corresponds to 60*50=3,000 steps. To find the corresponding 
checkpoint, simply open the `log.txt` file to see:

```
*29950, train error: 40.18884, time: 2.11
29950, train error: 39.89196, time: 2.11
29960, train error: 41.85508, time: 2.11
29970, train error: 38.16329, time: 2.11
29980, train error: 38.91000, time: 2.11
29990, train error: 37.54842, time: 2.11
./
Saving model checkpoint to: ./checkpoints/experiment-11m-27d-08h-26m-01s.pkl
*30000, train error: 39.29796, time: 2.11
30000, train error: 40.01302, time: 2.11
30010, train error: 38.51210, time: 2.11
30020, train error: 47.76662, time: 2.11
30030, train error: 42.10740, time: 2.11
30040, train error: 38.92048, time: 2.11
*30050, train error: 41.59723, time: 2.11
30050, train error: 40.67958, time: 2.11
```
From this, we know then set `unsupervised_layer2` in `checkpoints.py` to:

`/path/to/repo/stl10/cae/unsupervised_layer1/checkpoints/experiment-11m-27d-08h-26m-01s.pkl`.

## Supervised Training

Now that you have successfully trained the convolutional autoencoder, you are
ready to train the four supervised cnns. The four folders starting with `cnn_` 
each contain a `train.py` file which will train the cnn subject to the 
regularizations described in the folder's suffix. 

### How to train a cnn with data augmentation and dropout

For example, `cnn_ad` will train a cnn from a random initialization with data augmentation and dropout, according to the legend given above. 


You can train the cnn with following command: 
``` shell
# Snippet: supervised training
$ THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True' \ 
python -u train.py --split 0  \ 
> log0.txt & 
```

Since the [STL-10][STL-10] dataset asks that people train their models on 10 
pre-specified splits and average the results, the `--split` option indicates 
which of the 10 splits to use (0-9) when training. The code will save the `.pkl` 
file containing the network parameters to a directory called `./checkpoints_0/` 
which will denote the split used.


### How to evaluate a cnn with data augmentation and dropout

After you have trained a split to completion, you can find the best performing
checkpoint by running the checkpoint evaluator found in 
`checkpoint_checker.py`. We will use the model trained in `cnn_ad` as an 
example. Simply run the following command:

``` shell
# Snippet: supervised evaluation
$ THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True' \ 
python -u checkpoint_checker.py --split 0 ./cnn_ad/checkpoints_0/ \
> cnn_ad_best_performance_split_0.txt &
```

With this command, `checkpoint_checker.py` will iterate over the list of
checkpoints found in `./cnn_ad/checkpoints_0/` and compute the accuracy on 
the test set. It will then select the checkpoint that yielded the highest
accuracy. The command also writes all of the results to a text file called 
`cnn_ad_best_performance_split_0.txt`. 

### How to train the rest of the cnns

Now, if you want to train a network with specific regularizations active, 
the process is very simple. 

1. Using the legend above, create a suffix string (S) that corresponds to the 
   regularizations you wish to impose. 
2. Go to the `./cnn_S/` folder.
3. Run the `train.py` file as shown in the Superivsed Training section.


[STL-10]:http://cs.stanford.edu/~acoates/stl10/
