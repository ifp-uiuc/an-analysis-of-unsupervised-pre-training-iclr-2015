# STL-10 Experiments

In this folder, we present all of the code used to obtain our results for 
four different networks we trained on the [STL-10][STL-10] dataset. In each of 
the four cases, we trained a three layer convolutional neural network (cnn), 
some of these we pre-train using a convolutional autoencoder (cae). Overall 
this experiment uses four types of regularization which are denoted as follows:

+ a = Data Augmentation
+ d = Dropout
+ c = Color Augmentation
+ u = Unsupervised Pre-training (with cae)

We recommend training the cae first and then proceeding to the supervised 
training experiments.


# Unsupervised Pre-training

We will now disucss how to train the convolutional autoencoder. We train the 
cae in a greedy fashion. Specifically, ... The code to do this can be found in 
the `cae` folder. (DESCRIBE THE TRAINING PROCESS IN MORE DETAIL).

## How to train layer 1

In order to train the first layer of the neural network, all you need to do is 
navigative to `./cae/unsupervised_layer1/` and type the following command
into the terminal:

``` shell
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
(WRITE CODE TO AUTOMATICALLY SELECT A CHECKPOINT).

## How to train layers 2 and 3

When training the second and third layer, the process is almost identical
to training the first. The process can be broken down into three repeatable 
steps. 

For a given layer L:  

1.  Navigate to the `unsupervised_layer_L` directory in the `cae` directory  
2.  Run the terminal command above to train layer L  
3.  Set the `unsupervised_layer_L` variable in `checkpoints.py` to the 
	appropriate `.pkl` file in `./cae/unsupervised_layer_L/checkpoints/`  


# Supervised Training

Now that you have successfully trained the convolutional autoencoder, you are
ready to train the four supervised cnns. The four folders starting `cnn_` 
each contain a `train.py` file which will train the cnn subject to the 
regularizations described in the folder suffix. 

For example, `cnn_ad` will train a cnn from a random intialzation with data augmentation and dropout, according to the legend given above. 


You can train the cnn with following command: 
``` shell
$ THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True' \ 
python -u train.py --split 0  \ 
> log0.txt & 
```

Since the [STL-10][STL-10] datsaet asks that people train their models on 10 
pre-specified splits and average the results, the `--split` option indicates 
which of the 10 splits to use (0-9) when traning. The code will save the `.pkl` 
file containing the network parameters to a directory called `./checkpoints_0/` 
which will denote the split used.


# Evaluation

After you have trained a split to completion, you can find the best performing
checkpoint by running the checkpoint evaluator found in 
`find_best_performance.py`. We will use the model trained in `cnn_ad` as an 
example. Simply run the following command:

``` shell
$ THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True' \ 
python -u find_best_performance.py --split 0 ./cnn_ad/checkpoints_0/ \
> cnn_ad_best_performance_split_0.txt &
```

With this command, `find_best_peformance.py` will iterate over the list of
checkpoints found in `./cnn_ad/checkpoints_0/` and compute the accuracy on 
the test set. It will then select the checkpoint that yielded the highest
accuracy. The command also writes all of the results to a text file called 
`cnn_ad_best_performance_split_0.txt`. 


# Results

If you are able to run through all of these steps successfully, you will
hopefully obtain results similar to ours:

(INSERT STL-10 TABLE (Table 3) FROM PAPER HERE).


[STL-10]:http://cs.stanford.edu/~acoates/stl10/
