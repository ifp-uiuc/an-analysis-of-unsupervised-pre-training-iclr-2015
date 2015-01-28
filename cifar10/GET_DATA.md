# Getting the CIFAR-10 data

Here we will describe how to acquire the [CIFAR-10][CIFAR-10] dataset from 
the web and process it such that you can run our experiments. 

Before you start, we suggest that you save the data to its own directory on
your hard drive, say `my_data_dir`. Then, make it easily accessible by creating
a symbolic link (`/data`):

``` shell
$ ln -s {/path/to/my_data_dir} {/data}
```

We have provided a script called `make_cifar10_dataset.py`in the `data_scripts` 
folder which will do the data acquistion and conversion for you. 

## Download the data

The following shell command will download the dataset from the web. It will
also unpack the .tar.gz file and place the files in `/data`.

``` shell
$ python make_cifar10_dataset.py -d -p /data
```

Now if you list the contents of the `/data`, you will see a folder called 
`cifar10` which will contain the following .mat files:

```
cifar-10-batches-py
cifar-10-python.tar.gz
```

## Convert the batch files to the .npy files

Next, you will need to convert the downloaded batch files found in the 
`cifar-10-batches-py` folder to .npy files with the following command:

``` shell
$ python make_cifar10_dataset.py -c -p /data
```

Afterward, the `/data/cifar10` directory should have these .npy files:
```
train_X.npy
train_y.npy
test_X.npy
test_y.npy
```

## Make the reduced training sets

Lastly, you will need to construct the reduced CIFAR-10 datasets. More 
specifically, these datasets will contain just a subset of the labeled 
training samples. The three training sizes we consider are:

+  1,000 samples total - (100 samples/class)
+  5,000 samples total - (500 samples/class)
+  10,000 samples total - (1,000 samples/class)


These sets are created by running the following in the terminal:

``` shell
$ python make_cifar10_dataset.py -r -p /data
```

This command will create a new directory `/data/cifar10/reduced` which will
contain the following files:
```
cifar10_100/train_X_split_0.npy
cifar10_100/train_y_split_0.npy
cifar10_500/train_X_split_0.npy
cifar10_500/train_y_split_0.npy
cifar10_1000/train_X_split_0.npy
cifar10_1000/train_y_split_0.npy
```

[CIFAR-10]:http://www.cs.toronto.edu/~kriz/cifar.html