# Getting the STL-10 data

Here we will describe how to acquire the [STL-10][STL-10] dataset from the web
and process it such that you can run our experiments. 

Before you start, we suggest that you save the data to its own directory on
your hard drive, say `my_data_dir`. Then make it easily accessible by creating
a symbolic link (`/data`):

``` shell
$ ln -s {/path/to/my_data_dir} {/data}
```

We have provided a script called `make_stl10_dataset.py`in the `data_scripts` 
folder which will do the data acquistion and conversion for you. 

## Download the data

The following shell command will download the dataset from the web. It will
also unpack the .tar.gz file and place the .mat files in `/data`.

``` shell
$ python make_stl10_dataset.py -d -p /data
```

Now if you list the contents of the `/data`, you will see a folder called 
`stl10_matlab` which will contain the following .mat files:

```
unlabeld.mat
train.mat
test.mat
```

## Convert the .mat files to the .npy files

Next, you will need to convert the downloaded .mat files to .npy files with
the following command:

``` shell
$ python make_stl10_dataset.py -c -p /data
```

Afterward, the `/data/stl10_matlab` directory should have these .npy files:
```
unsupervised.npy
train_X.npy
train_y.npy
train_fold_indices.npy
test_X.npy
test_y.npy
```

## Split the data

Since the STL-10 dataset requires that the training set be split into 10 
folds during supervised training, we also provide a means to do this.

``` shell
$ python make_stl10_dataset.py -s -p /data
```

This command will create a new directory `/data/stl10_matlab/train_splits` 
which will contain each split's data and the corresponding labels.

[STL-10]:http://cs.stanford.edu/~acoates/stl10/