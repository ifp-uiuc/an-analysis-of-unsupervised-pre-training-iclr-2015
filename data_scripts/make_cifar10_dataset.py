import argparse
import os
import cPickle as pickle

import numpy

from pylearn2.datasets import cifar10, dense_design_matrix


def download_cifar10(output_path):
    os.system('wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    os.system('mv cifar-10-python.tar.gz %s' % output_path)
    cur_dir = os.getcwd()
    os.chdir(output_path)
    os.system('tar -xzvf cifar-10-python.tar.gz')
    os.chdir(cur_dir)


def load_dataset():
    dataset = cifar10.CIFAR10(which_set='train',
                              center=False,
                              rescale=True,
                              axes=['c', 0, 1, 'b'])
    return dataset


def reduce_dataset(dataset, N, output_path):

    print 'Making dataset with %d samples per class (%d samples total)' \
          % (N, N * 10)
    X = dataset.X
    y = dataset.y
    view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 3),
                                                              ['c', 0, 1, 'b'])

    output_path += '/cifar10_'+str(N)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    num_classes = len(numpy.unique(y))
    rng_seeds = [0]
    for i, seed in enumerate(rng_seeds):

        print 'Seed = %d' % seed
        numpy.random.seed(seed)
        X_all = []
        y_all = []

        for c in range(num_classes):
            print 'Class = %d' % c
            # Only take sample with class c
            mask = (y == c).ravel()
            X_temp = X[mask, :]

            # Randomly select N samples from class c
            indices = numpy.random.permutation(X_temp.shape[0])
            indices = indices[0:N]
            X_temp = X_temp[indices, :]
            y_temp = c * numpy.ones((N, 1))
            X_all.append(X_temp)
            y_all.append(y_temp)

        X_all = numpy.concatenate(X_all, axis=0)
        y_all = numpy.int32(numpy.concatenate(y_all, axis=0))
        print X_all.shape
        print y_all.shape
        print y_all.dtype

        reduced_dataset = dense_design_matrix.DenseDesignMatrix(
            X=X_all,
            y=y_all,
            view_converter=view_converter,
            y_labels=num_classes)
        print reduced_dataset.X.shape
        print reduced_dataset.y.shape
        print '\n'

        reduced_dataset_filename = os.path.join(output_path,
                                                'split_'+str(seed)+'.pkl')
        f = open(reduced_dataset_filename, 'wb')
        pickle.dump(reduced_dataset, f)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='make_cifar10_dataset',
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter,
                                     description='Script to download and \
                                                  reduce the CIFAR10 dataset.')
    parser.add_argument('-d', '--download', action='store_true',
                        help='Flag specifying whether to download the dataset \
                              from the web or not.')
    parser.add_argument('-r', '--reduce', action='store_true',
                        help='Flag specifying whether to reduce the size of \
                              training set to create sets of size 1000, 5000, \
                              and 10000.')
    parser.add_argument('-o', '--output_path', dest='output_path',
                        default='./CIFAR10_HERE',
                        help='Path to save all of the output files.')
    #parser.add_argument('-v', action='store_true', help='Verbose')
    args = parser.parse_args()

    print('\n================================================================')
    print('                      CIFAR-10 Dataset Manager                    ')
    print('================================================================\n')

    print args
    download_flag = args.download
    reduce_flag = args.reduce
    output_path = args.output_path

    print 'Download flag: ', download_flag
    print 'Reduce flag: ', reduce_flag
    print 'Output path: ', output_path

    # wget cifar10 data from the web
    #destination_path = './temp'

    if download_flag:
        print '\nDownloading CIFAR-10 data from the web.'
        download_cifar10(output_path)

    # load the dataset
    print '\nLoading the dataset.'
    dataset = load_dataset()

    if reduce_flag:
        # construct reduced versions of the training set
        print '\nMaking reduced dataset.'
        reduce_dataset(dataset, 100, output_path)
        reduce_dataset(dataset, 500, output_path)
        reduce_dataset(dataset, 1000, output_path)
