import argparse
import os
import cPickle as pickle

import numpy


def download_cifar10(output_path):
    os.system('wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    os.system('mv cifar-10-python.tar.gz %s' % output_path)
    cur_dir = os.getcwd()
    os.chdir(output_path)
    os.system('tar -xzvf cifar-10-python.tar.gz')
    os.chdir(cur_dir)


def unpickle(file_path):
    # This function unpickles a file and returns a dictionary
    # Credit: Alex Krizhevsky
    # http://www.cs.toronto.edu/~kriz/cifar.html
    fo = open(file_path, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict


def load_batch(path, filename, batch_shape):
    print 'Loading: ', filename
    batch_dict = unpickle(os.path.join(path, filename))
    batch_data = batch_dict['data']
    batch_data = numpy.reshape(batch_data, batch_shape)
    batch_labels = numpy.asarray(batch_dict['labels']).astype('uint8')
    return batch_data, batch_labels


def load_dataset(path, center=False, rescale=True):
    path = os.path.join(path, 'cifar-10-batches-py')
    num_channels, width, height = (3, 32, 32)
    num_train_samples = 50000
    num_test_samples = 10000
    num_batch_samples = 10000
    batch_shape = (num_batch_samples, num_channels, width, height)

    ## Load CIFAR-10 train data
    train_X = numpy.zeros((num_train_samples, num_channels,
                           width, height), dtype='uint8')
    train_y = numpy.zeros((num_train_samples), dtype='uint8')
    test_X = numpy.zeros((num_test_samples, num_channels,
                          width, height), dtype='uint8')
    test_y = numpy.zeros((num_test_samples), dtype='uint8')

    for i in range(0, 5):
        filename = 'data_batch_%d' % (i+1)
        batch_data, batch_labels = load_batch(path, filename, batch_shape)
        batch_slice = slice(i*num_batch_samples, (i+1)*num_batch_samples)
        train_X[batch_slice, :, :, :] = batch_data
        train_y[batch_slice] = batch_labels

    ## Load CIFAR-10 test data
    filename = 'test_batch'
    test_X, test_y = load_batch(path, filename, batch_shape)

    # Check preprocessing options
    train_X = numpy.cast['float32'](train_X)
    test_X = numpy.cast['float32'](test_X)

    if center:
        train_X -= 127.5
        test_X -= 127.5

    if rescale:
        train_X /= 127.5
        test_X /= 127.5

    return train_X, train_y, test_X, test_y


def convert_dataset(path):
    #path = os.path.join(path, 'cifar-10-batches-py')
    train_X, train_y, test_X, test_y = load_dataset(path)

    print 'Saving data to .npy files.'
    numpy.save(os.path.join(path, 'train_X.npy'), train_X)
    numpy.save(os.path.join(path, 'train_y.npy'), train_y)
    numpy.save(os.path.join(path, 'test_X.npy'), test_X)
    numpy.save(os.path.join(path, 'test_y.npy'), test_y)


def reduce_dataset(data_path, N):
    print 'Constructing reduced CIFAR-10 dataset'
    print '%d samples total - (%d samples/class)' % (N*10, N)

    # Load cifar-10 training data
    X = numpy.load(os.path.join(data_path, 'train_X.npy'))
    y = numpy.load(os.path.join(data_path, 'train_y.npy'))

    data_path = os.path.join(data_path, 'reduced')
    data_path = os.path.join(data_path, 'cifar10_'+str(N))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

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
        y_all = numpy.uint8(numpy.concatenate(y_all, axis=0)).ravel()

        # Saving data out to .npy files.
        reduced_dataset_filename = os.path.join(data_path,
                                                'train_X_split_'+str(seed))
        reduced_labels_filename = os.path.join(data_path,
                                               'train_y_split_'+str(seed))
        numpy.save(reduced_dataset_filename, X_all)
        numpy.save(reduced_labels_filename, y_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='make_cifar10_dataset',
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter,
                                     description='Script to download and \
                                                  reduce the CIFAR10 dataset.')
    parser.add_argument('-d', '--download', action='store_true',
                        help='Flag specifying whether to download the dataset \
                              from the web or not.')
    parser.add_argument('-c', '--convert', action='store_true',
                        help='Flag specifying whether to load the dataset, \
                              convert it to .npy files and save it .')
    parser.add_argument('-r', '--reduce', action='store_true',
                        help='Flag specifying whether to reduce the size of \
                              training set to create sets of size 1000, 5000, \
                              and 10000.')
    parser.add_argument('-p', '--data_path', dest='data_path',
                        default='./CIFAR10_HERE',
                        help='Path to download, convert and save all of the \
                        dataset files.')
    #parser.add_argument('-v', action='store_true', help='Verbose')
    args = parser.parse_args()

    print('\n================================================================')
    print('                      CIFAR-10 Dataset Manager                    ')
    print('================================================================\n')

    print args
    download_flag = args.download
    convert_flag = args.convert
    reduce_flag = args.reduce
    data_path = args.data_path

    print 'Download flag: ', download_flag
    print 'Convert flag: ', convert_flag
    print 'Reduce flag: ', reduce_flag
    print 'Data path: ', data_path

    data_path = os.path.join(data_path, 'cifar10')

    # get cifar10 data from the web
    if download_flag:
        print '\nDownloading CIFAR-10 data from the web.'
        download_cifar10(data_path)

    # load/convert the dataset
    if convert_flag:
        print '\nLoading the dataset.'
        dataset = convert_dataset(data_path)

    if reduce_flag:
        # construct reduced versions of the training set
        reduce_dataset(data_path, 100)
        reduce_dataset(data_path, 500)
        reduce_dataset(data_path, 1000)
