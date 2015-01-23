import argparse
import os

import numpy
from scipy.io import loadmat
import h5py


def download_stl10(output_path):
    print 'Download STL-10 data from the web.'
    cur_dir = os.getcwd()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    os.chdir(output_path)
    os.system('wget http://ai.stanford.edu/~acoates/stl10/stl10_matlab.tar.gz')
    os.system('tar -xzvf stl10_matlab.tar.gz')
    os.chdir(cur_dir)


def convert_stl10(output_path):
    print 'Convert STL-10 .mat files to .npy files.'

    print 'Loading the unlabled data.'
    D_u = h5py.File(os.path.join(output_path, 'unlabeled.mat'))
    print D_u.keys()
    X_u = numpy.array(D_u['X'])
    X_u = X_u.T
    num_unlabeled = X_u.shape[0]
    X_u = numpy.reshape(X_u, (num_unlabeled, 3, 96, 96))
    X_u = X_u.transpose(0, 1, 3, 2)
    print 'Saving the labeled data to .npy file.'
    numpy.save(os.path.join(output_path, 'unsupervised.npy'), X_u)

    print 'Loading the labeled training data.'
    D_train = loadmat(os.path.join(output_path, 'train.mat'))
    print D_train.keys()
    train_X = D_train['X']
    num_train = train_X.shape[0]
    train_X = numpy.reshape(train_X, (num_train, 3, 96, 96))
    train_X = train_X.transpose(0, 1, 3, 2)
    train_y = D_train['y'].ravel()
    print 'Saving the labeled training data to .npy files.'
    numpy.save(os.path.join(output_path, 'train_X.npy'), train_X)
    numpy.save(os.path.join(output_path, 'train_y.npy'), train_y)
    numpy.save(os.path.join(output_path, 'train_fold_indices.npy'),
               D_train['fold_indices'])

    print 'Loading the labeled testing data.'
    D_test = loadmat(os.path.join(output_path, 'test.mat'))
    print D_test.keys()
    test_X = D_test['X']
    num_test = test_X.shape[0]
    test_X = numpy.reshape(test_X, (num_test, 3, 96, 96))
    test_X = test_X.transpose(0, 1, 3, 2)
    test_y = D_test['y'].ravel()
    print 'Saving the labeled test data to .npy files.'
    numpy.save(os.path.join(output_path, 'test_X.npy'), test_X)
    numpy.save(os.path.join(output_path, 'test_y.npy'), test_y)


def split_stl10(output_path):
    train_X = numpy.load(os.path.join(output_path, 'train_X.npy'))
    train_y = numpy.load(os.path.join(output_path, 'train_y.npy'))
    fold_ind = numpy.load(os.path.join(output_path, 'train_fold_indices.npy'))
    fold_ind = fold_ind[0]

    print train_X.shape
    print train_y.shape
    print fold_ind.shape
    print '\n'

    print('Breaking up STL10 training set into 10 splits.')
    #num_splits = len(fold_ind)
    for i, split in enumerate(fold_ind):
        print('Split {}: {}'.format(i, split.shape))
        split = split.ravel()-1
        train_X_split = train_X[split, :, :, :]
        train_y_split = train_y[split]
        print train_X_split.shape
        print train_y_split.shape

        train_splits_path = os.path.join(output_path, 'train_splits')
        if not os.path.exists(train_splits_path):
            os.makedirs(train_splits_path)

        train_X_split_filename = os.path.join(output_path, 'train_splits',
                                              'train_X_'+str(i)+'.npy')
        train_y_split_filename = os.path.join(output_path, 'train_splits',
                                              'train_y_'+str(i)+'.npy')

        numpy.save(train_X_split_filename, train_X_split)
        numpy.save(train_y_split_filename, train_y_split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='make_stl10_dataset',
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter,
                                     description='Script to download and \
                                                  split the STL-10 dataset.')
    parser.add_argument('-d', '--download', action='store_true',
                        help='Flag specifying whether to download the dataset \
                              from the web or not.')
    parser.add_argument('-c', '--convert', action='store_true',
                        help='Flag specifying whether to convert the .mat \
                              files to .npy files.')
    parser.add_argument('-s', '--split', action='store_true',
                        help='Flag specifying whether to split the training  \
                             set according to the fold defined in \
                             training_fold_indices.npy.')
    parser.add_argument('-o', '--output_path', dest='output_path',
                        default='./STL10_HERE',
                        help='Path to save all of the output files.')
    #parser.add_argument('-v', action='store_true', help='Verbose')
    args = parser.parse_args()

    print('\n================================================================')
    print('                        STL-10 Dataset Manager                    ')
    print('================================================================\n')

    print args
    download_flag = args.download
    convert_flag = args.convert
    split_flag = args.split
    output_path = args.output_path

    print 'Download flag: ', download_flag
    print 'Convert flag: ', convert_flag
    print 'Split flag: ', split_flag
    print 'Output path: ', output_path

    if download_flag:
        download_stl10(output_path)

    if convert_flag:
        convert_stl10(output_path)

    if split_flag:
        split_stl10(output_path)
