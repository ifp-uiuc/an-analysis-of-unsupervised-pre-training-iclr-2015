import sys
import os
import argparse

import numpy

from model import CNNModel
from anna import util
from anna.datasets import supervised_dataset


def load_cifar10_data():
    # Load CIFAR-10 data
    print 'Loading CIFAR-10 Testing Data'
    X_test = numpy.load('/data/cifar10/test_X.npy')
    y_test = numpy.load('/data/cifar10/test_y.npy')

    test_dataset = supervised_dataset.SupervisedDataset(X_test, y_test)
    test_iterator = test_dataset.iterator(mode='sequential', batch_size=128)

    return test_iterator


def compute_overall_accuracy(model, normer, mode, iterator):
    accuracy_list = []
    # Compute accuracy on each batch
    i = 0
    for x_batch, y_batch in iterator:
        x_batch = x_batch.transpose(1, 2, 3, 0)
        x_batch = normer.run(x_batch)
        batch_accuracy = model.eval(x_batch, y_batch-1)
        accuracy_list.append(batch_accuracy)
        i += 1

    # Compute overall accuracy
    overall_accuracy = numpy.mean(numpy.hstack(accuracy_list))
    print('\nOverall {} Accuracy: {}\n'.format(mode, overall_accuracy))
    return overall_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='cifar10_checkpoint_checker',
        description='Script to select best performing checkpoint on CIFAR-10.')
    parser.add_argument(
        "checkpoint_dir",
        help='Folder containing all .pkl checkpoint files.')
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        raise Exception('Checkpoint directory does not exist.')
    checkpoint_list = sorted(os.listdir(checkpoint_dir))

    model = CNNModel('xxx', './')
    model.fc4.dropout = 0.0
    model._compile()
    num_channels = model.conv1.filter_shape[0]
    filter_size = model.conv1.filter_shape[1]

    # Get iterators for cifar10 test set
    test_iterator = load_cifar10_data()

    # Create object to local contrast normalize a batch.
    # Note: Every batch must be normalized before use.
    normer = util.Normer2(filter_size=filter_size, num_channels=num_channels)

    test_accuracies = []

    for i, checkpoint_file in enumerate(checkpoint_list):
        print 'Loading Checkpoint %s' % checkpoint_file
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        util.load_checkpoint(model, checkpoint_path)

        print 'Compute Test Accuracy'
        test_accuracies.append(compute_overall_accuracy(
            model, normer, 'test', test_iterator))
        print '\n'

        test_iterator.reset()

    max_test_accuracy = numpy.max(test_accuracies)
    max_index = numpy.argmax(test_accuracies)
    max_checkpoint = checkpoint_list[max_index]

    print 'Max Test Accuracy: %.2f' % (max_test_accuracy*100)
    print 'Max Checkpoint: %s' % max_checkpoint
