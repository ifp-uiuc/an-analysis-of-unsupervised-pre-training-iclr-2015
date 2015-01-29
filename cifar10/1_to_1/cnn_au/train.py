import os
import sys
sys.path.append('../..')

import numpy

from anna import util
from anna.datasets import supervised_dataset

import checkpoints
from models import CNNModel

print('Start')

pid = os.getpid()
print('PID: {}'.format(pid))
f = open('pid', 'wb')
f.write(str(pid)+'\n')
f.close()

model = CNNModel('experiment', './', learning_rate=1e-2)
checkpoint = checkpoints.unsupervised_layer3
util.set_parameters_from_unsupervised_model(model, checkpoint)
monitor = util.Monitor(model)

# Loading CIFAR-10 dataset
print('Loading Data')
train_data = numpy.load('/data/cifar10/train_X.npy')
train_labels = numpy.load('/data/cifar10/train_y.npy')
test_data = numpy.load('/data/cifar10/test_X.npy')
test_labels = numpy.load('/data/cifar10/test_y.npy')

train_dataset = supervised_dataset.SupervisedDataset(train_data, train_labels)
test_dataset = supervised_dataset.SupervisedDataset(test_data, test_labels)
train_iterator = train_dataset.iterator(
    mode='random_uniform', batch_size=128, num_batches=100000)
test_iterator = test_dataset.iterator(mode='sequential', batch_size=128)

normer = util.Normer2(filter_size=5, num_channels=3)
augmenter = util.DataAugmenter(2, (32, 32), flip=False)

print('Training Model')
for x_batch, y_batch in train_iterator:
    x_batch = x_batch.transpose(1, 2, 3, 0)
    x_batch = augmenter.run(x_batch)
    x_batch = normer.run(x_batch)
    #y_batch = numpy.int64(numpy.argmax(y_batch, axis=1))
    monitor.start()
    log_prob, accuracy = model.train(x_batch, y_batch)
    monitor.stop(1-accuracy)  # monitor takes error instead of accuracy

    if monitor.test:
        monitor.start()
        x_test_batch, y_test_batch = test_iterator.next()
        x_test_batch = x_test_batch.transpose(1, 2, 3, 0)
        x_test_batch = normer.run(x_test_batch)
        #y_test_batch = numpy.int64(numpy.argmax(y_test_batch, axis=1))
        test_accuracy = model.eval(x_test_batch, y_test_batch)
        monitor.stop_test(1-test_accuracy)
