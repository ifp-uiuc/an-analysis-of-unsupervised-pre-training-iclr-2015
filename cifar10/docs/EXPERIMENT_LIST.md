# CIFAR-10 Experiment List

In this directory we have all of the experiments we conducted on CIFAR-10 to 
analyze the effects of different types of regularization (droupout, 
data augmentation, and unsupervised pre-training). We also provide experiments
to analyze the effects of regularization as the ratio of unlabeled samples
to labeled samples changes. Specifically, we consider the following ratios 
(unlabeled to labeled):

+ 50:1 - 50,000:1,000 - (100 labeled samples per class)
+ 10:1 - 50,000:5,000 - (500 labeled samples per class)
+ 5:1  - 50,000:10,000 - (1,000 labeled samples per class)
+ 1:1  - 50,000:50,000 - (5,000 labeled samples per class)

## Types of Regularization

Each ratio is given its own directory and within each of them are 8 folders
describing the 8 training scenarios we considered. The suffix of each folder 
denotes the types of regularizations we imposed during training. We provide 
a key below:

+ a - Data Augmentation
+ d - Dropout
+ u - Unsupervised Pre-training (with cae)

Thus, for a given ratio, the experiments are as follows:

1. cnn - CNN trained from random initialization
2. cnn_a - CNN + data augmentation
3. cnn_d - CNN + dropout
4. cnn_u - CNN + unsupervised initialization
5. cnn_ad - CNN + data augmentation + dropout
6. cnn_au - CNN + data augmentation + unsupervised initialization
7. cnn_du - CNN + dropout + unsupervised initialization
8. cnn_adu - CNN + data augmentation + dropout + unsupervised initialization

We recommend training the convolutional autoencoder (cae) first before 
proceeding to supervised experiments. They are listed below:

1. [ ] cae - layer 1
2. [ ] cae - layer 2
3. [ ] cae - layer 3


## Complete Checklist

For your reference, we provide a checklist of all 32 experiments in the cifar10 directory below (sorted by ratio):

a. 50_to_1

1. [ ] cnn 
2. [ ] cnn_a 
3. [ ] cnn_d 
4. [ ] cnn_u 
5. [ ] cnn_ad 
6. [ ] cnn_au 
7. [ ] cnn_du 
8. [ ] cnn_adu 

b. 10_to_1

1. [ ] cnn 
2. [ ] cnn_a 
3. [ ] cnn_d 
4. [ ] cnn_u 
5. [ ] cnn_ad 
6. [ ] cnn_au 
7. [ ] cnn_du 
8. [ ] cnn_adu 

c. 5_to_1

1. [ ] cnn 
2. [ ] cnn_a 
3. [ ] cnn_d 
4. [ ] cnn_u 
5. [ ] cnn_ad 
6. [ ] cnn_au 
7. [ ] cnn_du 
8. [ ] cnn_adu 

d. 1_to_1

1. [ ] cnn 
2. [ ] cnn_a 
3. [ ] cnn_d 
4. [ ] cnn_u 
5. [ ] cnn_ad 
6. [ ] cnn_au 
7. [ ] cnn_du 
8. [ ] cnn_adu 
