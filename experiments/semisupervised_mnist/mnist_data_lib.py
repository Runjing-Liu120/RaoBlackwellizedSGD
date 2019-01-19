import os

import torch

import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.utils.data import Dataset

import numpy as np

def load_mnist_data(data_dir = '../mnist_data/', train = True):
    assert os.path.exists(data_dir)

    trans = lambda x: transforms.ToTensor()(x).bernoulli()

    data = dset.MNIST(root=data_dir, train=train,
                            transform=trans, download=True)

    return data

class MNISTDataSet(Dataset):

    def __init__(self, data_dir = '../mnist_data/',
                    propn_sample = 1.0,
                    indices = None,
                    train_set = True):

        super(MNISTDataSet, self).__init__()

        # Load MNIST dataset
        assert os.path.exists(data_dir)

        # This is the full dataset
        self.mnist_data_set = load_mnist_data(data_dir = data_dir, train = train_set)

        if train_set:
            n_image_full = len(self.mnist_data_set.train_labels)
        else:
            n_image_full = len(self.mnist_data_set.test_labels)

        # we may wish to subset
        if indices is None:
            self.num_images = round(n_image_full * propn_sample)
            self.sample_indx = np.random.choice(n_image_full, self.num_images, replace = False)
        else:
            self.num_images = len(indices)
            self.sample_indx = indices

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return {'image' : self.mnist_data_set[self.sample_indx[idx]][0].squeeze(),
                'label' : self.mnist_data_set[self.sample_indx[idx]][1].squeeze()}

# def get_mnist_dataset(data_dir = '../mnist_data/',
#                     propn_sample = 1.0):
#     train_set = MNISTDataSet(data_dir = data_dir,
#                             propn_sample = propn_sample,
#                             train_set = True)
#
#     test_set = MNISTDataSet(data_dir = data_dir,
#                             propn_sample = propn_sample,
#                             train_set = False)
#
#     return train_set, test_set
#
# def get_mnist_dataset_semisupervised(data_dir = './mnist_data/',
#                     propn_sample = 1.0, propn_labeled = 0.1):
#
#     total_num_train_images = 60000 # is there way to read this in?
#
#     # subsample training set if desired
#     num_train_images = round(propn_sample * total_num_train_images)
#     subs_train_set = np.random.choice(total_num_train_images, \
#                         num_train_images,
#                         replace = False)
#
#     # split training set into labeled and unlabled images
#     num_labeled_images = round(num_train_images * propn_labeled)
#     train_set_labeled = MNISTDataSet(data_dir = data_dir,
#                             indices = subs_train_set[:num_labeled_images],
#                             train_set = True)
#     train_set_unlabeled = MNISTDataSet(data_dir = data_dir,
#                             indices = subs_train_set[num_labeled_images:],
#                             train_set = True)
#
#     # get test set as usual
#     test_set = MNISTDataSet(data_dir = data_dir,
#                             propn_sample = propn_sample,
#                             train_set = False)
#
#     return train_set_labeled, train_set_unlabeled, test_set

def get_mnist_dataset_semisupervised(data_dir = './mnist_data/',
                                    train_test_split_folder = './train_test_splits/'
                                        eval_test_set = False):

    labeled_indx = np.load(train_test_split_folder + 'labeled_train_indx.npy')
    train_set_labeled = MNISTDataSet(data_dir = data_dir,
                            indices = labeled_indx,
                            train_set = True)

    unlabeled_indx = np.load(train_test_split_folder + 'unlabeled_train_indx.npy')
    train_set_unlabeled = MNISTDataSet(data_dir = data_dir,
                            indices = unlabeled_indx,
                            train_set = True)

    print('number labeled: ', len(train_set_labeled))
    print('number unlabeled: ', len(train_set_unlabeled))

    if eval_test_set:
        # get test set as usual
        print('evaluating on test set. ')
        test_set = MNISTDataSet(data_dir = data_dir,
                                train_set = False)
    else:
        print('evaluating on validation set. ')
        validation_indx = np.load(train_test_split_folder + 'validation_indx.npy')
        test_set = MNISTDataSet(data_dir = data_dir,
                                indices = validation_indx,
                                train_set = True)

    return train_set_labeled, train_set_unlabeled, test_set
