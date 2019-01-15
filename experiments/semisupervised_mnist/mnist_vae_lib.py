import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

import vae_utils

import sys
sys.path.insert(0, '../../../rb_utils/')
sys.path.insert(0, '../../rb_utils/')

from common_utils import get_one_hot_encoding_from_int


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLPEncoder(nn.Module):
    def __init__(self, latent_dim = 5,
                    slen = 28,
                    n_classes = 10):
        # the encoder returns the mean and variance of the latent parameters
        # given the image and its class (one hot encoded)

        super(MLPEncoder, self).__init__()

        # image / model parameters
        self.n_pixels = slen ** 2
        self.latent_dim = latent_dim
        self.slen = slen
        self.n_classes = n_classes

        # define the linear layers
        self.fc1 = nn.Linear(self.n_pixels + self.n_classes, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, latent_dim * 2)


    def forward(self, image, one_hot_label):
        assert one_hot_label.shape[1] == self.n_classes # label should be one hot encoded
        assert image.shape[0] == one_hot_label.shape[0]

        # feed through neural network
        h = image.view(-1, self.n_pixels)
        h = torch.cat((h, one_hot_label), dim = 1)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)

        # get means, std, and class weights
        indx1 = self.latent_dim
        indx2 = 2 * self.latent_dim

        latent_means = h[:, 0:indx1]
        latent_std = torch.exp(h[:, indx1:indx2])

        return latent_means, latent_std


class Classifier(nn.Module):
    def __init__(self, slen = 28, n_classes = 10):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()

        self.slen = slen
        self.n_pixels = slen ** 2
        self.n_classes = n_classes

        self.fc1 = nn.Linear(self.n_pixels, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, n_classes)

        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, image):
        h = image.view(-1, self.n_pixels)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)

        return self.log_softmax(h)

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim = 5,
                        n_classes = 10,
                        slen = 28):

        # This takes the latent parameters and returns the
        # mean and variance for the image reconstruction

        super(MLPDecoder, self).__init__()

        # image/model parameters
        self.n_pixels = slen ** 2
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.slen = slen

        self.fc1 = nn.Linear(latent_dim + n_classes, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.n_pixels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, latent_params, one_hot_label):
        assert latent_params.shape[1] == self.latent_dim
        assert one_hot_label.shape[1] == self.n_classes # label should be one hot encoded
        assert latent_params.shape[0] == one_hot_label.shape[0]

        h = torch.cat((latent_params, one_hot_label), dim = 1)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)

        h = h.view(-1, self.slen, self.slen)

        image_mean = self.sigmoid(h)

        return image_mean

class MNISTVAE(nn.Module):

    def __init__(self, encoder, decoder):
        super(MNISTVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        assert self.encoder.latent_dim == self.decoder.latent_dim
        assert self.encoder.n_classes == self.decoder.n_classes
        assert self.encoder.slen == self.decoder.slen

        # save some parameters
        self.latent_dim = self.encoder.latent_dim
        self.n_classes = self.encoder.n_classes
        self.slen = self.encoder.slen

    def get_one_hot_encoding_from_label(self, label):
        return get_one_hot_encoding_from_int(label, self.n_classes)

    def forward(self, image, one_hot_label):

        assert one_hot_label.shape[0] == image.shape[0]
        assert one_hot_label.shape[1] == self.n_classes

        # pass through encoder
        latent_means, latent_std = self.encoder(image, one_hot_label)

        # sample latent dimension
        latent_samples = torch.randn(latent_means.shape).to(device) * \
                            latent_std + latent_means

        assert one_hot_label.shape[0] == latent_samples.shape[0]
        assert one_hot_label.shape[1] == self.n_classes

        # pass through decoder
        image_mean = self.decoder(latent_samples, one_hot_label)

        return latent_means, latent_std, latent_samples, image_mean


def get_mnist_vae_and_classifier(latent_dim = 5,
                                    n_classes = 10,
                                    slen = 28):

    encoder = MLPEncoder(latent_dim = latent_dim,
                            slen = slen,
                            n_classes = n_classes)
    decoder = MLPDecoder(latent_dim = latent_dim,
                                    slen = slen,
                                    n_classes = n_classes)
    vae = MNISTVAE(encoder, decoder)

    classifier = Classifier(n_classes = n_classes, slen = slen)

    return vae, classifier
