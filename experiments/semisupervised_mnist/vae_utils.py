import torch
from torch import nn

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_one_hot_encoding_from_int(z, n_classes):
    # z is a sequence of integers in {0, ...., n_classes}
    #  corresponding to categorires
    # we return a matrix of shape len(z) x n_classes
    # corresponding to the one hot encoding of z

    assert (torch.max(z) + 1) <= n_classes

    batch_size = len(z)
    one_hot_z = torch.zeros((batch_size, n_classes)).to(device)

    for i in range(n_classes):
        one_hot_z[z == i, i] = 1.

    return one_hot_z

def get_class_label_cross_entropy(log_class_weights, labels):
        assert np.all(log_class_weights.detach().cpu().numpy() <= 0)
        assert log_class_weights.shape[0] == len(labels)
        # assert log_class_weights.shape[1] == self.n_classes
        n_classes = log_class_weights.shape[1]

        return torch.sum(
            -log_class_weights * \
            get_one_hot_encoding_from_int(labels, n_classes))

def get_reconstruction_loss(x_reconstructed, x):
    batch_size = x.shape[0]
    return nn.BCELoss(reduce=False)(x_reconstructed, x).view(batch_size, -1).sum(dim = 1)

def get_kl_divergence_loss(mean, logvar):
    batch_size = mean.shape[0]
    return ((mean**2 + logvar.exp() - 1 - logvar) / 2).view(batch_size, -1).sum(dim = 1)

def get_labeled_loss(vae, image, label):
    latent_means, latent_std, latent_samples, image_mean = \
        image_reconstructed = vae(image, label)

    reconstruction_loss = get_reconstruction_loss(image_mean, image)
    kl_divergence_loss = get_kl_divergence_loss(latent_means, \
                                    2 * torch.log(latent_std))

    return reconstruction_loss + kl_divergence_loss