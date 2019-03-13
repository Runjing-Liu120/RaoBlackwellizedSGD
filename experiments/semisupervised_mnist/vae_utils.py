import torch
from torch import nn

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import sys
sys.path.insert(0, '../../../rb_utils/')
sys.path.insert(0, '../../rb_utils/')

from common_utils import get_one_hot_encoding_from_int


def get_class_label_cross_entropy(log_class_weights, labels):
        assert np.all(log_class_weights.detach().cpu().numpy() <= 0)
        assert log_class_weights.shape[0] == len(labels)
        # assert log_class_weights.shape[1] == self.n_classes
        n_classes = log_class_weights.shape[1]

        return torch.sum(
            -log_class_weights * \
            get_one_hot_encoding_from_int(labels, n_classes), dim = 1)

def get_reconstruction_loss(x_reconstructed, x):
    batch_size = x.shape[0]

    bce_loss = -x * torch.log(x_reconstructed + 1e-8) - \
                    (1 - x) * torch.log(1 - x_reconstructed + 1e-8)

    return bce_loss.view(batch_size, -1).sum(dim = 1)
    # return nn.BCELoss(reduce=False)(x_reconstructed, x).view(batch_size, -1).sum(dim = 1)

def get_kl_divergence_loss(mean, logvar):
    batch_size = mean.shape[0]
    return ((mean**2 + logvar.exp() - 1 - logvar) / 2).view(batch_size, -1).sum(dim = 1)

def get_loss_from_one_hot_label(vae, image, one_hot_label):
    latent_means, latent_std, latent_samples, image_mean = \
        image_reconstructed = vae(image, one_hot_label)

    reconstruction_loss = get_reconstruction_loss(image_mean, image)
    kl_divergence_loss = get_kl_divergence_loss(latent_means, \
                                    2 * torch.log(latent_std))

    return reconstruction_loss + kl_divergence_loss

def get_labeled_loss(vae, image, label):
    one_hot_label = vae.get_one_hot_encoding_from_label(label)

    return get_loss_from_one_hot_label(vae, image, one_hot_label)

def get_classification_accuracy_on_batch(classifier, image, label):
    log_q = classifier(image).detach()
    z_ind = torch.argmax(log_q, dim = 1)

    return torch.mean((z_ind == label).float())


def get_classification_accuracy(classifier, loader,
                                max_images = np.inf):

    n_images = 0.0
    accuracy = 0.0

    for batch_idx, data in enumerate(loader):
        images = data['image'].to(device)
        labels = data['label'].to(device)

        accuracy += \
            get_classification_accuracy_on_batch(classifier, images, labels) * \
                    images.shape[0]

        n_images += images.shape[0]

        if n_images > max_images:
            break

    return accuracy / n_images
