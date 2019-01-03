import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Categorical

import modeling_lib
import mnist_data_utils

import timeit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import sys
sys.path.insert(0, '../../../rb_utils/')
sys.path.insert(0, '../../rb_utils/')
import rao_blackwellization_lib as rb_lib

class MLPEncoder(nn.Module):
    def __init__(self, latent_dim = 5,
                    slen = 28):

        super(MLPEncoder, self).__init__()

        # image / model parameters
        self.slen = slen
        self.n_pixels = self.slen ** 2
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(self.n_pixels, 256)
        self.fc2 = nn.Linear(256, self.latent_dim * 2)

        self.tanh = torch.nn.Tanh()

    def forward(self, image):

        h = image.view(-1, self.n_pixels)

        h = self.tanh(self.fc1(h))
        h = self.fc2(h)

        latent_mean = h[:, 0:self.latent_dim]
        latent_log_std = h[:, self.latent_dim:(2 * self.latent_dim)]

        return latent_mean, latent_log_std

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim = 5,
                    slen = 28):

        super(MLPDecoder, self).__init__()

        # image / model parameters
        self.slen = slen
        self.n_pixels = self.slen ** 2
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(self.latent_dim, 256)
        self.fc2 = nn.Linear(256, self.n_pixels)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, latent_samples):

        h = self.tanh(self.fc1(latent_samples))
        h = self.fc2(h)

        return self.sigmoid(h).view(-1, 1, self.slen, self.slen)

class HandwritingVAE(nn.Module):
    def __init__(self, latent_dim = 5,
                        slen = 28):

        super(HandwritingVAE, self).__init__()

        self.latent_dim = latent_dim
        self.slen = slen

        self.encoder = MLPEncoder(self.latent_dim, self.slen)
        self.decoder = MLPDecoder(self.latent_dim, self.slen)

    def forward(self, image):

        # get latent means and std
        latent_mean, latent_log_std = self.encoder(image)

        # sample latent params
        latent_samples = torch.randn(latent_mean.shape).to(device) * \
                            torch.exp(latent_log_std) + latent_mean

        # pass through decoder
        recon_mean = self.decoder(latent_samples)

        return recon_mean, latent_mean, latent_log_std, latent_samples

    def get_loss(self, image):

        recon_mean, latent_mean, latent_log_std, latent_samples = \
            self.forward(image)

        # kl term
        kl_q = modeling_lib.get_kl_q_standard_normal(latent_mean, latent_log_std)

        # bernoulli likelihood
        loglik = modeling_lib.get_bernoulli_loglik(recon_mean, image)

        return -loglik + kl_q


class Flatten(nn.Module):
    def forward(self, tensor):
        return tensor.view(tensor.size(0), -1)

class PixelAttention(nn.Module):
    def __init__(self, slen):

        super(PixelAttention, self).__init__()

        # attention mechanism

        # convolution layers
        self.attn = nn.Sequential(
            nn.Conv2d(1, 7, 3, padding=0),
            nn.ReLU(),

            nn.Conv2d(7, 7, 3, padding=0),
            nn.ReLU(),

            nn.Conv2d(7, 7, 3, padding=0),
            nn.ReLU(),

            nn.Conv2d(7, 1, 3, padding=0),
            Flatten())

        # one more fully connected layer
        self.slen = slen
        self.fc1 = nn.Linear((self.slen - 8)**2, self.slen**2)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, image):
        h = self.attn(image)
        h = self.fc1(h)

        return self.log_softmax(h)

class MovingHandwritingVAE(nn.Module):
    def __init__(self, latent_dim = 5,
                        mnist_slen = 28,
                        full_slen = 68):

        super(MovingHandwritingVAE, self).__init__()

        # mnist_slen is size of the mnist digit
        # full_slen is the size of the padded image

        self.latent_dim = latent_dim
        self.mnist_slen = mnist_slen
        self.full_slen = full_slen

        self.mnist_vae = HandwritingVAE(latent_dim = self.latent_dim,
                                        slen = self.mnist_slen + 1)

        self.pixel_attention = PixelAttention(slen = self.full_slen)

        # TODO check these grids
        # cache meshgrid required for padding images
        r0 = (self.full_slen - 1) / 2
        self.grid_out = \
            torch.FloatTensor(
                np.mgrid[0:self.full_slen, 0:self.full_slen].transpose() - r0).to(device)

        # cache meshgrid required for cropping image
        r = self.mnist_slen // 2
        self.grid0 = torch.from_numpy(\
                    np.mgrid[(-r):(r+1), (-r):(r+1)].transpose([2, 1, 0])).to(device)

    def pad_image(self, image, pixel_2d):
        return mnist_data_utils.pad_image(image,
                                    pixel_2d,
                                    grid_out = self.grid_out)

    def crop_image(self, image, pixel_2d):
        return mnist_data_utils.crop_image(image,
                            pixel_2d, grid0 = self.grid0)

    def forward_cond_pixel_1d(self, image, pixel_1d):
        # image should be N x slen x slen
        assert len(image.shape) == 4
        assert image.shape[1] == 1
        assert image.shape[2] == self.full_slen
        assert image.shape[3] == self.full_slen

        pixel_2d = mnist_data_utils.pixel_1d_to_2d(self.full_slen,
                                    padding = 0,
                                    pixel_1d = pixel_1d)

        image_cropped = self.crop_image(image, pixel_2d)

        # pass through mnist vae
        recon_mean_cropped, latent_mean, latent_log_std, latent_samples = \
            self.mnist_vae(image_cropped)

        recon_mean = self.pad_image(recon_mean_cropped, pixel_2d)

        return recon_mean, latent_mean, latent_log_std, latent_samples, pixel_2d

    def get_loss_cond_pixel_1d(self, image, pixel_1d):

        # forward
        recon_mean, latent_mean, latent_log_std, latent_samples, pixel_2d = \
                        self.forward_cond_pixel_1d(image, pixel_1d)

        # kl term
        kl_latent = \
            modeling_lib.get_kl_q_standard_normal(latent_mean, latent_log_std)

        # bernoulli likelihood
        loglik = modeling_lib.get_bernoulli_loglik(recon_mean, image)

        return -loglik + kl_latent

    def get_rb_loss(self, image,
                        topk = 0,
                        use_baseline = True,
                        n_samples = 1,
                        true_pixel_2d = None):

        if true_pixel_2d is None:
            log_class_weights = self.pixel_attention(image)
            class_weights = torch.exp(log_class_weights)
        else:
            class_weights = self._get_class_weights_from_pixel_2d(true_pixel_2d)
            log_class_weights = torch.log(class_weights)

        # kl term
        kl_pixel_probs = (class_weights * log_class_weights).sum()

        f_pixel = lambda i : self.get_loss_cond_pixel_1d(image, i)

        avg_pm_loss = 0.0
        # TODO: n_samples would be more elegant as an
        # argument to get_partial_marginal_loss
        for k in range(n_samples):
            pm_loss = rb_lib.get_raoblackwell_ps_loss(f_pixel,
                                        log_class_weights, topk,
                                        use_baseline = use_baseline)

            avg_pm_loss += pm_loss / n_samples

        map_locations = torch.argmax(log_class_weights.detach(), dim = 1)
        map_cond_losses = f_pixel(map_locations).sum()

        return avg_pm_loss + kl_pixel_probs, map_cond_losses

    def _pixel_1d_from_2d(self, pixel_2d):
        return pixel_2d[:, 0] * self.full_slen + pixel_2d[:, 1]

    def _get_class_weights_from_pixel_2d(self, pixel_2d):

        batchsize = pixel_2d.shape[0]
        seq_tensor = torch.LongTensor([i for i in range(batchsize)])

        pixel_1d = self._pixel_1d_from_2d(pixel_2d)

        class_weights = torch.ones((batchsize, self.full_slen ** 2)) * 1e-12

        class_weights[seq_tensor, pixel_1d] = 1

        return (class_weights / class_weights.sum(1, keepdim = True)).to(device)
