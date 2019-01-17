import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Normal, Categorical, Bernoulli
import torch.nn.functional as F

import timeit

from copy import deepcopy

from sklearn.cluster import KMeans

import itertools

import sys
sys.path.insert(0, '../../rb_utils/')
import rao_blackwellization_lib as rb_lib

softmax = nn.Softmax(dim = 0)
log_softmax = nn.LogSoftmax(dim = 1)

# The variational distribution for the class labels
class GMMEncoder(nn.Module):
    def __init__(self, data_dim, n_classes, hidden_dim = 50):

        super(GMMEncoder, self).__init__()

        # image / model parameters
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # define the linear layers
        self.fc1 = nn.Linear(self.data_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.n_classes)

        self.log_softmax = nn.LogSoftmax(dim = 1)

    def forward(self, x):

        # feed through neural network
        h = F.relu(self.fc1(x))
        # h = F.relu(self.fc2(h))

        fudge_lower_bdd = torch.Tensor([-8])
        h = torch.max(self.fc3(h), fudge_lower_bdd)
        h = torch.min(h, - fudge_lower_bdd)

        log_class_weights = self.log_softmax(h)

        return log_class_weights

def get_normal_loglik(x, mu, log_sigma):
    return - (x - mu)**2 / (2 * torch.exp(log_sigma) ** 2) - log_sigma

class GMMExperiments(object):
    def __init__(self, n_obs, mu0, sigma0, n_clusters, hidden_dim = 30):

        # dimension of the problem
        self.dim = len(mu0)
        self.n_clusters = n_clusters
        self.n_obs = n_obs

        # prior parameters
        self.mu0 = mu0
        self.sigma0 = torch.Tensor([sigma0])
        # uniform prior on weights
        self.prior_weights = torch.ones(self.n_clusters) / self.n_clusters

        # true parameters
        self.set_true_params()
        self.cat_rv = Categorical(probs = self.prior_weights)

        # the encoder
        # self.gmm_encoder = GMMEncoder(data_dim = self.dim,
        #                      n_classes = self.n_clusters,
        #                      hidden_dim = hidden_dim)
        #
        # self.var_params = {'encoder_params': self.gmm_encoder.parameters()}


        # other variational paramters: we use point masses for
        # the means and variances
        self.set_random_var_params()

        # draw data
        self.n_obs = n_obs
        self.y, self.z = self.draw_data(n_obs = n_obs)

    def set_var_params(self, init_mu, init_log_sigma):
        self.var_params['centroids'] = init_mu
        self.var_params['log_sigma'] = init_log_sigma

    def set_random_var_params(self):
        init_mu = torch.randn((self.n_clusters, self.dim)) * self.sigma0 + self.mu0
        init_mu.requires_grad_(True)

        init_log_sigma = torch.log(torch.Tensor([self.true_sigma]))# torch.log(torch.rand(1))
        init_log_sigma.requires_grad_(True)

        self.init_free_class_weights = torch.rand((self.n_obs, self.n_clusters))
        init_free_class_weights = deepcopy(self.init_free_class_weights)
        init_free_class_weights = init_free_class_weights.requires_grad_(True)
        self.var_params = {'free_class_weights': init_free_class_weights}

        self.set_var_params(init_mu, init_log_sigma)

    def set_kmeans_init_var_params(self, n_kmeans_init = 10):

        for i in range(n_kmeans_init):
            km = KMeans(n_clusters = self.n_clusters).fit(self.y)
            enertia = km.inertia_
            if (i == 0):
                enertia_best = enertia
                km_best = deepcopy(km)
            elif (enertia < enertia_best):
                enertia_best = enertia
                km_best = deepcopy(km)

        init_free_class_weights = torch.zeros((self.n_obs, self.n_clusters))
        for n in range(len(km_best.labels_)):
            init_free_class_weights[n, km_best.labels_[n]] = 3.0

        self.init_free_class_weights = deepcopy(init_free_class_weights)

        init_free_class_weights.requires_grad_(True)
        self.var_params['free_class_weights'] = init_free_class_weights
        # init_centroids = torch.Tensor(km_best.cluster_centers_)
        # init_centroids.requires_grad_(True)
        # self.var_params['centroids'] = init_centroids

    def set_true_params(self):
        # draw means from the prior
        # each row is a cluster mean
        self.true_mus = torch.randn((self.n_clusters, self.dim)) * self.sigma0 + self.mu0

        # just set a data variance
        self.true_sigma = 1.0

    def draw_data(self, n_obs = 1):

        y = torch.zeros((n_obs, self.dim))
        z = torch.zeros(n_obs)
        for i in range(n_obs):
            # class belonging
            z_sample = self.cat_rv.sample()
            z[i] = z_sample

            # observed data
            y[i, :] = self.true_mus[z_sample, :] + torch.randn(2) * self.true_sigma

        # some indices we cache and use later
        self.seq_tensor = torch.LongTensor([i for i in range(n_obs)])

        return y, z

    def get_log_q(self):
        # self.log_class_weights = self.gmm_encoder.forward(self.y)

        fudge_lower_bdd = torch.Tensor([-8])
        self.log_class_weights = log_softmax(torch.max(self.var_params['free_class_weights'], fudge_lower_bdd)) #

        return self.log_class_weights

    def _get_centroid_mask(self, z):
        mask = torch.zeros((self.n_obs, self.n_clusters))
        mask[self.seq_tensor, z] = 1

        return mask.detach()

    def f_one_hot_z(self, one_hot_z):
        centroids = self.var_params['centroids'] #
        log_sigma = torch.log(torch.Tensor([self.true_sigma]))  #

        centroids_masked = torch.matmul(one_hot_z, centroids)

        loglik_z = get_normal_loglik(self.y, centroids_masked, log_sigma).sum(dim = 1)

        mu_prior_term = get_normal_loglik(centroids, self.mu0, torch.log(self.sigma0)).mean()

        z_prior_term = 0.0 # torch.log(self.prior_weights[z])

        z_entropy_term = (- torch.exp(self.log_class_weights) * self.log_class_weights).mean()

        # print('z_ent_term', z_entropy_term)
        # print('mu_prior_term', mu_prior_term)
        # print('loglik', loglik_z)

        return - (loglik_z + mu_prior_term + z_prior_term + z_entropy_term)

    def f_z(self, z):

        centroid_mask = self._get_centroid_mask(z)

        return self.f_one_hot_z(centroid_mask)

    def get_pm_loss(self, topk, grad_estimator, n_samples = 1):
        log_q = self.get_log_q()

        pm_loss = 0.0
        # print(n_samples)
        for i in range(n_samples):
            pm_loss += rb_lib.get_raoblackwell_ps_loss(self.f_one_hot_z, log_q,
                    topk, grad_estimator)

        return pm_loss / n_samples

    def get_full_loss(self):
        log_q = self.get_log_q()
        class_weights = torch.exp(log_q)
        return rb_lib.get_full_loss(self.f_one_hot_z, class_weights)
