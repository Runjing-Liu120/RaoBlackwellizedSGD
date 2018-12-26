import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Normal, Categorical, Bernoulli
import torch.nn.functional as F

import sys
sys.path.insert(0, '../../rb_utils/')
import rao_blackwellization_lib as rb_lib

import timeit

from copy import deepcopy

import itertools

softmax = nn.Softmax(dim = 0)
sigmoid = nn.Sigmoid()

def get_bernoulli_log_prob(e_b, draw):
    return torch.log(e_b + 1e-12) * torch.sum(draw) + \
                torch.log(1 - e_b + 1e-12) * torch.sum(1 - draw)

# class to run Bernoulli Experiments
class BernoulliExperiments(object):
    def __init__(self, p0, dim, phi0):
        self.p0 = p0
        self.dim = dim

        self.set_draw_array()

        self.var_params = {'phi': deepcopy(phi0)}

    def set_var_params(self, phi):
        self.var_params = {'phi': deepcopy(phi)}

    def set_draw_array(self):
        # defines the 2**d vector of possible combinations

        self.draw_array = []
        i = 0
        for draw in itertools.product(range(2), repeat=self.dim):
            draw_tensor = torch.Tensor(draw)
            self.draw_array.append(draw_tensor)

    def f_z(self, i):
        # returns the loss for the ith entry in draw array
        return torch.Tensor([torch.sum((self.draw_array[i] - self.p0) ** 2)])

    def get_log_q(self):
        # returns a vector of log probabilities for all the possible draws
        log_probs = torch.zeros((1, len(self.draw_array)))

        e_b = sigmoid(self.var_params['phi'])

        for i in range(len(self.draw_array)):
            draw_tensor = torch.Tensor(self.draw_array[i])
            log_probs[0, i] = get_bernoulli_log_prob(e_b, draw_tensor)

        return log_probs

    def get_bernoulli_prob_vec(self):
        # returns a vector of probabilities for all the possible draws
        return torch.exp(self.get_bernoulli_log_prob_vec())

    def get_bernoulli_log_prob_i(self, i):
        # returns the log probabilities for draw i
        e_b = sigmoid(self.var_params['phi'])
        return get_bernoulli_log_prob(e_b, self.draw_array[i])

    def get_pm_loss(self, alpha, topk, use_baseline):
        log_q = self.get_log_q()
        pm_loss = rb_lib.get_raoblackwell_ps_loss(self.f_z, log_q, topk,
                                    use_baseline = use_baseline)

        return pm_loss

    def get_full_loss(self):
        log_q = self.get_log_q()
        class_weights = torch.exp(log_q)
        return rb_lib.get_full_loss(self.f_z, class_weights)


def sample_bern_gradient(phi0, bern_experiment, topk, alpha,
                            use_baseline = True,
                            n_samples = 10000):
    params = [phi0]
    optimizer = optim.SGD(params, lr = 1.0)

    grad_array = torch.zeros(n_samples)

    for i in range(n_samples):
        bern_experiment.set_var_params(deepcopy(phi0))
        optimizer.zero_grad()
        ps_loss = bern_experiment.get_pm_loss(alpha = alpha, topk = topk, use_baseline = True)
        ps_loss.backward()

        grad_array[i] = bern_experiment.var_params['phi'].grad

    return grad_array
