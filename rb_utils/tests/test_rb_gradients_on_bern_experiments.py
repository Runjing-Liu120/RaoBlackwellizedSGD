#!/usr/bin/env python3

import numpy as np
import scipy as sp

import torch

import unittest

import rao_blackwellization_lib as rb_lib
import baselines_lib as bs_lib

import sys
sys.path.insert(0, './../experiments/bernoulli_experiments/')
import bernoulli_experiments_lib as bern_lib

import torch.optim as optim

from copy import deepcopy

np.random.seed(24524)

class TestBernoulliGradients(unittest.TestCase):
    def set_params(self):
        # fixed parameters
        d = 3
        p0 = torch.Tensor([0.6, 0.51, 0.48])

        # phi at which we will evaluate gradit
        phi0 = torch.Tensor([2.3])
        phi0.requires_grad_(True)

        # set up optimizer
        params = [phi0]
        optimizer = optim.SGD(params, lr = 1.0)

        # the class
        bern_experiment = bern_lib.BernoulliExperiments(p0, d, phi0)
        bern_experiment.set_var_params(deepcopy(phi0))

        return phi0, optimizer, bern_experiment

    def test_reinforce(self):
        phi0, optimizer, bern_experiment = self.set_params()

        # true gradient
        loss = bern_experiment.get_full_loss()
        loss.backward()

        true_grad = deepcopy(bern_experiment.var_params['phi'].grad)
        print('true_grad', true_grad.numpy())

        # analytically integrate reinforce gradient
        bern_experiment.set_var_params(deepcopy(phi0))
        optimizer.zero_grad()
        ps_loss = bern_experiment.get_pm_loss(topk = 8,
                                    grad_estimator = bs_lib.reinforce)

        ps_loss.backward()
        reinforce_analytic_grad = \
            deepcopy(bern_experiment.var_params['phi'].grad)
        print('reinforce_analytic_grad', reinforce_analytic_grad.numpy())

        assert reinforce_analytic_grad == true_grad

        # check sampling error of reinforce estimator
        n_samples = 10000
        reinforce_grads = bern_lib.sample_bern_gradient(phi0, bern_experiment,
                                          topk = 0,
                                          grad_estimator = bs_lib.reinforce,
                                          n_samples = n_samples)

        mean_reinforce_grad = torch.mean(reinforce_grads).numpy()
        std_reinforce_grad = (torch.std(reinforce_grads).numpy() / \
                                np.sqrt(n_samples))

        print('mean_reinforce_grad, ', mean_reinforce_grad)
        print('tol ', 3 * std_reinforce_grad)

        assert np.abs(true_grad.numpy() - mean_reinforce_grad) < \
                        (3 * std_reinforce_grad)

        # check with topk = 3
        # check sampling error of reinforce estimator
        n_samples = 10000
        rb_reinforce_grads = \
            bern_lib.sample_bern_gradient(phi0, bern_experiment,
                                      topk = 3,
                                      grad_estimator = bs_lib.reinforce,
                                      n_samples = n_samples)

        mean_rb_reinforce_grad = torch.mean(rb_reinforce_grads).numpy()
        std_rb_reinforce_grad = (torch.std(rb_reinforce_grads).numpy() / \
                                    np.sqrt(n_samples))

        print('mean_rb_reinforce_grad, ', mean_rb_reinforce_grad)
        print('tol ', 3 * std_rb_reinforce_grad)

        assert np.abs(true_grad.numpy() - mean_rb_reinforce_grad) < \
                        (3 * std_rb_reinforce_grad)

if __name__ == '__main__':
    unittest.main()
