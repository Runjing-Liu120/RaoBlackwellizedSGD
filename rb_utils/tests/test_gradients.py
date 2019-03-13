#!/usr/bin/env python3

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import itertools

from copy import deepcopy

import sys
sys.path.insert(0, '../')
import optimization_lib as optim_lib
import rao_blackwellization_lib as rb_lib
import baselines_lib as bs_lib

from toy_experiment_lib import ToyExperiment

import unittest

np.random.seed(454)
_ = torch.manual_seed(454)

def assert_close(x, y, tol):
    diff = np.abs(x - y)
    assert diff < tol, 'difference = {}, tolerance = {}'.format(diff, tol)

class TestRB(unittest.TestCase):
    def get_params_get_true_grad(self):
        # fixed parameters
        k = 5
        p0 = torch.rand(k)

        eta = torch.Tensor([5.])
        toy_experiment = ToyExperiment(eta, p0)

        # get true gradient
        toy_experiment.set_parameter(eta)
        full_loss = toy_experiment.get_full_loss()
        full_loss.backward()
        true_grad = toy_experiment.eta.grad

        return k, eta, toy_experiment, true_grad

    def test_reinforce(self):
        k, eta, toy_experiment, true_grad = self.get_params_get_true_grad()

        toy_experiment.set_parameter(eta)

        # analytically integrate out REINFORCE and check against true gradient
        pm_loss = toy_experiment.get_pm_loss(topk = k,
                    grad_estimator = bs_lib.reinforce)
        pm_loss.backward()

        assert_close(true_grad, toy_experiment.eta.grad, tol = 1e-6)

        # sample reinforce and check against true gradient
        n_samples = 10000

        reinforce_grads = torch.zeros(n_samples)
        reinforce_pm_losses = torch.zeros(n_samples)

        for i in range(n_samples):
            toy_experiment.set_parameter(eta)
            pm_loss = toy_experiment.get_pm_loss(topk = 0,
                        grad_estimator = bs_lib.reinforce)
            reinforce_pm_losses[i] = pm_loss
            pm_loss.backward()

            reinforce_grads[i] = toy_experiment.eta.grad

        assert_close(torch.mean(reinforce_grads),
                 true_grad,
                 3 * torch.std(reinforce_grads) / np.sqrt(n_samples))
    def test_reinforce_bs(self):
        # add a simple baseline to reinforce
        # assert it is unbiased

        k, eta, toy_experiment, true_grad = self.get_params_get_true_grad()

        n_samples = 10000

        reinforce_cv_grads = torch.zeros(n_samples)

        for i in range(n_samples):
            toy_experiment.set_parameter(eta)
            pm_loss = toy_experiment.get_pm_loss(
                    topk = 0,
                    grad_estimator = bs_lib.reinforce_w_double_sample_baseline)
            pm_loss.backward()

            reinforce_cv_grads[i] = toy_experiment.eta.grad

        assert_close(torch.mean(reinforce_cv_grads),
             true_grad,
             3 * torch.std(reinforce_cv_grads) / np.sqrt(n_samples))

    def test_rb_reinforce(self):
        k, eta, toy_experiment, true_grad = self.get_params_get_true_grad()

        # assert our rao blackwellization returns an unbiased gradient
        n_samples = 10000

        reinforce_rb_grads = torch.zeros(n_samples)

        for i in range(n_samples):
            toy_experiment.set_parameter(eta)
            pm_loss = toy_experiment.get_pm_loss(topk = 3,
                grad_estimator = bs_lib.reinforce_w_double_sample_baseline)
            pm_loss.backward()

            reinforce_rb_grads[i] = toy_experiment.eta.grad

        assert_close(torch.mean(reinforce_rb_grads),
             true_grad,
             3 * torch.std(reinforce_rb_grads) / np.sqrt(n_samples))

    def test_rebar(self):
        k, eta, toy_experiment, true_grad = self.get_params_get_true_grad()

        # sample rebar gradient estimator, assert it is unbiased
        n_samples = 10000
        rebar_grads = torch.zeros(n_samples)
        rebar_pm_losses = torch.zeros(n_samples)
        for i in range(n_samples):
            toy_experiment.set_parameter(eta)
            pm_loss = toy_experiment.get_pm_loss(
                    topk = 0,
                    grad_estimator = bs_lib.relax,
                    grad_estimator_kwargs = {'temperature': torch.Tensor([0.1]),
                                                'eta': 1.0})
            rebar_pm_losses[i] = pm_loss
            pm_loss.backward()

            rebar_grads[i] = toy_experiment.eta.grad

        assert_close(true_grad, torch.mean(rebar_grads),
             tol = 3 * torch.std(rebar_grads) / np.sqrt(n_samples))

    def test_nvil(self):
        k, eta, toy_experiment, true_grad = self.get_params_get_true_grad()

        n_samples = 10000

        nvil_grads = torch.zeros(n_samples)

        baseline_nn = bs_lib.BaselineNN(slen = 5)

        for i in range(n_samples):
            toy_experiment.set_parameter(eta)
            pm_loss = toy_experiment.get_pm_loss(
                        topk = 0,
                        grad_estimator = bs_lib.nvil,
                        grad_estimator_kwargs = {'baseline_nn': baseline_nn})
            pm_loss.backward()

            nvil_grads[i] = toy_experiment.eta.grad

        assert_close(true_grad, torch.mean(nvil_grads),
             tol = 3 * torch.std(nvil_grads) / np.sqrt(n_samples))

if __name__ == '__main__':
    unittest.main()
