import numpy as np

import torch

from torch.distributions import Categorical
from common_utils import get_one_hot_encoding_from_int

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sample_class_weights(class_weights):
    # draw a sample from Categorical variable with
    # probabilities class_weights

    cat_rv = Categorical(probs = class_weights)
    return cat_rv.sample().detach()

def get_reinforce_grad_sample(conditional_loss, log_class_weights,
                                baseline = 0.0):
    # computes the REINFORCE gradient estimate
    assert len(conditional_loss) == len(log_class_weights)

    return (conditional_loss - baseline).detach() * log_class_weights

def reinforce(conditional_loss_fun, log_class_weights,
                class_weights_detached, seq_tensor,
                z_sample, grad_estimator_kwargs = None):
    # z_sample should be a vector of categories

    # conditional_loss_fun is a function that takes in a one hot encoding
    # of z and returns the loss

    assert len(z_sample) == log_class_weights.shape[0]

    # compute loss from those categories
    n_classes = log_class_weights.shape[1]
    one_hot_z_sample = get_one_hot_encoding_from_int(z_sample, n_classes)
    conditional_loss_fun_i = conditional_loss_fun(one_hot_z_sample)
    assert len(conditional_loss_fun_i) == log_class_weights.shape[0]

    # get log class_weights
    log_class_weights_i = log_class_weights[seq_tensor, z_sample]

    return get_reinforce_grad_sample(conditional_loss_fun_i,
                    log_class_weights_i, baseline = 0.0) + \
                        conditional_loss_fun_i

def reinforce_w_double_sample_baseline(\
            conditional_loss_fun, log_class_weights,
            class_weights_detached, seq_tensor, z_sample,
            grad_estimator_kwargs = None):

    assert len(z_sample) == log_class_weights.shape[0]

    # compute loss from those categories
    n_classes = log_class_weights.shape[1]
    one_hot_z_sample = get_one_hot_encoding_from_int(z_sample, n_classes)
    conditional_loss_fun_i = conditional_loss_fun(one_hot_z_sample)
    assert len(conditional_loss_fun_i) == log_class_weights.shape[0]

    # get log class_weights
    log_class_weights_i = log_class_weights[seq_tensor, z_sample]

    # get baseline
    z_sample2 = sample_class_weights(class_weights_detached)
    one_hot_z_sample2 = get_one_hot_encoding_from_int(z_sample2, n_classes)
    baseline = conditional_loss_fun(one_hot_z_sample2)

    return get_reinforce_grad_sample(conditional_loss_fun_i,
                    log_class_weights_i, baseline) + conditional_loss_fun_i

# def mu_prop(conditional_loss_fun, log_class_weights,
#             class_weights_detached, seq_tensor, z_sample):
#     assert len(z_sample) == log_class_weights.shape[0]
#
#     # compute loss from those categories
#     conditional_loss_fun_i = conditional_loss_fun(z_sample)
#     assert len(conditional_loss_fun_i) == log_class_weights.shape[0]
#
#     # get log class_weights
#     log_class_weights_i = log_class_weights[seq_tensor, z_sample]
#
#     # compute baseline with mu_prop
#     # we will evalute Taylor expansion about z_bar
#     z_bar = torch.ones(log_class_weights.shape) * 1 / log_class_weights.shape[1]
#     z_bar.requires_grad_(True)
#     # get gradient
#     f_z_bar = conditional_loss_fun(z_bar)
#
#     f_z_bar.backwards()
#     f_grad = z_bar.grad
#
#     # get baseline
#     baseline = f_z_bar + f_grad

def rebar(conditional_loss_fun, log_class_weights,
            class_weights_detached, seq_tensor, z_sample,
            temperature = 1., eta = 1.):

    n_classes = log_class_weights.shape[1]

    # sample z using gumbel reparameterization
    z_softmax = gumbel_softmax_sample(log_class_weights, temperature)
    _, z_ind = z_softmax.max(dim=-1)
    z_one_hot = get_one_hot_encoding_from_int(z_ind, n_classes)

    # log class weights evaluated at sampeld z
    log_class_weights_i = log_class_weights[seq_tensor, z_ind]

    # conditional softmax term
    z_cond_softmax = \
        gumbel_softmax_conditional_sample(log_class_weights, temperature,
                                        z_one_hot)
    # reinforce term:
    reinforce_term = (conditional_loss_fun(z_one_hot) - \
                        eta * conditional_loss_fun(z_cond_softmax)).detach() * \
                        log_class_weights_i
    # correction term:
    correction_term = eta * conditional_loss_fun(z_softmax) - \
                        eta * conditional_loss_fun(z_cond_softmax)

    return reinforce_term + correction_term
