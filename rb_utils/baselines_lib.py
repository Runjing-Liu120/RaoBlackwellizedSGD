import numpy as np

import torch

from torch.distributions import Categorical

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

def reinforce(conditional_loss_fun, log_class_weights, z_sample):
    assert len(z_sample) == log_class_weights.shape[0]

    # compute loss from those categories
    conditional_loss_fun_i = conditional_loss_fun(z_sample)

    assert len(conditional_loss_fun_i) == log_class_weights.shape[0]

    # get log class_weights
    seq_tensor = torch.LongTensor([i for i in range(log_class_weights.shape[0])])
    log_class_weights_i = log_class_weights[seq_tensor, z_sample]

    return get_reinforce_grad_sample(conditional_loss_fun_i,
                    log_class_weights_i, baseline = 0.0) + \
                        conditional_loss_fun_i

def reinforce_w_double_sample_baseline(\
            conditional_loss_fun, log_class_weights, z_sample):

    assert len(z_sample) == log_class_weights.shape[0]

    # compute loss from those categories
    conditional_loss_fun_i = conditional_loss_fun(z_sample)
    assert len(conditional_loss_fun_i) == log_class_weights.shape[0]

    # get log class_weights
    seq_tensor = torch.LongTensor([i for i in range(log_class_weights.shape[0])])
    log_class_weights_i = log_class_weights[seq_tensor, z_sample]

    # get baseline
    class_weights = torch.exp(log_class_weights.detach())
    z_sample2 = sample_class_weights(class_weights)
    baseline = conditional_loss_fun(z_sample2)

    return get_reinforce_grad_sample(conditional_loss_fun_i,
                    log_class_weights_i, baseline) + conditional_loss_fun_i
