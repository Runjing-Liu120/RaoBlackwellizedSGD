import numpy as np

import torch

from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_concentrated_mask(class_weights, topk):

    """
    Returns a logical mask indicating the categories with the top k largest
    probabilities, as well as the catogories corresponding to those with the
    top k largest probabilities.
    """

    mask_topk = torch.zeros(class_weights.shape).to(device)

    # TODO: can we cache this somehow?
    seq_tensor = torch.LongTensor([i for i in range(class_weights.shape[0])])

    if topk > 0:
        _, topk_domain = torch.topk(class_weights, topk)

        for i in range(topk):
            mask_topk[seq_tensor, topk_domain[:, i]] = 1
    else:
        topk_domain = None

    return mask_topk, topk_domain, seq_tensor

def get_full_loss(conditional_loss_fun, class_weights):
    """
    Returns the loss averaged over the class weights.
    `conditional_loss_fun` is a function that returns the loss conditional
    on an instance of the categorical random variable.
    """

    full_loss = 0.0

    for i in range(class_weights.shape[1]):

        conditional_loss = conditional_loss_fun(i)
        assert len(conditional_loss) == class_weights.shape[0]

        full_loss = full_loss + class_weights[:, i] * conditional_loss

    return full_loss.sum()

def sample_class_weights(class_weights):
    # draw a sample from Categorical variable with
    # probabilities class_weights

    cat_rv = Categorical(probs = class_weights)
    return cat_rv.sample().detach()

def get_reinforce_grad_sample(conditional_loss, log_class_weights,
                                baseline = 0.0):
    # computes the REINFORCE gradient estimate

    return (conditional_loss - baseline).detach() * log_class_weights

def get_raoblackwell_ps_loss(conditional_loss_fun, log_class_weights, topk,
                                use_baseline = True):

    """
    Returns a pseudo_loss, such that the gradient obtained by calling
    pseudo_loss.backwards() is unbiased for the true loss

    Parameters
    ----------
    conditional_loss_fun : function
        A function that returns the loss conditional on an instance of the
        categorical random variable. It must take in a vector of categorical
        random variables and return a vector of losses.
    log_class_weights : Tensor
        A tensor of shape batchsize x n_categories of the log class weights
    topk : Integer
        The number of categories to sum over
    use_baseline :
        Whether a baseline will be used to compute the REINFORCE estimator

    """

    # class weights from the variational distribution
    assert np.all(log_class_weights.detach().cpu().numpy() <= 0)
    class_weights = torch.exp(log_class_weights.detach())

    # this is the indicator C_\alpha
    concentrated_mask, topk_domain, seq_tensor = \
        get_concentrated_mask(class_weights, topk)
    concentrated_mask = concentrated_mask.float().detach()

    ############################
    # compute the summed term
    summed_term = 0.0

    for i in range(topk):
        # get categories to be summed
        summed_indx = topk_domain[:, i]

        # compute loss from those categories
        conditional_loss_fun_i = conditional_loss_fun(summed_indx)
        assert len(conditional_loss_fun_i) == log_class_weights.shape[0]

        # get log class_weights
        log_class_weights_i = log_class_weights[seq_tensor, summed_indx]

        # get baseline
        if use_baseline:
            z_sample2 = sample_class_weights(class_weights)
            baseline = conditional_loss_fun(z_sample2)

        else:
            baseline = 0.0

        # compute gradient estimate
        reinforce_grad_sample = \
                get_reinforce_grad_sample(conditional_loss_fun_i,
                                log_class_weights_i, baseline)

        # sum
        summed_term = summed_term + \
                        ((reinforce_grad_sample + conditional_loss_fun_i) * \
                        class_weights[seq_tensor, summed_indx].squeeze()).sum()

    ############################
    # compute sampled term
    sampled_weight = torch.sum(class_weights * (1 - concentrated_mask), dim = 1,
                                keepdim = True)

    if not(topk == class_weights.shape[1]): # i.e. if we didn't sum everything
        # for numerical issues:
        class_weights_ = class_weights * (1 - concentrated_mask) + 1e-8
        sampled_weight_ = torch.sum(class_weights_, dim = 1, keepdim = True)

        # class weights conditioned on being in the diffuse set
        conditional_class_weights = class_weights_ / sampled_weight_
        assert not np.all(np.isnan(conditional_class_weights))

        # sample from conditional distribution
        conditional_z_sample = sample_class_weights(conditional_class_weights)

        # just for my own sanity ... check we actually sampled from diffuse set
        assert np.all((1 - concentrated_mask)[seq_tensor, conditional_z_sample].cpu().numpy() == 1.), \
                    'sampled_weight {}'.format(sampled_weight)

        # compute loss from sampled category
        conditional_loss_fun_i_sample = conditional_loss_fun(conditional_z_sample)
        assert len(conditional_loss_fun_i_sample) == log_class_weights.shape[0]

        # get log q
        log_class_weights_i_sample = \
            log_class_weights[seq_tensor, conditional_z_sample]

        if use_baseline:
            z_sample2 = sample_class_weights(class_weights)
            baseline2 = conditional_loss_fun(z_sample2)
        else:
            baseline2 = 0.0

        sampled_term = get_reinforce_grad_sample(\
                                    conditional_loss_fun_i_sample,
                                    log_class_weights_i_sample,
                                    baseline2) + \
                                    conditional_loss_fun_i_sample

    else:
        sampled_term = 0.

    return (sampled_term * sampled_weight.squeeze()).sum() + summed_term
