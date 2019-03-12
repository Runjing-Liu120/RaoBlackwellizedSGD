import numpy as np

import torch

from common_utils import get_one_hot_encoding_from_int
from baselines_lib import sample_class_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_concentrated_mask(class_weights, topk):
    """
    Returns a logical mask indicating the categories with the top k largest
    probabilities, as well as the catogories corresponding to those with the
    top k largest probabilities.

    Parameters
    ----------
    class_weights : torch.Tensor
        Array of class weights, with each row corresponding to a datapoint,
        each column corresponding to its weight
    topk : int
        the k in top-k
    
    Returns
    -------
    mask_topk : torch.Tensor
        Boolean array with entry 1 if the corresponding class weight is
        in the topk for that observation
    topk_domain: torch.LongTensor
        Array specifying the indices of class_weights that correspond to
        the topk observations
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

    Parameters
    ----------
    conditional_loss_fun : function
        Function that takes input a one-hot encoding of the discrete random
        variable z and outputs the loss conditional on z
    class_weights : torch.Tensor
        Array of class weights, with each row corresponding to a datapoint,
        each column corresponding to its weight
    Returns
    -------
    full_loss : float
        The loss averaged over the class weights of the discrete random variable
    """

    full_loss = 0.0

    for i in range(class_weights.shape[1]):

        i_rep = (torch.ones(class_weights.shape[0]) * i).type(torch.LongTensor)
        one_hot_i = get_one_hot_encoding_from_int(i_rep,
                        class_weights.shape[1])

        conditional_loss = conditional_loss_fun(one_hot_i)
        assert len(conditional_loss) == class_weights.shape[0]

        full_loss = full_loss + class_weights[:, i] * conditional_loss

    return full_loss.sum()

def get_raoblackwell_ps_loss(conditional_loss_fun, log_class_weights, topk,
                        grad_estimator,
                        grad_estimator_kwargs = {'grad_estimator_kwargs': None},
                        epoch = None,
                        data = None):

    """
    Returns a pseudo_loss, such that the gradient obtained by calling
    pseudo_loss.backwards() is unbiased for the true loss

    Parameters
    ----------
    conditional_loss_fun : function
        A function that returns the loss conditional on an instance of the
        categorical random variable. It must take in a one-hot-encoding
        matrix (batchsize x n_categories) and return a vector of
        losses, one for each observation in the batch.
    log_class_weights : torch.Tensor
        A tensor of shape batchsize x n_categories of the log class weights
    topk : Integer
        The number of categories to sum over
    grad_estimator : function
        A function that returns the pseudo loss, that is, the loss which
        gives a gradient estimator when .backwards() is called.
        See baselines_lib for details.
    grad_estimator_kwargs : dict
        keyword arguments to gradient estimator
    """

    # class weights from the variational distribution
    assert np.all(log_class_weights.detach().cpu().numpy() <= 0)
    class_weights = torch.exp(log_class_weights.detach())

    # this is the indicator C_k
    concentrated_mask, topk_domain, seq_tensor = \
        get_concentrated_mask(class_weights, topk)
    concentrated_mask = concentrated_mask.float().detach()

    ############################
    # compute the summed term
    summed_term = 0.0

    for i in range(topk):
        # get categories to be summed
        summed_indx = topk_domain[:, i]

        # compute gradient estimate
        grad_summed = \
                grad_estimator(conditional_loss_fun, log_class_weights,
                                class_weights, seq_tensor, \
                                z_sample = summed_indx,
                                epoch = epoch,
                                data = data,
                                **grad_estimator_kwargs)

        # sum
        summed_weights = class_weights[seq_tensor, summed_indx].squeeze()
        summed_term = summed_term + \
                        (grad_summed * summed_weights).sum()

    ############################
    # compute sampled term
    sampled_weight = torch.sum(class_weights * (1 - concentrated_mask), dim = 1,
                                keepdim = True)

    if not(topk == class_weights.shape[1]):
        # if we didn't sum everything
        # we sample the remaining terms

        # class weights conditioned on being in the diffuse set
        conditional_class_weights = (class_weights + 1e-12) * \
                    (1 - concentrated_mask)  / (sampled_weight + 1e-12)
        # assert not np.any(np.isnan(conditional_class_weights))

        # sample from conditional distribution
        conditional_z_sample = sample_class_weights(conditional_class_weights)

        # just for my own sanity ... check we actually sampled from diffuse set
        assert np.all((1 - concentrated_mask)[seq_tensor, conditional_z_sample].cpu().numpy() == 1.), \
                    'sampled_weight {}'.format(sampled_weight)

        grad_sampled = grad_estimator(conditional_loss_fun, log_class_weights,
                                class_weights, seq_tensor,
                                z_sample = conditional_z_sample,
                                epoch = epoch,
                                data = data,
                                **grad_estimator_kwargs)

    else:
        grad_sampled = 0.

    return (grad_sampled * sampled_weight.squeeze()).sum() + summed_term
