import numpy as np

import torch

import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_SGD(get_loss, params,
                lr = 1.0, n_steps = 10000,
                get_full_loss = None,
                **kwargs):

    # get_loss should be a function that returns a ps_loss such that
    # ps_loss.backward() returns an unbiased estimate of the gradient.
    # in general, ps_loss might not equal the actual loss.

    # set up optimizer
    params_list = [{'params': params[key]} for key in params]
    optimizer = optim.SGD(params_list, lr = lr)

    loss_array = np.zeros(n_steps)

    for i in range(n_steps):
        # run gradient descent
        optimizer.zero_grad()

        loss = get_loss(**kwargs)
        loss.backward()
        optimizer.step()

        # save losses
        if get_full_loss is not None:
            full_loss = get_full_loss()
        else:
            full_loss = loss

        loss_array[i] = full_loss.detach().numpy()

    opt_params = params

    return loss_array, opt_params
