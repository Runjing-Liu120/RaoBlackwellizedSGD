import torch

from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_kl_q_standard_normal(mu, log_sigma):
    # The KL between a Gaussian variational distribution
    # and a standard normal
    mu = mu.view(mu.shape[0], -1)
    log_sigma = log_sigma.view(log_sigma.shape[0], -1)
    return 0.5 * torch.sum(-1 - 2 * log_sigma + \
                                mu**2 + torch.exp(log_sigma)**2, dim = 1)

def get_bernoulli_loglik(pi, x):
    loglik = (x * torch.log(pi + 1e-8) + (1 - x) * torch.log(1 - pi + 1e-8))
    return loglik.view(x.size(0), -1).sum(1)

def get_multinomial_kl(probs):
    # the kl between a categorical distribution with probability probs
    # and a uniform distribution
    return (torch.log(probs + 1e-8) * probs).sum(1)
