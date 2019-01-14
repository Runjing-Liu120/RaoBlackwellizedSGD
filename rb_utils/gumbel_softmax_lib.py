import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sample_gumbel(shape, eps=1e-20):
    # samples from gumbel distribution

    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)

    return (y_hard - y).detach() + y

# TODO: test this function
# used for REBAR, see https://arxiv.org/pdf/1703.07370.pdf
def gumbel_softmax_conditional_sample(logits, temperature, one_hot_z):
    U = torch.rand(logits.shape).to(device)
    log_U = torch.log(U + eps)
    log_U_k = (one_hot_z * log_U).sum(dim = -1, keepdim = True)

    gumbel_conditional_sample = \
        -torch.log(-log_U_k + \
                -log_U / (torch.exp(logits) + 1e-12) * (1 - one_hot_z))

    return F.softmax(gumbel_conditional_sample / temperature, dim=-1)
