import torch
import rao_blackwellization_lib as rb_lib

from copy import deepcopy

class ToyExperiment(object):
    def __init__(self, eta, p0):
        # number of categories
        self.k = len(p0)
        self.seq_tensor = torch.LongTensor([[i for i in range(self.k)]]).float()

        self.p0 = p0

        # the parameter
        self.set_parameter(eta)

        self.log_softmax = torch.nn.LogSoftmax(dim = 0)

    def set_parameter(self, eta):
        self.eta = deepcopy(eta)
        self.eta.requires_grad_(True)

    def get_log_q(self):
        return self.log_softmax(self.eta * self.p0).view(1, self.k)

    def get_f_z(self, z):
        return ((z * self.seq_tensor).sum() - self.eta) ** 2

    def get_pm_loss(self, topk, grad_estimator,
                    grad_estimator_kwargs = {'grad_estimator_kwargs': None}):
        data = torch.rand((1, 5, 5))
        log_class_weights = self.get_log_q()
        return rb_lib.get_raoblackwell_ps_loss(self.get_f_z, log_class_weights, topk,
                                grad_estimator,
                                grad_estimator_kwargs = grad_estimator_kwargs,
                                data = data)

    def get_full_loss(self):
        log_class_weights = self.get_log_q()
        class_weights = torch.exp(log_class_weights)

        return rb_lib.get_full_loss(self.get_f_z, class_weights)
