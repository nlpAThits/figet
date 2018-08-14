#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer, required


class Optim(object):

    def __init__(self, learning_rate, max_grad_norm):
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

    def set_parameters(self, params):
        self.params = list(params)
        self.optimizer = optim.Adam(self.params, lr=self.learning_rate)

    def step(self):
        if self.max_grad_norm != -1:    # -1 by default
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict())


spten_t = torch.sparse.FloatTensor


def poincare_grad(p, d_p):
    r"""
    Function to compute Riemannian gradient from the
    Euclidean gradient in the Poincaré ball.

    Args:
        p (Tensor): Current point in the ball
        d_p (Tensor): Euclidean gradient at p
    """
    if d_p.is_sparse:
        p_sqnorm = torch.sum(
            p.data[d_p._indices()[0].squeeze()] ** 2, dim=1,
            keepdim=True
        ).expand_as(d_p._values())
        n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
        d_p = spten_t(d_p._indices(), n_vals, d_p.size())
    else:
        p_sqnorm = torch.sum(p.data ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
    return d_p


def euclidean_grad(p, d_p):
    return d_p


def euclidean_retraction(p, d_p, lr):
    p.data.add_(-lr, d_p)


class RiemannianSGD(Optimizer):
    r"""Riemannian stochastic gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rgrad (Function): Function to compute the Riemannian gradient from
            an Euclidean gradient
        retraction (Function): Function to update the parameters via a
            retraction of the Riemannian gradient
        lr (float): learning rate
    """

    def __init__(self, params, lr=required, rgrad=required, retraction=required):
        defaults = dict(lr=lr, rgrad=rgrad, retraction=retraction)
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None):
        """Performs a single optimization step.

        Arguments:
            lr (float, optional): learning rate for the current update.
        """
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if lr is None:
                    lr = group['lr']
                d_p = group['rgrad'](p, d_p)
                group['retraction'](p, d_p, lr)

        return loss
