#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from torch.autograd import Function
from abc import abstractmethod


class Acosh(Function):
    @staticmethod
    def forward(ctx, x, eps):
        z = th.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        ctx.eps = eps
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = th.clamp(z, min=ctx.eps)
        z = g / z
        return z, None


acosh = Acosh.apply


class Manifold(object):
    def __init__(self, *args, **kwargs):
        pass

    def init_weights(self, w, scale=1e-4):
        w.data.uniform_(-scale, scale)

    @staticmethod
    def dim(dim):
        return dim

    def normalize(self, u):
        return u

    @abstractmethod
    def distance(self, u, v):
        """
        Distance function
        """
        raise NotImplementedError

    @abstractmethod
    def expm(self, p, d_p, lr=None, out=None):
        """
        Exponential map
        """
        raise NotImplementedError

    @abstractmethod
    def logm(self, x, y):
        """
        Logarithmic map
        """
        raise NotImplementedError

    @abstractmethod
    def ptransp(self, x, y, v, ix=None, out=None):
        """
        Parallel transport
        """
        raise NotImplementedError


class EuclideanManifold(Manifold):
    __slots__ = ["max_norm"]

    def __init__(self, max_norm=1, **kwargs):
        self.max_norm = max_norm

    def normalize(self, u):
        d = u.size(-1)
        u.view(-1, d).renorm_(2, 0, self.max_norm)
        return u

    def distance(self, u, v):
        return (u - v).pow(2).sum(dim=-1)

    def pnorm(self, u, dim=-1):
        return (u * u).sum(dim=dim).sqrt()

    def rgrad(self, p, d_p):
        return d_p

    def expm(self, p, d_p, normalize=False, lr=None, out=None):
        if lr is not None:
            d_p.mul_(-lr)
        if out is None:
            out = p
        out.add_(d_p)
        if normalize:
            self.normalize(out)
        return out

    def logm(self, p, d_p, out=None):
        return p - d_p

    def ptransp(self, p, x, y, v):
        ix, v_ = v._indices().squeeze(), v._values()
        return p.index_copy_(0, ix, v_)


class PoincareManifold(EuclideanManifold):
    def __init__(self, eps=1e-5, **kwargs):
        super(PoincareManifold, self).__init__(**kwargs)
        self.eps = eps
        self.boundary = 1 - eps
        self.max_norm = self.boundary

    def distance(self, u, v):
        return Distance.apply(u, v, self.eps)

    def rgrad(self, p, d_p):
        if d_p.is_sparse:
            p_sqnorm = th.sum(
                p[d_p._indices()[0].squeeze()] ** 2, dim=1,
                keepdim=True
            ).expand_as(d_p._values())
            n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
            n_vals.renorm_(2, 0, 5)
            d_p = th.sparse.DoubleTensor(d_p._indices(), n_vals, d_p.size())
        else:
            p_sqnorm = th.sum(p ** 2, dim=-1, keepdim=True)
            d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
        return d_p


class Distance(Function):
    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist, eps):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * th.sum(x * v, dim=-1) + 1) / th.pow(alpha, 2))\
            .unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = th.sqrt(th.pow(z, 2) - 1)
        z = th.clamp(z * beta, min=eps).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

    @staticmethod
    def forward(ctx, u, v, eps):
        squnorm = th.clamp(th.sum(u * u, dim=-1), 0, 1 - eps)
        sqvnorm = th.clamp(th.sum(v * v, dim=-1), 0, 1 - eps)
        sqdist = th.sum(th.pow(u - v, 2), dim=-1)
        ctx.eps = eps
        ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        # arcosh
        z = th.sqrt(th.pow(x, 2) - 1)
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = Distance.grad(u, v, squnorm, sqvnorm, sqdist, ctx.eps)
        gv = Distance.grad(v, u, sqvnorm, squnorm, sqdist, ctx.eps)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv, None


class LorentzManifold(Manifold):
    __slots__ = ["eps", "_eps", "norm_clip", "max_norm", "debug"]

    @staticmethod
    def dim(dim):
        return dim + 1

    def __init__(self, eps=1e-5, _eps=1e-5, norm_clip=1, max_norm=1e6, debug=False, **kwargs):
        self.eps = eps
        self._eps = _eps
        self.norm_clip = norm_clip
        self.max_norm = max_norm
        self.debug = debug

    @staticmethod
    def ldot(u, v, keepdim=False):
        """Lorentzian Scalar Product"""
        uv = u * v
        uv.narrow(-1, 0, 1).mul_(-1)
        return th.sum(uv, dim=-1, keepdim=keepdim)

    def to_poincare_ball(self, u):
        x = u.clone()
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)

    def distance(self, u, v):
        d = -LorentzDot.apply(u, v)
        d.data.clamp_(min=1)
        return acosh(d, self._eps)

    def pnorm(self, u):
        return th.sqrt(th.sum(th.pow(self.to_poincare_ball(u), 2), dim=-1))

    def normalize(self, w):
        """Normalize vector such that it is located on the hyperboloid"""
        d = w.size(-1) - 1
        narrowed = w.narrow(-1, 1, d)
        if self.max_norm:
            narrowed.view(-1, d).renorm_(p=2, dim=0, maxnorm=self.max_norm)
        tmp = 1 + th.sum(th.pow(narrowed, 2), dim=-1, keepdim=True)
        tmp.sqrt_()
        w.narrow(-1, 0, 1).copy_(tmp)
        return w

    def normalize_tan(self, x_all, v_all):
        d = v_all.size(1) - 1
        x = x_all.narrow(1, 1, d)
        xv = th.sum(x * v_all.narrow(1, 1, d), dim=1, keepdim=True)
        tmp = 1 + th.sum(th.pow(x_all.narrow(1, 1, d), 2), dim=1, keepdim=True)
        tmp.sqrt_().clamp_(min=self._eps)
        v_all.narrow(1, 0, 1).copy_(xv / tmp)
        return v_all

    def init_weights(self, w, irange=1e-5):
        w.data.uniform_(-irange, irange)
        w.data.copy_(self.normalize(w.data))

    def rgrad(self, p, d_p):
        """Riemannian gradient for hyperboloid"""
        if d_p.is_sparse:
            u = d_p._values()
            x = p.index_select(0, d_p._indices().squeeze())
        else:
            u = d_p
            x = p
        u.narrow(-1, 0, 1).mul_(-1)
        u.addcmul_(self.ldot(x, u, keepdim=True).expand_as(x), x)
        return d_p

    def expm(self, p, d_p, lr=None, out=None, normalize=False):
        """Exponential map for hyperboloid"""
        if out is None:
            out = p
        if d_p.is_sparse:
            ix, d_val = d_p._indices().squeeze(), d_p._values()
            # This pulls `ix` out of the original embedding table, which could
            # be in a corrupted state.  normalize it to fix it back to the
            # surface of the hyperboloid...
            # TODO: we should only do the normalize if we know that we are
            # training with multiple threads, otherwise this is a bit wasteful
            p_val = self.normalize(p.index_select(0, ix))
            ldv = self.ldot(d_val, d_val, keepdim=True)
            if self.debug:
                assert all(ldv > 0), "Tangent norm must be greater 0"
                assert all(ldv == ldv), "Tangent norm includes NaNs"
            nd_p = ldv.clamp_(min=0).sqrt_()
            t = th.clamp(nd_p, max=self.norm_clip)
            nd_p.clamp_(min=self.eps)
            newp = (th.cosh(t) * p_val).addcdiv_(th.sinh(t) * d_val, nd_p)
            if normalize:
                newp = self.normalize(newp)
            p.index_copy_(0, ix, newp)
        else:
            if lr is not None:
                d_p.narrow(-1, 0, 1).mul_(-1)
                d_p.addcmul_((self.ldot(p, d_p, keepdim=True)).expand_as(p), p)
                d_p.mul_(-lr)
            ldv = self.ldot(d_p, d_p, keepdim=True)
            if self.debug:
                assert all(ldv > 0), "Tangent norm must be greater 0"
                assert all(ldv == ldv), "Tangent norm includes NaNs"
            nd_p = ldv.clamp_(min=0).sqrt_()
            t = th.clamp(nd_p, max=self.norm_clip)
            nd_p.clamp_(min=self.eps)
            newp = (th.cosh(t) * p).addcdiv_(th.sinh(t) * d_p, nd_p)
            if normalize:
                newp = self.normalize(newp)
            p.copy_(newp)

    def logm(self, x, y):
        """Logarithmic map on the Lorenz Manifold"""
        xy = th.clamp(self.ldot(x, y).unsqueeze(-1), max=-1)
        v = acosh(-xy, self.eps).div_(
            th.clamp(th.sqrt(xy * xy - 1), min=self._eps)
        ) * th.addcmul(y, xy, x)
        return self.normalize_tan(x, v)

    def ptransp(self, x, y, v, ix=None, out=None):
        """Parallel transport for hyperboloid"""
        if ix is not None:
            v_ = v
            x_ = x.index_select(0, ix)
            y_ = y.index_select(0, ix)
        elif v.is_sparse:
            ix, v_ = v._indices().squeeze(), v._values()
            x_ = x.index_select(0, ix)
            y_ = y.index_select(0, ix)
        else:
            raise NotImplementedError
        xy = self.ldot(x_, y_, keepdim=True).expand_as(x_)
        vy = self.ldot(v_, y_, keepdim=True).expand_as(x_)
        vnew = v_ + vy / (1 - xy) * (x_ + y_)
        if out is None:
            return vnew
        else:
            out.index_copy_(0, ix, vnew)


class LorentzDot(Function):
    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        return LorentzManifold.ldot(u, v)

    @staticmethod
    def backward(ctx, g):
        u, v = ctx.saved_tensors
        g = g.unsqueeze(-1).expand_as(u).clone()
        g.narrow(-1, 0, 1).mul_(-1)
        return g * v, g * u


def rsgd_params(manifold, parameters):
    return [{
        'params': parameters,
        'rgrad': manifold.rgrad,
        'expm': manifold.expm,
        'logm': manifold.logm,
        'ptransp': manifold.ptransp,
    }, ]
