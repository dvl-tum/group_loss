import torch.nn as nn
import torch
import dynamics
import torch.nn.functional as F


class GTG(nn.Module):
    def __init__(self, total_classes, tol=-1., max_iter=5, mode='replicator', device='cuda:0'):
        super(GTG, self).__init__()
        self.m = total_classes
        self.tol = tol
        self.max_iter = max_iter
        self.mode = mode
        self.device = device

    def _init_probs_uniform(self, labs, L, U):
        """ Initialized the probabilities of GTG from uniform distribution """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = 1. / self.m
        ps[L, labs] = 1.

        # check if probs sum up to 1.
        assert torch.allclose(ps.sum(dim=1), torch.ones(n).cuda())
        return ps

    def _init_probs_prior(self, probs, labs, L, U):
        """ Initiallized probabilities from the softmax layer of the CNN """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = probs[U, :]
        ps[L, labs] = 1.
        
        # check if probs sum up to 1.
        assert torch.allclose(ps.sum(dim=1), torch.ones(n).cuda())
        return ps

    def _init_probs_prior_only_classes(self, probs, labs, L, U, classes_to_use):
        """ Different version of the previous version when it considers only classes in the minibatch,
            might need tuning in order to reach the same performance as _init_probs_prior """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = probs[torch.meshgrid(torch.tensor(U), torch.from_numpy(classes_to_use))]
        ps[L, labs] = 1.
        ps /= ps.sum(dim=ps.dim() - 1).unsqueeze(ps.dim() - 1)
        return ps

    def set_negative_to_zero(self, W):
        return F.relu(W)

    def _get_W(self, x):

        x = (x - x.mean(dim=1).unsqueeze(1))
        norms = x.norm(dim=1)
        W = torch.mm(x, x.t()) / torch.ger(norms, norms)

        W = self.set_negative_to_zero(W.cuda())
        return W

    def forward(self, fc7, num_points, labs, L, U, probs=None, classes_to_use=None):
        W = self._get_W(fc7)
        if type(probs) is type(None):
            ps = self._init_probs_uniform(labs, L, U)
        else:
            if type(classes_to_use) is type(None):
                ps = probs
                ps = self._init_probs_prior(ps, labs, L, U)
            else:
                ps = probs
                ps = self._init_probs_prior_only_classes(ps, labs, L, U, classes_to_use)
        ps = dynamics.dynamics(W, ps, self.tol, self.max_iter, self.mode)
        return ps, W
