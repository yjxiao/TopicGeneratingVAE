import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import LogNormal
from torch.distributions import kl_divergence

import numpy as np


def seq_recon_loss(outputs, targets, pad_id):
    return F.cross_entropy(
        outputs.view(-1, outputs.size(2)),
        targets.view(-1),
        size_average=False, ignore_index=pad_id)


def bow_recon_loss(outputs, targets):
    """
    Note that outputs is the bag-of-words log likelihood predictions. 
    targets is the target counts. 

    """
    return - torch.sum(targets * outputs)


def standard_prior_like(posterior):
    loc = torch.zeros_like(posterior.loc)
    scale = torch.ones_like(posterior.scale)
    Dist = type(posterior)
    return Dist(loc, scale)


def total_kld(posterior, prior=None):
    if prior is None:
        prior = standard_prior_like(posterior)
    return torch.sum(kl_divergence(posterior, prior))
