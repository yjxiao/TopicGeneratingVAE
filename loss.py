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


def e_log_p(posterior, prior):
    """
    Calculate the expected value of prior under the posterior. 
    Only works for Normal and LogNormal
    D(q||p) = - H(q) - e_log_p(q, p)

    """
    batch_size = prior.loc.size(0)
    comp1 = - prior.scale.log() - 0.5 * np.log(2 * np.pi)
    comp2 = - 0.5 * ((posterior.loc - prior.loc) / prior.scale).pow(2)
    comp3 = - 0.5 * (posterior.scale / prior.scale).pow(2)
    if isinstance(posterior, LogNormal):
        comp4 = - posterior.loc
    elif isinstance(posterior, Normal):
        comp4 = 0
    else:
        raise NotImplementedError("function not implemented for type: ", type(posterior))
    return torch.sum(comp1 + comp2 + comp3 + comp4) / batch_size


def kld_decomp(posterior_z, z, posterior_t, t,
               prior_z=None, prior_t=None):
    batch_size = z.size(0)
    code_size = z.size(1)
    num_topics = t.size(1)
    if prior_z is None:
        prior_z = standard_prior_like(posterior_z)
    if prior_t is None:
        prior_t = standard_prior_like(posterior_t)
    e_log_qzx = - torch.sum(posterior_z.entropy().sum(1)) / batch_size
    e_log_qtx = - torch.sum(posterior_t.entropy().sum(1)) / batch_size
    
    # approximating using minibatches
    log_probs_z = posterior_z.log_prob(
        z.unsqueeze(1).expand(-1, batch_size, -1)
    )
    log_probs_t = posterior_t.log_prob(
        t.unsqueeze(1).expand(-1, batch_size, -1)
    )
    e_log_qzt = torch.sum(
        (log_probs_z.sum(2) + log_probs_t.sum(2)).exp().sum(1).log()
    ) / batch_size - np.log(batch_size)
    e_log_qz = torch.sum(
        log_probs_z.sum(2).exp().sum(1).log()
    ) / batch_size - np.log(batch_size)
    e_log_qt = torch.sum(
        log_probs_t.sum(2).exp().sum(1).log()
    ) / batch_size - np.log(batch_size)
    e_log_pz = e_log_p(posterior_z, prior_z)
    e_log_pt = e_log_p(posterior_t, prior_t)

    return (e_log_qzx, e_log_qtx, e_log_qzt,
            e_log_qz, e_log_qt, e_log_pz, e_log_pt)
    
