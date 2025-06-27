import torch
from torch.distributions import Normal
import numpy as np
import logging
from typing import Dict, Any, Tuple


def acyclic_constr(g: torch.Tensor, d: int) -> torch.Tensor:
    alpha = 1.0 / d
    eye = torch.eye(d, device=g.device, dtype=g.dtype)
    m = eye + alpha * g

    if d <= 10:
        return torch.trace(torch.linalg.matrix_power(m, d)) - d

def log_gaussian_likelihood(x: torch.Tensor, pred_mean: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    gaussian_dist = Normal(loc=pred_mean, scale=sigma)
    log_prob = gaussian_dist.log_prob(x)
    return torch.sum(log_prob)

def scores(z: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
  """
    z shape (n, d, k 2) 
    returns raw scores 
  """
  u, v = z[..., 0], z[..., 1]
  raw_scores = hparams["alpha"] * torch.einsum('...ik,...jk->...ij', u, v)
  _, d, _ = z.shape[:-1]
  diag_mask = 1.0 - torch.eye(d, device=z.device, dtype=z.dtype)
  return raw_scores * diag_mask

def soft_gmat(z: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    """
    apply sigmoid to raw scores
    get the edge probability matrix
    return shape (d, d)
    """
    raw_scores = scores(z, hparams)
    edge_probs = torch.sigmoid(raw_scores)
    diag_mask = 1.0 - torch.eye(d, device=z.device, dtype=z.dtype)
    return edge_probs * diag_mask

def bernoulli_sample(g_soft: torch.Tensor) -> torch.Tensor:
    """
    sample from bernoulli distribution
    return shape (d, d) hard graph
    """
    return torch.bernoulli(g_soft)

def gumbel_softmax_sample(g_soft: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    skip for now
    """
    return "skip for now"

def log_full_likelihood(data: Dict[str, Any], g_soft: torch.Tensor, theta: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    """
    calculate the log full likelihood
    expert belief: update this to use interventions, change the full likelihood 
    """
    x_data = data['x']
    effective_W = theta * g_soft
    pred_mean = torch.matmul(x_data, effective_W)
    sigma_obs = hparams.get('sigma_obs_noise', 0.1)
    return log_gaussian_likelihood(x_data, pred_mean, sigma=sigma_obs)

def log_theta_prior(theta_effective: torch.Tensor, sigma: float) -> torch.Tensor:
    return log_gaussian_likelihood(theta_effective, torch.zeros_like(theta_effective), sigma=sigma)

def gumbel_acyclic_constr_mc(z: torch.Tensor, d: int, hparams: Dict[str, Any]) -> torch.Tensor:
    """
    implement the gumbel_acyclic_constr_mc later
    for now use REINFORCE estimator
    """
    return "skip for now"


## TODO CHECK THE CODE
def score_acyclic_constr_mc(z: torch.Tensor, d: int, hparams: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimates the expected acyclicity E[h(G)] using Monte Carlo sampling and prepares the terms needed for the score function estimator's surrogate loss.
    and prepares the terms needed for the score function estimator's surrogate loss.
    
    Returns:
        A tuple containing:
        - h_g (torch.Tensor): The acyclicity values for each sample [n_samples].
        - log_prob_g (torch.Tensor): The log probability of each sample [n_samples].
    """
    n_samples = hparams.get('n_mc_samples', 64)
    
    # 1. Sample hard graphs
    g_samples = bernoulli_sample(z, hparams, n_samples=n_samples)
    
    # 2. Calculate the "reward" h(G) for each sample
    # We don't detach here; the calling function will detach it when creating the surrogate loss.
    h_g = torch.vmap(lambda g: acyclic_constr(g, d))(g_samples)
    
    # 3. Calculate the log probability of each sample
    log_p, log_1_p = edge_log_probs(z, hparams)
    log_prob_g = log_prob_graph(g_samples, log_p, log_1_p)
    
    return h_g, log_prob_g
