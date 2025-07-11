import torch
import torch.nn as nn
from torch.func import vmap, grad
from typing import Dict, Any, Tuple
import logging

log = logging.getLogger(__name__)

def get_graph_scores(z: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    u, v = z[..., 0], z[..., 1]
    raw_scores = hparams["alpha"] * (u @ v.T)
    d = z.shape[0]
    diag_mask = 1.0 - torch.eye(d, device=z.device, dtype=z.dtype)
    return raw_scores * diag_mask

def log_gaussian_likelihood(x: torch.Tensor, pred_mean: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma_tensor = torch.tensor(sigma, dtype=pred_mean.dtype, device=pred_mean.device)
    residuals = x - pred_mean
    log_prob = -0.5 * torch.log(2 * torch.pi * sigma_tensor**2) - 0.5 * ((residuals / sigma_tensor)**2)
    return torch.sum(log_prob)

def log_likelihood_given_g_and_theta(data: Dict[str, Any], g: torch.Tensor, theta: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    x_data = data['x']
    effective_W = theta * g
    pred_mean = torch.matmul(x_data, effective_W)
    return log_gaussian_likelihood(x_data, pred_mean, sigma=hparams['sigma_obs_noise'])

def get_gumbel_soft_gmat(z: torch.Tensor, hparams: Dict[str, Any], noise: torch.Tensor) -> torch.Tensor:
    scores = get_graph_scores(z, hparams)
    logits = (scores + noise) / hparams["tau"]
    g_soft = torch.sigmoid(logits)
    d = g_soft.size(-1)
    mask = 1.0 - torch.eye(d, device=z.device, dtype=z.dtype)
    return g_soft * mask


def log_prior_theta(g: torch.Tensor, theta: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    return log_gaussian_likelihood(theta * g, torch.zeros_like(theta), sigma=hparams['theta_prior_sigma'])

def log_prior_z_grad(z: torch.Tensor) -> torch.Tensor:
    return -(1 / torch.sqrt(torch.tensor(z.shape[0]))) * z

def acyclic_constr(g: torch.Tensor) -> torch.Tensor:
    d = g.shape[0]
    alpha = 1.0 / d
    eye = torch.eye(d, device=g.device, dtype=g.dtype)
    m = eye + alpha * g
    return torch.trace(torch.linalg.matrix_power(m, d)) - d

def stable_expectation(grad_samples: torch.Tensor, log_density_samples: torch.Tensor) -> torch.Tensor:
    # Mimicking JAX's stable grad computation with logsumexp and signs
    # Assumes grad_samples and log_density_samples are over MC dimension 0
    n_samples = grad_samples.shape[0]
    
    # Broadcast log_density to match grad dims
    while log_density_samples.dim() < grad_samples.dim():
        log_density_samples = log_density_samples.unsqueeze(-1)
    
    lse_den = torch.logsumexp(log_density_samples, dim=0) - torch.log(torch.tensor(n_samples, dtype=torch.float32, device=grad_samples.device))
    
    pos_grads = torch.where(grad_samples >= 0, grad_samples, torch.tensor(0.0, device=grad_samples.device))
    neg_grads = torch.where(grad_samples < 0, -grad_samples, torch.tensor(0.0, device=grad_samples.device))
    
    lse_num_pos = torch.logsumexp(log_density_samples + torch.log(pos_grads + 1e-30), dim=0) - torch.log(torch.tensor(n_samples, dtype=torch.float32, device=grad_samples.device))
    lse_num_neg = torch.logsumexp(log_density_samples + torch.log(neg_grads + 1e-30), dim=0) - torch.log(torch.tensor(n_samples, dtype=torch.float32, device=grad_samples.device))
    
    return torch.exp(lse_num_pos - lse_den) - torch.exp(lse_num_neg - lse_den)

def compute_reparam_z_gradient(params: Dict[str, nn.Parameter], data: Dict[str, Any], hparams: Dict[str, Any]) -> torch.Tensor:
    z = params['z']
    theta = params['theta']
    d = z.shape[0]
    n_mc = hparams['n_mc_samples']
    beta = hparams['beta']
    sigma_z = hparams['sigma_z']  # Assume this is in hparams, e.g., 1/sqrt(d)
    
    # Generate MC Gumbel noises: log(u) - log(1-u)
    u = torch.rand(n_mc, d, d, device=z.device)
    noises = torch.log(u) - torch.log1p(-u)
    
    # 1. Prior grad: - (1/sigma_z^2) * z - beta * grad E[h(G)]
    grad_z_prior = - (1 / sigma_z**2) * z
    
    def per_sample_h(noise):
        g_soft = get_gumbel_soft_gmat(z, hparams, noise)
        return acyclic_constr(g_soft)
    
    grad_h_fn = grad(per_sample_h)
    grad_h_samples = vmap(grad_h_fn)(noises)
    grad_acyclic = stable_expectation(grad_h_samples, torch.zeros(n_mc, device=z.device))  # Denom is 1 since uniform noise
    grad_z_prior -= beta * grad_acyclic
    
    # 2. Likelihood term: grad_Z E[p(Θ,D|G)] / E[p(Θ,D|G)]
    def per_sample_log_p_theta_D_given_G(noise):
        g_soft = get_gumbel_soft_gmat(z, hparams, noise)
        return log_likelihood_given_g_and_theta(data, g_soft, theta, hparams) + log_prior_theta(g_soft, theta, hparams)
    
    log_p_samples = vmap(per_sample_log_p_theta_D_given_G)(noises)
    grad_p_fn = grad(per_sample_log_p_theta_D_given_G)
    grad_p_samples = vmap(grad_p_fn)(noises)
    grad_lik = stable_expectation(grad_p_samples, log_p_samples)
    
    return grad_z_prior + grad_lik

def compute_reparam_theta_gradient(params: Dict[str, nn.Parameter], data: Dict[str, Any], hparams: Dict[str, Any]) -> torch.Tensor:
    z = params['z']
    theta = params['theta']
    d = z.shape[0]
    n_mc = hparams['n_mc_samples']
    
    # Generate MC Gumbel noises
    u = torch.rand(n_mc, d, d, device=z.device)
    noises = torch.log(u) - torch.log1p(-u)
    
    # E[grad_Θ p(Θ,D|G)] / E[p(Θ,D|G)]
    def per_sample_log_p_theta_D_given_G(noise):
        g_soft = get_gumbel_soft_gmat(z, hparams, noise)
        return log_likelihood_given_g_and_theta(data, g_soft, theta, hparams) + log_prior_theta(g_soft, theta, hparams)
    
    log_p_samples = vmap(per_sample_log_p_theta_D_given_G)(noises)
    grad_p_fn = grad(per_sample_log_p_theta_D_given_G)
    grad_p_samples = vmap(grad_p_fn)(noises)
    return stable_expectation(grad_p_samples, log_p_samples)

# Example usage (adapt to your trainer):
# grad_z = compute_reparam_z_gradient(params, data, hparams)
# grad_theta = compute_reparam_theta_gradient(params, data, hparams) 