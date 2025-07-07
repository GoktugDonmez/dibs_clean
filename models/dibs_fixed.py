import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

log = logging.getLogger(__name__)

def acyclic_constr(g: torch.Tensor, d: int) -> torch.Tensor:
    """
    Acyclicity constraint h(G) from NOTEARS with numerical stability.
    Returns h(G) = tr((I + αG)^d) - d, where α = 1/d
    """
    alpha = 1.0 / d
    eye = torch.eye(d, device=g.device, dtype=g.dtype)
    m = eye + alpha * g
    
    if d <= 10:
        # Use matrix power for small graphs
        return torch.trace(torch.linalg.matrix_power(m, d)) - d
    else:
        # Use eigenvalues for larger graphs for numerical stability
        try:
            eigvals = torch.linalg.eigvals(m)
            return torch.sum(torch.real(eigvals ** d)) - d
        except RuntimeError:
            # Fallback to series expansion if eigenvalue computation fails
            trace = torch.tensor(0.0, device=g.device, dtype=g.dtype)
            p = g.clone()
            for k in range(1, min(d + 1, 20)):
                trace += (alpha ** k) * torch.trace(p) / k
                if k < 19:
                    p = p @ g
            return trace

def log_gaussian_likelihood(x: torch.Tensor, pred_mean: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """
    Compute log p(x | pred_mean, sigma) for Gaussian likelihood.
    """
    sigma_tensor = torch.tensor(sigma, dtype=pred_mean.dtype, device=pred_mean.device)
    residuals = x - pred_mean
    log_prob = -0.5 * (torch.log(2 * torch.pi * sigma_tensor**2)) - 0.5 * ((residuals / sigma_tensor)**2)
    return torch.sum(log_prob)

def scores(z: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Compute raw edge scores s_ij = α * u_i^T v_j
    Args:
        z: (d, k, 2) latent variables [u, v]
        alpha: scaling parameter
    Returns:
        (d, d) score matrix with diagonal masked
    """
    u, v = z[..., 0], z[..., 1]  # (d, k)
    raw_scores = alpha * torch.einsum('ik,jk->ij', u, v)  # (d, d)
    d = z.shape[0]
    diag_mask = 1.0 - torch.eye(d, device=z.device, dtype=z.dtype)
    return raw_scores * diag_mask

def soft_gmat(z: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Compute soft adjacency matrix G_ij = σ(s_ij)
    """
    raw_scores = scores(z, alpha)
    return torch.sigmoid(raw_scores)

def log_full_likelihood(data: Dict[str, Any], g: torch.Tensor, theta: torch.Tensor, sigma_obs: float) -> torch.Tensor:
    """
    Compute log p(X | G, Θ) for linear Gaussian SCM.
    """
    x_data = data['x']
    effective_W = theta * g
    pred_mean = torch.matmul(x_data, effective_W)
    return log_gaussian_likelihood(x_data, pred_mean, sigma=sigma_obs)

def log_theta_prior(theta_effective: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Compute log p(Θ_eff) = log N(Θ_eff | 0, σ²I)
    """
    return log_gaussian_likelihood(theta_effective, torch.zeros_like(theta_effective), sigma=sigma)

def stable_ratio_estimator(grad_samples: list, log_density_samples: list) -> torch.Tensor:
    """
    Numerically stable ratio estimator for gradients.
    Computes: (Σ w_i * |∇_i| * sign(∇_i)) where w_i = exp(log_p_i) / Σ exp(log_p_j)
    """
    eps = 1e-30
    
    # Stack samples
    log_p = torch.stack(log_density_samples)  # (M,)
    grads = torch.stack(grad_samples)  # (M, ...)
    
    # Expand log_p to match grads dimensions
    while log_p.dim() < grads.dim():
        log_p = log_p.unsqueeze(-1)
    
    # Compute stable softmax weights
    log_p_max = log_p.max(dim=0, keepdim=True)[0]
    log_p_shifted = log_p - log_p_max
    weights = torch.exp(log_p_shifted)
    weights = weights / (weights.sum(dim=0, keepdim=True) + eps)
    
    # Separate positive and negative gradients for stability
    pos_mask = grads >= 0
    neg_mask = grads < 0
    
    # Weighted sum for positive gradients
    pos_grads = torch.where(pos_mask, grads, torch.zeros_like(grads))
    pos_weighted = torch.sum(weights * pos_grads, dim=0)
    
    # Weighted sum for negative gradients
    neg_grads = torch.where(neg_mask, -grads, torch.zeros_like(grads))
    neg_weighted = torch.sum(weights * neg_grads, dim=0)
    
    return pos_weighted - neg_weighted

def score_g_given_z(z: torch.Tensor, g: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Analytic computation of ∇_z log q(G|z) for Bernoulli distribution.
    """
    g_soft = soft_gmat(z, alpha)
    diff = g - g_soft  # (g_ij - σ(s_ij))
    u, v = z[..., 0], z[..., 1]  # (d, k)
    
    # Gradients w.r.t. u and v
    grad_u = alpha * torch.einsum('ij,jk->ik', diff, v)  # (d, k)
    grad_v = alpha * torch.einsum('ij,ik->jk', diff, u)  # (d, k)
    
    return torch.stack([grad_u, grad_v], dim=-1)  # (d, k, 2)

def grad_z_log_joint(z: torch.Tensor, theta: torch.Tensor, data: Dict[str, Any], hparams: Dict[str, Any]) -> torch.Tensor:
    """
    Compute ∇_z log p(z, θ | D) using stable score function estimator.
    """
    d = z.shape[0]
    K = hparams['n_mc_samples']
    alpha = hparams['alpha']
    beta = hparams['beta']
    sigma_z = hparams['sigma_z']
    sigma_obs = hparams['sigma_obs']
    theta_prior_sigma = hparams['theta_prior_sigma']
    
    # Prior gradient: ∇_z log p(z)
    grad_z_prior = -z / (sigma_z ** 2)
    
    # Sample hard graphs once for consistency
    with torch.no_grad():
        g_soft = soft_gmat(z, alpha)
        hard_graphs = torch.bernoulli(g_soft.unsqueeze(0).repeat(K, 1, 1))  # (K, d, d)
    
    # Collect samples for likelihood and acyclicity terms
    ll_grad_samples = []
    ll_log_density_samples = []
    acyc_grad_accumulator = torch.zeros_like(z)
    
    for k in range(K):
        g_k = hard_graphs[k]  # (d, d)
        
        # Score function ∇_z log q(G_k | z)
        score_k = score_g_given_z(z, g_k, alpha)
        
        # Log likelihood + theta prior for this graph
        with torch.no_grad():
            log_lik = log_full_likelihood(data, g_k, theta, sigma_obs)
            theta_eff = theta * g_k
            log_theta_pr = log_theta_prior(theta_eff, theta_prior_sigma)
            log_density_k = log_lik + log_theta_pr
        
        ll_grad_samples.append(score_k)
        ll_log_density_samples.append(log_density_k)
        
        # Acyclicity constraint gradient
        with torch.no_grad():
            h_k = acyclic_constr(g_k, d)
        acyc_grad_accumulator += h_k * score_k
    
    # Stable likelihood gradient
    grad_z_likelihood = stable_ratio_estimator(ll_grad_samples, ll_log_density_samples)
    
    # Acyclicity gradient (simple Monte Carlo average)
    grad_z_acyclic = acyc_grad_accumulator / K
    
    # Total gradient
    total_grad = grad_z_prior + grad_z_likelihood - beta * grad_z_acyclic
    
    return total_grad

def grad_theta_log_joint(z: torch.Tensor, theta: torch.Tensor, data: Dict[str, Any], hparams: Dict[str, Any]) -> torch.Tensor:
    """
    Compute ∇_θ log p(z, θ | D) using stable ratio estimator.
    """
    K = hparams['n_mc_samples']
    alpha = hparams['alpha']
    sigma_obs = hparams['sigma_obs']
    theta_prior_sigma = hparams['theta_prior_sigma']
    
    # Sample hard graphs
    with torch.no_grad():
        g_soft = soft_gmat(z, alpha)
        hard_graphs = torch.bernoulli(g_soft.unsqueeze(0).repeat(K, 1, 1))  # (K, d, d)
    
    grad_samples = []
    log_density_samples = []
    
    for k in range(K):
        g_k = hard_graphs[k]  # (d, d)
        
        # Create copy of theta for gradient computation
        theta_k = theta.clone().requires_grad_(True)
        
        # Compute log likelihood + theta prior
        log_lik = log_full_likelihood(data, g_k, theta_k, sigma_obs)
        theta_eff = theta_k * g_k
        log_theta_pr = log_theta_prior(theta_eff, theta_prior_sigma)
        log_density_k = log_lik + log_theta_pr
        
        # Compute gradient
        grad_k, = torch.autograd.grad(log_density_k, theta_k, create_graph=False)
        
        grad_samples.append(grad_k.detach())
        log_density_samples.append(log_density_k.detach())
    
    # Use stable ratio estimator
    return stable_ratio_estimator(grad_samples, log_density_samples)

def log_joint(params: Dict[str, torch.Tensor], data: Dict[str, Any], hparams: Dict[str, Any]) -> torch.Tensor:
    """
    Compute log p(z, θ | D) for monitoring purposes.
    """
    z = params['z']
    theta = params['theta']
    d = z.shape[0]
    alpha = hparams['alpha']
    beta = hparams['beta']
    sigma_z = hparams['sigma_z']
    sigma_obs = hparams['sigma_obs']
    theta_prior_sigma = hparams['theta_prior_sigma']
    
    # Use soft graph for approximation
    g_soft = soft_gmat(z, alpha)
    
    # Likelihood
    log_lik = log_full_likelihood(data, g_soft, theta, sigma_obs)
    
    # Z prior (Gaussian + acyclicity)
    log_z_gaussian = torch.sum(Normal(0.0, sigma_z).log_prob(z))
    
    # Approximate acyclicity constraint
    h_soft = acyclic_constr(g_soft, d)
    log_z_acyclic = -beta * h_soft
    
    # Theta prior
    theta_eff = theta * g_soft
    log_theta_pr = log_theta_prior(theta_eff, theta_prior_sigma)
    
    return log_lik + log_z_gaussian + log_z_acyclic + log_theta_pr

def update_hparams(hparams: Dict[str, Any], t: int) -> Dict[str, Any]:
    """
    Update hyperparameters with annealing schedule.
    """
    total_steps = hparams['total_steps']
    progress = t / total_steps
    
    # Create a copy to avoid modifying the original
    updated_hparams = hparams.copy()
    
    # Alpha annealing: gradual increase
    updated_hparams['alpha'] = hparams['alpha_base'] * min(1.0, progress * 2.0)
    
    # Beta annealing: burn-in period then gradual increase
    burn_in = 0.2
    if progress < burn_in:
        updated_hparams['beta'] = 0.0
    else:
        adj_progress = (progress - burn_in) / (1.0 - burn_in)
        updated_hparams['beta'] = hparams['beta_base'] * adj_progress
    
    updated_hparams['current_iter'] = t
    return updated_hparams

def grad_log_joint_combined(params: Dict[str, torch.Tensor], data: Dict[str, Any], hparams: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Compute gradients for both z and theta.
    """
    # Detach parameters to avoid cross-contamination
    z_detached = params['z'].detach().requires_grad_(True)
    theta_detached = params['theta'].detach().requires_grad_(True)
    
    grad_z = grad_z_log_joint(z_detached, params['theta'].detach(), data, hparams)
    grad_theta = grad_theta_log_joint(params['z'].detach(), theta_detached, data, hparams)
    
    return {'z': grad_z, 'theta': grad_theta}

class DiBSFixed(nn.Module):
    """
    Fixed DiBS implementation with proper gradient computation.
    """
    
    def __init__(self, d: int, k: int = 2, device: str = 'cpu'):
        super().__init__()
        self.d = d
        self.k = k
        
        # Initialize latent variables z
        self.z = nn.Parameter(torch.randn(d, k, 2, device=device))
        
        # Initialize edge weights theta
        self.theta = nn.Parameter(torch.randn(d, d, device=device))
        
    def forward(self, data: Dict[str, Any], hparams: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute gradients and log joint probability.
        """
        params = {'z': self.z, 'theta': self.theta}
        
        # Compute gradients
        grads = grad_log_joint_combined(params, data, hparams)
        
        # Compute log joint for monitoring
        log_joint_val = log_joint(params, data, hparams)
        
        return {
            'grad_z': grads['z'],
            'grad_theta': grads['theta'],
            'log_joint': log_joint_val,
            'soft_adj': soft_gmat(self.z, hparams['alpha'])
        }
    
    def get_hard_adjacency(self, hparams: Dict[str, Any], threshold: float = 0.5) -> torch.Tensor:
        """
        Get hard adjacency matrix by thresholding soft probabilities.
        """
        g_soft = soft_gmat(self.z, hparams['alpha'])
        return (g_soft > threshold).float()