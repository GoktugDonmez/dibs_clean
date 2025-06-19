# models/dibs.py
import torch
from torch.distributions import Normal
import numpy as np
import logging
from typing import Dict, Any, Tuple

from .utils import acyclic_constr, stable_mean

log = logging.getLogger(__name__)

def log_gaussian_likelihood(x: torch.Tensor, pred_mean: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    sigma_tensor = torch.tensor(sigma, dtype=pred_mean.dtype, device=pred_mean.device)
    residuals = x - pred_mean
    log_prob = -0.5 * (np.log(2 * np.pi) + 2 * torch.log(sigma_tensor) + (residuals / sigma_tensor) ** 2)
    return torch.sum(log_prob)

def scores(z: torch.Tensor, alpha: float) -> torch.Tensor:
    u, v = z[..., 0], z[..., 1]
    raw_scores = alpha * torch.einsum('...ik,...jk->...ij', u, v)
    *batch_dims, d, _ = z.shape[:-1]
    diag_mask = 1.0 - torch.eye(d, device=z.device, dtype=z.dtype)
    if batch_dims:
        diag_mask = diag_mask.expand(*batch_dims, d, d)
    return raw_scores * diag_mask

def bernoulli_soft_gmat(z: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    probs = torch.sigmoid(scores(z, hparams["alpha"]))
    d = probs.shape[-1]
    diag_mask = 1.0 - torch.eye(d, device=probs.device, dtype=probs.dtype)
    if probs.ndim == 3:
        diag_mask = diag_mask.expand(probs.shape[0], d, d)
    return probs * diag_mask

def gumbel_soft_gmat(z: torch.Tensor,
                     hparams: Dict[str, Any]) -> torch.Tensor:
    """
    Soft Gumbel–Softmax adjacency  (Eq. B.6)

        g_ij  = σ_τ( L_ij + α⟨u_i , v_j⟩ )

    where  L_ij ~ Logistic(0,1)  and  τ = hparams['tau']. appendix b2
    """
    raw = scores(z, hparams["alpha"])

    # Logistic(0,1) noise   L = log U - log(1-U)
    u = torch.rand_like(raw)
    L = torch.log(u) - torch.log1p(-u)

    logits = (raw + L) / hparams["tau"]
    g_soft = torch.sigmoid(logits)

    d = g_soft.size(-1)
    mask = 1.0 - torch.eye(d, device=z.device, dtype=z.dtype)
    return g_soft * mask

def log_full_likelihood(data: Dict[str, Any], soft_gmat: torch.Tensor, theta: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    ## TODO: Expert belief: update this to use interventions, change the full likelihood 
    # and also add log bernoulli likelihood calculatior
    x_data = data['x']
    effective_W = theta * soft_gmat
    pred_mean = torch.matmul(x_data, effective_W)
    sigma_obs = hparams.get('sigma_obs_noise', 0.1)
    return log_gaussian_likelihood(x_data, pred_mean, sigma=sigma_obs)

def log_theta_prior(theta_effective: torch.Tensor, sigma: float) -> torch.Tensor:
    return log_gaussian_likelihood(theta_effective, torch.zeros_like(theta_effective), sigma=sigma)

def gumbel_acyclic_constr_mc(z: torch.Tensor, d: int, hparams: Dict[str, Any]) -> torch.Tensor:
    h_samples = []
    for _ in range(hparams['n_nongrad_mc_samples']):
        # FOR NOW, JUST GIVE THE SOFT MATRIX, AND BY ANNEALING IT TO HARD MATRIX
        g_soft = gumbel_soft_gmat(z, hparams)
        h_samples.append(acyclic_constr(g_soft, d))
        
        # should gumbel soft gmat to hard gmat be done with >0.5 or with a sigmoid?  
        # g_hard = torch.bernoulli(g_soft) ? 
        #g_hard = (g_soft > 0.5).float()
        #how about this  mentioned in dibs       g_ST   = g_hard + (g_soft - g_soft.detach())   # straight-through
        
        #TODO fix above
        # for now use g_soft
        
        ### STRAIGHT THROUGH
        #g_hard = (g_soft > 0.5).float()  # Convert to hard graph
        #g_ST = g_hard + (g_soft - g_soft.detach()) 
        #h_samples.append(acyclic_constr(g_ST, d))
    h_samples = torch.stack(h_samples)


    return torch.mean(h_samples, dim=0)

def grad_z_log_joint_gumbel(z: torch.Tensor, theta: torch.Tensor, data: Dict[str, Any], hparams: Dict[str, Any]) -> torch.Tensor:
    d = z.shape[0]
    beta = hparams['beta']
    sigma_z_sq = hparams['sigma_z']**2
    theta_const = theta.detach()
    n_samples_tensor = torch.tensor(float(hparams['n_grad_mc_samples']), device=z.device)

    z.requires_grad_(True)
    # --- Part 1: Prior Gradient ---
    # MC estimate of gradient of acyclicity constraint using Gumbel soft graphs

    h_mean = gumbel_acyclic_constr_mc(z, d, hparams)
    grad_h_mc = torch.autograd.grad(h_mean, z)[0]
    grad_log_z_prior_total = -beta * grad_h_mc - z / sigma_z_sq

    # --- Part 2: Likelihood Gradient ---
    
    log_density_samples = []
    for _ in range(hparams['n_grad_mc_samples']):
        g_soft = gumbel_soft_gmat(z, hparams)

        log_lik_val = log_full_likelihood(data, g_soft, theta_const, hparams)
        theta_eff_mc = theta_const * g_soft
        log_theta_prior_val = log_theta_prior(theta_eff_mc, hparams.get('theta_prior_sigma', 1.0))
        log_density_samples.append(log_lik_val + log_theta_prior_val)
    #print(f'log_density_samples shape: {len(log_density_samples)}')
    #print(f'log_density_samples values: {log_density_samples}')
    log_p = torch.stack(log_density_samples)

    log_sum = torch.logsumexp(log_p, dim=0)
    
    # Compute the gradient directly on log_sum. This avoids the unstable exp() and division.
    grad_lik, = torch.autograd.grad(log_sum, z)


    if z.grad is not None:
        z.grad.zero_()
    z.requires_grad_(False)
    
    # Final combined gradient
    



    # 3) Combine
    # ------------------------------------------------
    return grad_log_z_prior_total + grad_lik





def weighted_grad(log_p: torch.Tensor,
                  grad_p: torch.Tensor) -> torch.Tensor:
    """
    Return   Σ softmax(log_p)_m * grad_p[m]
    Shapes
        log_p   : (M,)
        grad_p  : (M, …)   (any extra dims)
    """
    # 1. numerically stable soft-max weights
    #print(f'log_p shape: {log_p.shape}, values:\n {log_p}')
    #print(f'grad_p shape: {grad_p.shape}, values: \n{grad_p}')
    log_p_shifted = log_p - log_p.max()          # (M,)
    #print(f'log_p_shifted shape: {log_p_shifted.shape}, values: \n {log_p_shifted}')
    w = torch.exp(log_p_shifted)
    #print(f'w shape: {w.shape}, values:\n {w}')
    w = w / w.sum()
    #print(f'w after normalization shape: {w.shape}, values:\n {w}')

    # 2. broadcast weights onto grad tensor
    while w.dim() < grad_p.dim():
        w = w.unsqueeze(-1)                      # (M,1,1,...)

    return (w * grad_p).sum(dim=0)               # same shape as grad slice



def logsumexp_v1(log_tensor: torch.Tensor) -> torch.Tensor:


    M = log_tensor.shape[0]
    logM = torch.log(torch.tensor(M, dtype=log_tensor.dtype, device=log_tensor.device))

    
    log_sum_exp = torch.logsumexp(log_tensor, dim=0)

    total = log_sum_exp - logM
    return total # torch.exp(total)

def manual_stable_gradient(log_p_tensor: torch.Tensor, grad_p_tensor: torch.Tensor) -> torch.Tensor:
# uses the logsumexp_v1 function to compute the stable gradient

    print(f'log density values and shape: {log_p_tensor}, {log_p_tensor.shape}')
    log_density_lse = torch.exp(torch.logsumexp(log_p_tensor, dim=0) - log_p_tensor.shape[0])  # logsumexp_v1(log_p_tensor)
    # logsumexp_v1(log_p_tensor)
    print(f'log density lse value: {log_density_lse}, shape: {log_density_lse.shape}')

    print('-' * 50)
    print(f'grad density values and shape: {grad_p_tensor}, {grad_p_tensor.shape}')
    grad_lse = logsumexp_v1(grad_p_tensor)
    print(f'grad density lse value: {grad_lse}, shape: {grad_lse.shape}')

    return torch.exp(logsumexp_v1(grad_p_tensor) - logsumexp_v1(log_p_tensor)) # grad_lse / log_density_lse[:, None]


def grad_theta_log_joint(z: torch.Tensor, theta: torch.Tensor, data: Dict[str, Any], hparams: Dict[str, Any]) -> torch.Tensor:
    theta.requires_grad_(True)
    n_samples = hparams.get('n_grad_mc_samples', 1)
    theta = theta.clone().detach().requires_grad_(True)
    log_density_samples = []
    grad_samples = []
    for _ in range(n_samples):
        g_soft = bernoulli_soft_gmat(z, hparams)
        #print(f"g_soft values: {g_soft}")
        g_hard = torch.bernoulli(g_soft)
        #print(f"g_hard values: {g_hard}")

        # tryign with gumbel to be consistent with grad z and gumbel mc acylci impelmentation
        #g_soft = gumbel_soft_gmat(z, hparams)





        log_lik_val = log_full_likelihood(data, g_hard, theta, hparams)
        theta_eff = theta * g_hard
        log_theta_prior_val = log_theta_prior(theta_eff, hparams.get('theta_prior_sigma', 1.0))
        #print(f"Grad_theta mc_samples Log likelihood:  {log_lik_val} Log theta prior: \n {log_theta_prior_val} \n ")

        current_log_density = log_lik_val + log_theta_prior_val
        current_grad ,= torch.autograd.grad(current_log_density, theta)
        log_density_samples.append(current_log_density) 

        grad_samples.append(current_grad)
    #print(f" END OF Grad_theta mc_samples, iter number: {hparams.get('current_iteration',1)}  \n")

    log_p_tensor = torch.stack(log_density_samples)
    grad_p_tensor = torch.stack(grad_samples)


    # Cleanup
    if theta.grad is not None:
        theta.grad.zero_()
    theta.requires_grad_(False)

    grad =weighted_grad(log_p_tensor, grad_p_tensor)
    #grad = stable_gradient_estimator(log_p_tensor, grad_p_tensor)
    #print(f"Grad_theta shape: {grad.shape}, values: \n {grad}")

    return  grad


def grad_log_joint(params: Dict[str, torch.Tensor], data: Dict[str, Any], hparams: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    z, theta = params['z'], params['theta']

    grad_z = grad_z_log_joint_gumbel(z, theta.detach(), data, hparams)
    grad_theta = grad_theta_log_joint(z.detach(), theta, data, hparams)
    
    return {"z": grad_z, "theta": grad_theta}

def log_joint(params: Dict[str, torch.Tensor], data: Dict[str, Any], hparams: Dict[str, Any]) -> torch.Tensor:
    hparams_updated = update_dibs_hparams(hparams, params["t"].item())
    z, theta = params['z'], params['theta']
    d = z.shape[0]

    g_soft = bernoulli_soft_gmat(z, hparams_updated)
    log_lik = log_full_likelihood(data, g_soft, theta, hparams_updated)

    log_prior_z_gaussian = torch.sum(Normal(0.0, hparams_updated['sigma_z']).log_prob(z))
    expected_h_val = gumbel_acyclic_constr_mc(z, d, hparams_updated)
    log_prior_z_acyclic = -hparams_updated['beta'] * expected_h_val
    log_prior_z = log_prior_z_gaussian + log_prior_z_acyclic
    
    theta_eff = theta * g_soft
    log_prior_theta = log_theta_prior(theta_eff, hparams_updated.get('theta_prior_sigma', 1.0))
    
    return log_lik + log_prior_z + log_prior_theta

def update_dibs_hparams(hparams: Dict[str, Any], t_step: float) -> Dict[str, Any]:
    t = max(t_step, 1.0) # Ensure t is at least 1


    beta_max = 2000.0
    beta_tau = 50         # iterations until β≈0.95*beta_max
    #hparams['beta'] = beta_max * (1 - np.exp(-t/beta_tau))
    #hparams['beta'] = hparams['beta_base'] * 0.2 * t

    # Linear annealing for beta
    #hparams['beta'] = hparams['beta_base'] + (t - 1) * 0.2 # Adjust the step size as needed

    # Your alpha annealing
    #hparams['alpha'] = hparams['alpha_base'] * 0.2 * t
    
    beta_max = 2000.0  # Set a reasonable maximum
    beta_tau = 1000
    hparams['beta'] = beta_max * (1 - np.exp(-t_step / beta_tau))
    
    # Anneal alpha (sharpness) even more slowly. Start it later if needed.
    alpha_max = 10.0
    alpha_tau = 2000
    hparams['alpha'] = alpha_max * (1 - np.exp(-t_step / alpha_tau))




    hparams['current_iteration'] = t_step # Store current iteration
    return hparams


def hard_gmat_from_z(z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    s = scores(z, alpha)
    return (s > 0).float()
