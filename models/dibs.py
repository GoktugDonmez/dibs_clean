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
    
    ## USING THE LIB
    normal_dist = Normal(loc=pred_mean, scale=sigma_tensor)
    log_prob = normal_dist.log_prob(x).sum()

    ## USING THE LOG PROBABILITY FROM THE GAUSSIAN DISTRIBUTION
    #residuals = x - pred_mean
    #log_prob = -0.5 * (torch.log(2 * torch.pi * sigma_tensor**2)) - 0.5 * ((residuals / sigma_tensor)**2)


    return log_prob

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
        #print(f'g_soft shape: {g_soft.shape}, values: \n {g_soft}')
        #if hparams['current_iteration'] % 1 == 0:
        #    print(f'g_soft shape: {g_soft.shape}, values: \n {g_soft}')
        #g_hard = torch.bernoulli(g_soft)
        #if hparams['current_iteration'] % 1 == 0:
        #    print(f'g_hard shape: {g_hard.shape}, values: \n {g_hard}')
        #print(f'g_hard shape: {g_hard.shape}, values: \n {g_hard}')
        #h_samples.append(acyclic_constr(g_hard, d))
        #g_hard = (g_soft > 0.5).float()
        #how about this  mentioned in dibs       g_ST   = g_hard + (g_soft - g_soft.detach())   # straight-through
        
        #TODO fix above
        # for now use g_soft
        
    h_samples = torch.stack(h_samples)


    return torch.mean(h_samples, dim=0)

def grad_z_log_joint_gumbel(z: torch.Tensor, theta: torch.Tensor, data: Dict[str, Any], hparams: Dict[str, Any]) -> torch.Tensor:
    d = z.shape[0]
    theta_const = theta
    
    #z.requires_grad_(True)
    # --- Part 1: Prior Gradient ---
    # MC estimate of gradient of acyclicity constraint using Gumbel soft graphs

    h_mean = gumbel_acyclic_constr_mc(z, d, hparams)
    grad_h_mc = torch.autograd.grad(h_mean, z)[0]
    grad_log_z_prior_total = -hparams['beta'] * grad_h_mc - (z / hparams['sigma_z']**2)

    # --- Part 2: Likelihood Gradient ---
    
    # 1. We need to collect the log-probability AND the gradient for each sample.
    log_density_samples = []
    grad_samples = []

    for _ in range( hparams['n_grad_mc_samples']):
        # 2. Generate a single soft graph sample.
        g_soft = gumbel_soft_gmat(z, hparams)

        # 3. Calculate the log-joint for this single sample.
        log_density_one_sample = log_full_likelihood(data, g_soft, theta_const, hparams) + \
                                 log_theta_prior(theta_const * g_soft, hparams.get('theta_prior_sigma', 1.0))

        # 4. Calculate the gradient for this single sample.
        # We must use retain_graph=True because we are doing a backward pass
        # inside a loop, and PyTorch would otherwise free the graph memory.
        grad, = torch.autograd.grad(log_density_one_sample, z)
        
        log_density_samples.append(log_density_one_sample)
        grad_samples.append(grad)

    # 5. After the loop, we can safely detach z_ from any further graph history.

    # 6. Compute the final likelihood gradient using the stable weighted average.
    # This correctly computes E[p*∇log(p)] / E[p]
    log_p = torch.stack(log_density_samples)
    grad_p = torch.stack(grad_samples)
    grad_lik = weighted_grad(log_p, grad_p)


    #if z.grad is not None:
    #    z.grad.zero_()
    #z.requires_grad_(False)
    
    # Final combined gradient
    


    total = grad_log_z_prior_total + grad_lik
    # 3) Combine
    # ------------------------------------------------
    return total.detach()


## SCORE BASED ESTIMATOR FOR GRADIENT Z 


# ------------------------------------------------------------
#  Score-function estimator for ∇_Z log p(Z,Θ | D)
#  (Section B.2 of the paper, b = 0)
# ------------------------------------------------------------
def analytic_score_g_given_z(z, g, hparams):
    # 1. logits and probabilities
    probs = bernoulli_soft_gmat(z, hparams)
    diff   = g - probs                 # (g_ij − σ(s_ij))
    u, v   = z[..., 0], z[..., 1]      # (d,k)

    # 2. gradients wrt u and v
    grad_u = hparams['alpha'] * torch.einsum('ij,jk->ik', diff, v)   # (d,k)
    grad_v = hparams['alpha'] * torch.einsum('ij,ik->jk', diff, u)   # (d,k)

    return torch.stack([grad_u, grad_v], dim=-1)          # (d,k,2)


def grad_z_log_joint_score(z: torch.Tensor,
                           theta: torch.Tensor,
                           data: Dict[str, Any],
                           hparams: Dict[str, Any]) -> torch.Tensor:
    """
    ∇_Z log p(Z,Θ | D)  using the score-function (REINFORCE) estimator.

    This replaces the Gumbel-soft estimator.
    """
    sigma_z2 = hparams['sigma_z'] ** 2
    beta     = hparams['beta']

    M = hparams['n_grad_mc_samples'] # M = 50
    d = z.shape[0]                   # d = 4
    theta_const = theta


    # 1. sample hard graphs 
    with torch.no_grad():
        g_hard_samples = [torch.bernoulli(bernoulli_soft_gmat(z, hparams)) for _ in range(M)]

    ll = []
    scores = []
    for g in g_hard_samples:
        log_lik = log_full_likelihood(data, g, theta_const, hparams)
        theta_eff = theta_const * g
        log_theta_prior_val = log_theta_prior(theta_eff, hparams.get('theta_prior_sigma', 1.0))
        
        # log likelihood 
        ll.append(log_lik + log_theta_prior_val)
        
        # score 
        scores.append(analytic_score_g_given_z(z, g, hparams))
    
    log_p = torch.stack(ll)
    grad_p = torch.stack(scores)

    log_p_max = log_p.max()
    log_p_shifted = log_p - log_p_max
    #print(f'log_p_shifted shape: {log_p_shifted.shape}, values: \n {log_p_shifted}')
    unnormalized_w = torch.exp(log_p_shifted/10)
    #print(f'unnormalized_w shape: {unnormalized_w.shape}, values: \n {unnormalized_w}')
    w = unnormalized_w / unnormalized_w.sum()
                     # (M,1,1,...)
    #print(f'w shape: {w.shape}, values: \n {w}')

    while w.dim() < grad_p.dim():
        w = w.unsqueeze(-1)                    
    
    ## compute the weighted avg 
    grad_lik = (w * grad_p).sum(dim=0)




    # ---- Z-prior: Gaussian + acyclicity penalty --------------
    # gumbel is possible cuz no differentiable function in expectation 
    z_ = z.detach().clone().requires_grad_(True)
    h_mean = gumbel_acyclic_constr_mc(z_, d, hparams)       # differentiable w.r.t z_
    grad_h, = torch.autograd.grad(h_mean, z_, retain_graph=False)
    grad_prior = -beta * grad_h - z_ / sigma_z2

    return (grad_lik + grad_prior).detach()



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



def grad_theta_log_joint(z: torch.Tensor, theta: torch.Tensor, data: Dict[str, Any], hparams: Dict[str, Any]) -> torch.Tensor:
    #theta.requires_grad_(True)
    n_samples = hparams.get('n_grad_mc_samples', 1)
    theta_ = theta.clone().detach().requires_grad_(True)
    log_density_samples = []
    grad_samples = []
    for _ in range(n_samples):
        g_soft = bernoulli_soft_gmat(z, hparams)
        #print(f"g_soft values: {g_soft}")
        g_hard = torch.bernoulli(g_soft)
        #print(f"g_hard values: {g_hard}")

        # tryign with gumbel to be consistent with grad z and gumbel mc acylci impelmentation
        #g_soft = gumbel_soft_gmat(z, hparams)




        log_lik_val = log_full_likelihood(data, g_hard, theta_, hparams)
        theta_eff = theta_ * g_hard
        log_theta_prior_val = log_theta_prior(theta_eff, hparams.get('theta_prior_sigma', 1.0))
        #ll_grad, = torch.autograd.grad(log_lik_val, theta_, retain_graph=True)
        #log_theta_prior_grad, = torch.autograd.grad(log_theta_prior_val, theta_ , retain_graph=True)
        #print(f"ll_grad shape: {ll_grad.shape}, values: {ll_grad}")
        #print(f"log_theta_prior_grad shape: {log_theta_prior_grad.shape}, values: {log_theta_prior_grad}")

        current_log_density = log_lik_val + log_theta_prior_val
        current_grad ,= torch.autograd.grad(current_log_density, theta_)
        log_density_samples.append(current_log_density) 

        grad_samples.append(current_grad)
    #print(f" END OF Grad_theta mc_samples, iter number: {hparams.get('current_iteration',1)}  \n")

    log_p_tensor = torch.stack(log_density_samples)
    grad_p_tensor = torch.stack(grad_samples)


    # Cleanup
    #if theta.grad is not None:
    #    theta.grad.zero_()
    #theta.requires_grad_(False)

    grad =weighted_grad(log_p_tensor, grad_p_tensor)
    #grad = stable_gradient_estimator(log_p_tensor, grad_p_tensor)
    #print(f"Grad_theta shape: {grad.shape}, values: \n {grad}")

    return  grad.detach()


def grad_log_joint(params: Dict[str, torch.Tensor], data: Dict[str, Any], hparams: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    grad_z = grad_z_log_joint_gumbel(params["z"], params["theta"].detach(), data, hparams)
    #grad_z = grad_z_log_joint_score(params["z"], params["theta"].detach(), data, hparams)
    grad_theta = grad_theta_log_joint(params["z"].detach(), params["theta"], data, hparams)
    
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

    if (hparams_updated['current_iteration'] > 850 and hparams_updated['current_iteration'] < 1200):
        with torch.no_grad():
            log_terms = {
                "log_lik":      log_lik.item(),
                "z_prior_gauss":log_prior_z_gaussian.item(),
                "z_prior_acyc": log_prior_z_acyclic.item(),   # usually ≤ 0
                "theta_prior":  log_prior_theta.item(),
                "log_joint": log_lik + log_prior_theta + log_prior_z + log_prior_z_acyclic,
                "penalty": -hparams_updated['beta'] * expected_h_val.item()
            }
        print(f"[dbg] {log_terms}")

    
    return log_lik + log_prior_z + log_prior_theta

def update_dibs_hparams(hparams: Dict[str, Any], t_step: float) -> Dict[str, Any]:

    hparams['beta'] = hparams['beta_base'] * t_step # linear 

    hparams['alpha'] = hparams['alpha_base'] * t_step  # linear slope 0.2





    hparams['current_iteration'] = t_step # Store current iteration
    return hparams


def hard_gmat_from_z(z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    s = scores(z, alpha)
    return (s > 0).float()
