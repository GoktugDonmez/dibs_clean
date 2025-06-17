# models/dibs.py
import torch
from torch.distributions import Normal
import numpy as np
import logging
from typing import Dict, Any, Tuple

from .utils import acyclic_constr, stable_mean

log = logging.getLogger(__name__)

def log_gaussian_likelihood(x: torch.Tensor, pred_mean: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    sigma_tensor = torch.clamp(torch.tensor(sigma, dtype=pred_mean.dtype, device=pred_mean.device), min=1e-12)
    residuals = torch.clamp(x - pred_mean, min=-1e3, max=1e3)
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

def gumbel_soft_gmat(z: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    raw = scores(z, hparams['alpha'])
    u = torch.rand_like(raw)
    noise = torch.log(u) - torch.log1p(-u)
    soft = torch.sigmoid((noise + raw) * hparams['tau'])
    d = soft.shape[-1]
    mask = 1.0 - torch.eye(d, device=soft.device, dtype=soft.dtype)
    return soft * mask

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
        g_soft = gumbel_soft_gmat(z, hparams)
        h_samples.append(acyclic_constr(torch.bernoulli(g_soft), d))
    return torch.mean(torch.stack(h_samples))

def gumbel_grad_acyclic_constr_mc(z: torch.Tensor, d: int, hparams: Dict[str, Any]) -> torch.Tensor:
    """
    Monte Carlo estimation of gradient of acyclicity constraint using Gumbel soft adjacency matrices.
    
    This implements the MC estimator for ∇_z E[h(G_soft)] where G_soft ~ Gumbel-softmax.
    
    For each MC sample:
    1. Sample a Gumbel soft adjacency matrix G_soft using gumbel_soft_gmat(z, hparams)
    2. Apply acyclicity constraint h(G_soft) 
    3. Compute gradient ∇_z h(G_soft) via backprop
    4. Return mean of all gradient samples for unbiased MC estimate
    
    Args:
        z: Latent variables [d, k, 2] 
        d: Number of nodes
        hparams: Hyperparameters including n_nongrad_mc_samples
    
    Returns:
        MC estimate of ∇_z E[h(G_soft)] with same shape as z
    """
    n_mc = hparams.get('n_nongrad_mc_samples', 10)

    # make a view that shares storage with z, but has a new batch dim
    z_rep = z.unsqueeze(0).expand(n_mc, *z.shape).contiguous()
    z_rep.requires_grad_(True)

    # sample your soft graphs
    g_soft = gumbel_soft_gmat(z_rep, hparams)      # [n_mc, d, d]

    # for each sample, get the cycle‐penalty
    # (we sample a *hard* graph for h if that's what your paper says,
    #  but you can also do it on the soft if you like)
    h_vals = []
    for i in range(n_mc):
        h_vals.append(acyclic_constr(g_soft[i], d))
    h_vals = torch.stack(h_vals)                   # [n_mc]

    # average penalty
    h_mean = h_vals.mean()

    # differentiate once
    grad_z_rep, = torch.autograd.grad(h_mean, z_rep)

    # average over MC batch
    return grad_z_rep.mean(0)                      # shape [d,k,2]

def grad_z_log_joint_gumbel(z: torch.Tensor, theta: torch.Tensor, data: Dict[str, Any], hparams: Dict[str, Any]) -> torch.Tensor:
    d = z.shape[0]
    beta = hparams['beta']
    sigma_z_sq = hparams['sigma_z']**2
    theta_const = theta.detach()
    n_samples_tensor = torch.tensor(float(hparams['n_grad_mc_samples']), device=z.device)

    z.requires_grad_(True)
    # --- Part 1: Prior Gradient ---
    # MC estimate of gradient of acyclicity constraint using Gumbel soft graphs
    # This computes ∇_z E[h(G_soft)] where G_soft ~ Gumbel-softmax
    grad_h_mc = gumbel_grad_acyclic_constr_mc(z, d, hparams)
    
    # Prior gradient: acyclicity penalty + gaussian prior term
    # ∇_z log p(z) = -β * ∇_z E[h(G_soft)] - z/σ²
    grad_log_z_prior_total = -beta * grad_h_mc - z / sigma_z_sq

    # --- Part 2: Likelihood Gradient ---
    
    log_density_samples = []
    for _ in range(hparams['n_grad_mc_samples']):
        g_soft_mc = gumbel_soft_gmat(z, hparams)
        log_lik_val = log_full_likelihood(data, g_soft_mc, theta_const, hparams)
        theta_eff_mc = theta_const * g_soft_mc
        log_theta_prior_val = log_theta_prior(theta_eff_mc, hparams.get('theta_prior_sigma', 1.0))
        print(f"Log likelihood: {log_lik_val}, Log theta prior: {log_theta_prior_val}")
        log_density_samples.append(log_lik_val + log_theta_prior_val)
        
    log_p_tensor = torch.stack(log_density_samples)

    with torch.no_grad():
        max_log_p = torch.max(log_p_tensor)    



    log_weights = log_p_tensor - max_log_p           # still on same device
    log_den = torch.logsumexp(log_weights, dim=0)    # log E[p]
    denominator = torch.exp(log_den)                 # never < 1
    weights = torch.exp(log_weights - log_den)       # soft-max, Σw = 1
    mean_p_for_grad = torch.sum(weights * torch.exp(log_weights))   # == 1
    numerator_grad, = torch.autograd.grad(mean_p_for_grad, z)

    if z.grad is not None:
        z.grad.zero_()
    z.requires_grad_(False)
    
    # Final combined gradient
    

    return grad_log_z_prior_total + (numerator_grad / denominator)



import torch.nn.functional as F

def manual_stable_gradient(log_p_tensor: torch.Tensor, grad_p_tensor: torch.Tensor) -> torch.Tensor:
    """
    Manually calculates the stable gradient for learning purposes,
    re-implementing the logic of softmax and a weighted sum.

    Args:
        log_p_tensor: A tensor of log-joint probabilities, shape (n_samples,).
        grad_p_tensor: A tensor of the gradients of the log-joint probabilities,
                       shape (n_samples, *theta.shape).

    Returns:
        The numerically stable estimate of the gradient.
    """
    # --- Step 1: Manual Log-Sum-Exp ---
    # Find the maximum log-probability to shift the values for stability.
    log_p_max = torch.max(log_p_tensor)
    # Calculate log(sum(exp(L_j))) stably.
    log_denominator = log_p_max + torch.log(torch.sum(torch.exp(log_p_tensor - log_p_max)))

    # --- Step 2: Manual Softmax Weights ---
    # Calculate the log of the weights: log(w_m) = log(p_m) - log(sum(p_j))
    log_weights = log_p_tensor - log_denominator
    # Get the weights by exponentiating the stable log-weights.
    weights = torch.exp(log_weights)

    # --- Step 3: Weighted Average ---
    # Reshape weights to broadcast with the gradient tensor.
    dims_to_add = grad_p_tensor.dim() - 1
    weights_reshaped = weights.view(-1, *([1] * dims_to_add))
    # Multiply and sum to get the final gradient estimate.
    final_gradient = torch.sum(weights_reshaped * grad_p_tensor, dim=0)
    print(f"Final gradient:\n {final_gradient}")

    return final_gradient

def softmax_stable_gradient(log_p_tensor: torch.Tensor, grad_p_tensor: torch.Tensor, verbose: bool = True) -> torch.Tensor:
    """
    Calculates the stable gradient using the built-in torch.nn.functional.softmax,
    with added prints for debugging.

    Args:
        log_p_tensor: A tensor of log-joint probabilities, shape (n_samples,).
        grad_p_tensor: A tensor of the gradients of the log-joint probabilities,
                       shape (n_samples, *theta.shape).
        verbose: If True, prints diagnostic information.

    Returns:
        The numerically stable estimate of the gradient.
    """
    if verbose:
        print("\n--- Softmax Version Diagnostics ---")
        # 1. Analyze the input log-probabilities. This tells you the range of your joint distribution values.
        print(f"\n[Input log_p_tensor stats]")
        print(f"  Min: {log_p_tensor.min():.4f}, Max: {log_p_tensor.max():.4f}, Mean: {log_p_tensor.mean():.4f}, Std: {log_p_tensor.std():.4f}")

        # 2. Analyze the raw gradients before they are averaged. This is key.
        # If these values are large, the output will likely be large.
        print(f"\n[Input grad_p_tensor stats (raw gradients)]")
        print(f"  Min: {grad_p_tensor.min():.4f}, Max: {grad_p_tensor.max():.4f}, Mean: {grad_p_tensor.mean():.4f}")

    # --- Step 1: Calculate weights using built-in softmax ---
    # This is the numerically stable way to get the weights w_m = p_m / sum(p_j).
    weights = F.softmax(log_p_tensor, dim=0)

    if verbose:
        # 3. Analyze the resulting weights. This tells you how the average is distributed.
        # If Max is close to 1.0, one sample is dominating the expectation.
        print(f"\n[Calculated softmax weights stats]")
        print(f"  Min: {weights.min():.4e}, Max: {weights.max():.4e}, Sum: {weights.sum():.4f}")
        # It's also useful to see how many weights are non-trivial
        non_trivial_weights = (weights > 1e-6).sum()
        print(f"  Number of non-trivial weights (>1e-6): {non_trivial_weights}/{len(weights)}")

    # --- Step 2: Calculate the weighted average ---
    # torch.einsum is a clean, general way to perform the weighted sum.
    # 's' is the sample dimension, '...' are the remaining gradient dimensions.
    final_gradient = torch.einsum('s,s...->...', weights, grad_p_tensor)

    if verbose:
        # 4. Analyze the final output gradient.
        print("\n[Final weighted gradient stats]")
        print(f"  Min: {final_gradient.min():.4f}, Max: {final_gradient.max():.4f}, Mean: {final_gradient.mean():.4f}")
        print(f"Final gradient tensor:\n{final_gradient}")
        print("--- End Diagnostics ---\n")

    return final_gradient



def grad_theta_log_joint(z: torch.Tensor, theta: torch.Tensor, data: Dict[str, Any], hparams: Dict[str, Any]) -> torch.Tensor:
    theta.requires_grad_(True)
    n_samples = hparams.get('n_grad_mc_samples', 1)

    theta = theta.clone().detach().requires_grad_(True)
    log_density_samples = []
    grad_samples = []
    for _ in range(n_samples):
        g_soft = bernoulli_soft_gmat(z, hparams)
        #g_hard = torch.bernoulli(g_soft)
        log_lik_val = log_full_likelihood(data, g_soft, theta, hparams)
        theta_eff = theta * g_soft
        log_theta_prior_val = log_theta_prior(theta_eff, hparams.get('theta_prior_sigma', 1.0))

        current_log_density = log_lik_val + log_theta_prior_val
        current_grad ,= torch.autograd.grad(current_log_density, theta)
        log_density_samples.append(current_log_density) 
        grad_samples.append(current_grad)

    log_p_tensor = torch.stack(log_density_samples)
    grad_p_tensor = torch.stack(grad_samples)


    # Cleanup
    if theta.grad is not None:
        theta.grad.zero_()
    theta.requires_grad_(False)

    #grad =manual_stable_gradient(log_p_tensor, grad_p_tensor)
    grad = softmax_stable_gradient(log_p_tensor, grad_p_tensor)

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
    t = max(t_step, 1e-3)
    factor = t + 1.0 / t               # JAX uses (t + 1/t)
    hparams['beta']  = hparams['beta_base'] * factor          
    hparams['current_iteration'] = t_step # Store current iteration
    hparams_updated = hparams
    return hparams_updated


def hard_gmat_from_z(z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    s = scores(z, alpha)
    return (s > 0).float()
