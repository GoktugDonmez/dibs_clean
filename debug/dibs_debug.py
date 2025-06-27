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
    d= z.shape[0]
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

def analytic_score_g_given_z(z, g, hparams):
    # 1. logits and probabilities
    probs = soft_gmat(z, hparams)
    diff   = g - probs                 # (g_ij − σ(s_ij))
    u, v   = z[..., 0], z[..., 1]      # (d,k)

    # 2. gradients wrt u and v
    grad_u = hparams['alpha'] * torch.einsum('ij,jk->ik', diff, v)   # (d,k)
    grad_v = hparams['alpha'] * torch.einsum('ij,ik->jk', diff, u)   # (d,k)

    return torch.stack([grad_u, grad_v], dim=-1)          # (d,k,2)



## TODO CHECK THE CODE
def score_acyclic_constr_mc(z: torch.Tensor, d: int, hparams: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    score estimator for the acyclicity constraint
        
    Returns:
        A tuple containing:
        - h_g (torch.Tensor): The acyclicity values for each sample [n_samples].
        - log_prob_g (torch.Tensor): The log probability of each sample [n_samples].
    """
    n_samples = hparams.get('n_mc_samples', 64)
    
    # 1. Sample hard graphs
    g_samples = bernoulli_sample(z, hparams, n_samples=n_samples)

    # 2. Calculate the "reward" h(G) for each sample
    total_acyl_score = 0
    for g in g_samples:
        reward = acyclic_constr(g, d)
        score = analytic_score_g_given_z(z, g, hparams)
        total_acyl_score += reward * score
    
    return total_acyl_score / n_samples



def grad_z_score(z: torch.Tensor, data: Dict[str, Any], theta: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    
    # acylic + prior + likelihood

    # acylic grad
    grad_acylic = score_acyclic_constr_mc(z, data, theta, hparams)
    grad_z_prior = - (z / hparams['sigma_z']**2)

    grad_prior = grad_z_prior - ( hparams['beta'] * grad_acylic)


    # likelihood grad with score estimator
    n_samples = hparams.get('n_mc_samples', 64)
    total_likelihood_score = 0
    for _ in range(n_samples):
        g_hard = bernoulli_sample(z, hparams)
        ## what should go here?  the full likelihood as the reward ?
        
        log_joint_reward  = log_full_likelihood(data, g_hard, theta, hparams)  + log_theta_prior(theta * g_hard, hparams.get('theta_prior_sigma'))

        score = analytic_score_g_given_z(z, g_hard, hparams)

        total_likelihood_score += log_joint_reward * score
    grad_likelihood = total_likelihood_score / n_samples

    total_grad = grad_prior + grad_likelihood

    return total_grad

def grad_theta_score(z: torch.Tensor, data: Dict[str, Any], theta: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    """
    Computes the gradient ∇_Θ log p(Z,Θ|D) using a self-normalized importance sampling
    estimator, which correctly implements the ratio formula from the DiBS paper.
    """
    n_samples = hparams.get('n_grad_mc_samples', 64)
    
    log_density_samples = []
    grad_samples = []

    for _ in range(n_samples):
        theta_for_grad = theta.clone().requires_grad_(True)
        
        # 1. Sample a hard graph G
        with torch.no_grad():
            g_hard = torch.bernoulli(soft_gmat(z, hparams))

        # 2. Calculate the log-density `log p(D,Θ|G)` for this sample.
        # This will be used to create the softmax weights.
        log_density = (
            log_full_likelihood(data, g_hard, theta_for_grad, hparams) + 
            log_theta_prior(theta_for_grad * g_hard, hparams.get('theta_prior_sigma', 1.0))
        )
        
        # 3. Calculate the gradient of this log-density, ∇_Θ log p(D,Θ|G).
        grad, = torch.autograd.grad(log_density, theta_for_grad)

        # Store the results for this sample
        log_density_samples.append(log_density)
        grad_samples.append(grad)

    # 4. Compute the final gradient as a weighted average.
    log_p = torch.stack(log_density_samples)
    grad_p = torch.stack(grad_samples)
    
    # Use the log-sum-exp trick for numerically stable softmax weights
    weights = torch.softmax(log_p, dim=0)

    # Broadcast weights `(n_samples)` to match gradient shape `(n_samples, d, d)`
    while weights.dim() < grad_p.dim():
        weights = weights.unsqueeze(-1)
        
    final_grad = (weights * grad_p).sum(dim=0)

    return final_grad    


def grad_log_joint(params: Dict[str, torch.Tensor], data: Dict[str, Any], hparams: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A top-level function that orchestrates the calculation of all gradients.
    """
    grad_z = grad_z_score(params["z"], data, params["theta"].detach(), hparams)
    grad_th = grad_theta_score(params["z"].detach(), data, params["theta"], hparams)
    return grad_z, grad_th

def log_joint_current(z: torch.Tensor, theta: torch.Tensor, data: Dict[str, Any], hparams: Dict[str, Any]) -> torch.Tensor:
    """
    Computes the value of the unnormalized log-joint probability for debugging.
    This function combines the different log-probability terms.
    """
    g_soft = soft_gmat(z, hparams)
    # For simplicity, we can approximate the expected likelihood with the soft graph
    log_likelihood_val = log_full_likelihood(data, g_soft, theta, hparams)
    log_theta_prior_val = log_theta_prior(theta, hparams.get('sigma_theta_prior', 1.0))
    # The acyclicity constraint is harder to evaluate without sampling, so we can skip it for a rough estimate
    # or use the soft graph version
    acyclicity_val = acyclic_constr(g_soft, z.shape[0])
    # Prior on Z
    log_z_prior = -0.5 * torch.sum(z**2) / hparams.get('latent_prior_std', 1.0)**2
    log_joint = log_likelihood_val + log_theta_prior_val + log_z_prior - hparams['beta'] * acyclicity_val
    logging.info(f"Log-joint: {log_joint.item():.4f}, "
                 f"Log-Likelihood: {log_likelihood_val.item():.4f}, "
                 f"Log-Theta-Prior: {log_theta_prior_val.item():.4f}, "
                 f"Acyclicity: {acyclicity_val.item():.4f}")
    return log_joint

def update_hyperparameters(hparams: Dict[str, Any], t: int) -> Dict[str, Any]:
    """
    Handles annealing schedules for hyperparameters.
    """
    # Simple linear annealing
    if hparams.get('alpha_linear'):
        hparams['alpha'] = hparams['alpha_linear'] * t *0.2
    if hparams.get('beta_linear'):
        hparams['beta'] = hparams['beta_linear'] * t
    hparams['current_t'] = t
    return hparams