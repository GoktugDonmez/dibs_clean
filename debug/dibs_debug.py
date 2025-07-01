import torch
from torch.distributions import Normal
import numpy as np
import logging
from typing import Dict, Any, Tuple
import torch.nn as nn
import igraph as ig  


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

def acyclic_constr(g: torch.Tensor, d: int) -> torch.Tensor:
    alpha = 1.0 / d
    eye = torch.eye(d, device=g.device, dtype=g.dtype)
    m = eye + alpha * g
    # The matrix power operation is a differentiable way to check for cycles.
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
  _, d = z.shape[:-1]
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

import torch.nn.functional as F
def score_autograd_g_given_z(z: torch.Tensor, g: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    logits = scores(z, hparams)
    log_prob_g = F.binary_cross_entropy_with_logits(input=logits, target=g, reduction='sum')
    score, = torch.autograd.grad(log_prob_g, z, create_graph=True)
    return score

import torch.nn.functional as F

def score_autograd_g_given_z(z: torch.Tensor,
                             g: torch.Tensor,
                             hparams: Dict[str, Any]) -> torch.Tensor:
    """∇_z log q(G | z) via autograd."""
    logits  = scores(z, hparams)
    log_q   = -F.binary_cross_entropy_with_logits(logits, g, reduction='sum')
    score_z = torch.autograd.grad(log_q, z, create_graph=True)[0]     # keep graph!
    return score_z


def grad_z_score_autograd_softmax(z: torch.Tensor,
                          data: Dict[str, Any],
                          theta: torch.Tensor,
                          hparams: Dict[str, Any]) -> torch.Tensor:
    """
    REINFORCE gradient with *autograd* score term and soft-max (self-normalised)
    weights for numerical stability.
    """
    # ---- z-prior ----------------------------------------------------------
    grad_z_prior = -(z / hparams['sigma_z'] ** 2)

    # ---- Monte-Carlo over graphs -----------------------------------------
    S           = hparams.get('n_mc_samples', 64)
    log_js      = []
    score_js    = []

    for _ in range(S):
        G        = torch.bernoulli(soft_gmat(z, hparams))

        # 1. log joint  log p(D, Θ | G)
        log_j    = (log_full_likelihood(data, G, theta, hparams) +
                    log_theta_prior(theta * G, hparams['theta_prior_sigma']))
        log_js.append(log_j)

        # 2. score term  ∇_z log q(G | z)
        score_js.append(score_autograd_g_given_z(z, G, hparams))

    log_js   = torch.stack(log_js)                # (S,)
    scores   = torch.stack(score_js)              # (S, d, k, 2)

    # ---- soft-max weights  w_s ∝ p(D,Θ|G_s) ------------------------------
    w = torch.softmax(log_js, dim=0)              # (S,)
    while w.dim() < scores.dim():                 # broadcast to score shape
        w = w.unsqueeze(-1)

    grad_like = (w * scores).sum(dim=0)           # (d, k, 2)

    # ---- acyclicity term --------------------------------------------------
    grad_acyc = score_acyclic_constr_mc(z, hparams)

    # ---- total ------------------------------------------------------------
    return grad_z_prior + grad_like - hparams['beta'] * grad_acyc


def grad_z_score_autograd_stable(z, data, theta, hparams):
    d         = z.size(0)
    S         = hparams.get('n_mc_samples', 64)
    sigma_z   = hparams['sigma_z']
    beta      = hparams['beta']

    # ---- prior term ---------------------------------------------------------
    grad_z  = -z / sigma_z**2                     # (d, k, 2)

    # ---- sample graphs ------------------------------------------------------
    with torch.no_grad():
        edge_probs = torch.sigmoid(scores(z, hparams))   # (d, d)
    g_samples = torch.bernoulli(edge_probs.expand(S, -1, -1))  # (S, d, d)

    # ---- score ∇_z log q(G|z) for every sample ------------------------------
    score_z   = torch.stack([
        score_autograd_g_given_z(z, g, hparams) for g in g_samples
    ])                                           # (S, d, k, 2)

    # ---- log joint reward ---------------------------------------------------
    with torch.no_grad():
        log_r = torch.stack([
            log_full_likelihood(data, g, theta, hparams) +
            log_theta_prior(theta * g, hparams['theta_prior_sigma'])
            for g in g_samples
        ])                                       # (S,)

    # ---- likelihood gradient via soft-max weights ---------------------------
    w = torch.softmax(log_r, dim=0)              # (S,)
    grad_lik = (w.view(-1,1,1,1) * score_z).sum(dim=0)

    # ---- acyclicity term ----------------------------------------------------
    with torch.no_grad():
        r_cyc = torch.stack([acyclic_constr(g, d) for g in g_samples])  # (S,)
    grad_cyc = (r_cyc.view(-1,1,1,1) * score_z).mean(dim=0)

    # ---- total --------------------------------------------------------------
    return grad_z + grad_lik - beta * grad_cyc


def grad_z_score_autograd(z: torch.Tensor, data: Dict[str, Any], theta: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    """
    Computes ∇_z log p(z|D,Θ) using the score function estimator,
    but calculates the score term `∇_z log q(g|z)` using torch.autograd.
    """
    d = z.shape[0]
    n_samples = hparams.get('n_mc_samples', 64)

    # --- Gradient of the Priors (Unaffected by the score function) ---
    # Gradient of log p(z)
    grad_z_prior = - (z / hparams['sigma_z']**2)

    # --- Score Function Estimation Part ---
    total_likelihood_score = 0
    total_acyclicity_score = 0
    
    # Pre-compute probabilities to sample from
    with torch.no_grad():
        edge_probs = torch.sigmoid(scores(z, hparams))

    log_density_samples = []
    score_samples = []
    for _ in range(n_samples):
        # 1. Sample a hard graph G. This operation is non-differentiable.
        g_hard = torch.bernoulli(edge_probs)

        # 2. Calculate the score `∇_z log q(g|z)` using our new autograd function.
        # This is the "manual gradient" part that we are automating.
        score = score_autograd_g_given_z(z, g_hard, hparams)

        # 3. Calculate the "rewards" for this sample G. These are treated as constants w.r.t. z.
        with torch.no_grad():
            # Likelihood and Theta prior reward
            log_joint_reward = (log_full_likelihood(data, g_hard, theta, hparams) + 
                                log_theta_prior(theta * g_hard, hparams.get('theta_prior_sigma')))
            
            # Acyclicity constraint reward
            acyclicity_reward = acyclic_constr(g_hard, d)
        
        # 4. Multiply rewards by the score and accumulate.
        log_density_samples.append(log_joint_reward)
        score_samples.append(score)
        total_acyclicity_score += acyclicity_reward * score

    # grad acyclicity  normal avg
    grad_acyclicity = total_acyclicity_score / n_samples
    
    # grad likelihood soft-max
    log_density_samples = torch.stack(log_density_samples)
    score_samples = torch.stack(score_samples)
    w = torch.softmax(log_density_samples, dim=0)
    grad_likelihood = (w.view(-1,1,1,1) * score_samples).sum(dim=0)

    # Combine all gradient components for z
    total_grad = grad_z_prior + grad_likelihood - (hparams['beta'] * grad_acyclicity)

    return total_grad

import torch, math
from typing import Dict, Any

# ---------- helper: SciPy-style logsumexp with sign ----------
def logsumexp_with_sign(a: torch.Tensor,
                        b: torch.Tensor,
                        dim: int = -1,
                        keepdim: bool = False,
                        eps: float = 1e-40):
    """
    Stable log|Σ b·exp(a)|  and  sign(Σ b·exp(a))

    Parameters
    ----------
    a : Tensor   – log-terms  (broadcastable to `b`)
    b : Tensor   – weights (can be ±)
    dim : int    – reduction dimension
    keepdim : bool – keep reduced dimension
    """
    a_max = torch.max(a, dim=dim, keepdim=True)[0]            # shift
    scaled = b * torch.exp(a - a_max)                         # safe range
    s      = scaled.sum(dim=dim, keepdim=keepdim)             # may be ±
    sign   = torch.sign(s).detach()                           # ±1 (no grad)
    logabs = torch.log(torch.abs(s) + eps) + a_max.squeeze(dim) \
             if not keepdim else torch.log(torch.abs(s) + eps) + a_max
    return logabs, sign
# -------------------------------------------------------------


def grad_z_score_stable_sign(z: torch.Tensor,
                             data: Dict[str, Any],
                             theta: torch.Tensor,
                             hparams: Dict[str, Any]) -> torch.Tensor:
    """
    Stable score-function estimator for ∇_z log p(z | D, Θ) that keeps
    track of the sign of the numerator (needed when individual score
    components can be negative).
    """
    d          = z.shape[0]
    K          = hparams.get('n_mc_samples', 64)

    # ---- 1.  prior gradient (∇ log p(z)) --------------------
    grad_z_prior = - z / hparams['sigma_z']**2

    # ---- 2.  Monte-Carlo loop ------------------------------
    with torch.no_grad():
        edge_probs = torch.sigmoid(scores(z, hparams))        # Bernoulli probs

    log_lik   = []        # ℓ_j
    score_raw = []        # b_j  (tensor shaped like z)
    tot_acycl = 0.0

    for _ in range(K):
        g_hard = torch.bernoulli(edge_probs)
        #   score w.r.t. z for *this* graph
        score   = score_autograd_g_given_z(z, g_hard, hparams)

        with torch.no_grad():
            ll = (log_full_likelihood(data, g_hard, theta, hparams) +
                  log_theta_prior(theta * g_hard,
                                  hparams.get('theta_prior_sigma')))
            h  = acyclic_constr(g_hard, d)

        log_lik.append(ll)
        score_raw.append(score)
        tot_acycl += h * score        # accumulate β∇h

    # ------------ stack tensors ------------------------------
    log_lik   = torch.stack(log_lik)            # [K]
    score_raw = torch.stack(score_raw)          # [K, *z.shape]

    # acyclicity term (simple average)
    grad_acycl = tot_acycl / K

    # ------------ stable numerator / denominator -------------
    flat_dim   = score_raw[0].numel()           # d*k*2
    scores_f   = score_raw.view(K, flat_dim).T  # [flat_dim, K]
    log_lik_b  = log_lik.unsqueeze(0)           # broadcast: [1,K] → [flat_dim,K]

    log_num, sign = logsumexp_with_sign(log_lik_b, scores_f, dim=1)  # per coord
    log_den       = torch.logsumexp(log_lik, dim=0)                  # scalar

    # N/D   (K cancels, but keep it to mirror JAX algebra)
    grad_lik_flat = sign * torch.exp(log_num - math.log(K) -
                                     log_den + math.log(K))

    grad_lik = grad_lik_flat.view_as(score_raw[0])        # [d,k,2]

    # ------------ final combined gradient --------------------
    beta      = hparams['beta']
    total_grad = grad_z_prior + grad_lik - beta * grad_acycl

    return total_grad


def analytic_score_g_given_z(z, g, hparams):
    # 1. logits and probabilities
    probs = soft_gmat(z, hparams)
    diff   = g - probs                 # (g_ij − σ(s_ij))
    u, v   = z[..., 0], z[..., 1]      # (d,k)

    # 2. gradients wrt u and v
    grad_u = hparams['alpha'] * torch.einsum('ij,jk->ik', diff, v)   # (d,k)
    grad_v = hparams['alpha'] * torch.einsum('ij,ik->jk', diff, u)   # (d,k)

    return torch.stack([grad_u, grad_v], dim=-1)          # (d,k,2)



def score_acyclic_constr_mc(z: torch.Tensor,  hparams: Dict[str, Any]) -> torch.Tensor:
    """
    score estimator for the acyclicity constraint
        
    Returns:
        A tuple containing:
        - h_g (torch.Tensor): The acyclicity values for each sample [n_samples].
        - log_prob_g (torch.Tensor): The log probability of each sample [n_samples].
    """
    d = z.shape[0]
    n_samples = hparams.get('n_mc_samples', 64)
    
    # 1. Sample hard graphs
    g_samples = [torch.bernoulli(soft_gmat(z, hparams)) for _ in range(n_samples)]

    # 2. Calculate the "reward" h(G) for each sample
    total_acyl_score = 0
    for g in g_samples:
        reward = acyclic_constr(g, d)
        score = analytic_score_g_given_z(z, g, hparams)
        total_acyl_score += reward * score
    
    return total_acyl_score / n_samples


def grad_z_score_softmax(z: torch.Tensor,
                 data: Dict[str, Any],
                 theta: torch.Tensor,
                 hparams: Dict[str, Any]) -> torch.Tensor:
    # ---- acyclicity & z-prior --------------------------------------------
    grad_acyc    = score_acyclic_constr_mc(z, hparams)
    grad_z_prior = -(z / hparams['sigma_z'] ** 2)
    grad_prior   = grad_z_prior - hparams['beta'] * grad_acyc

    # ---- likelihood / θ-prior part ---------------------------------------
    S          = hparams.get('n_mc_samples', 64)
    log_j_list = []
    score_list = []

    for _ in range(S):
        G       = torch.bernoulli(soft_gmat(z, hparams))
        log_j   = (log_full_likelihood(data, G, theta, hparams) +
                   log_theta_prior(theta * G, hparams['theta_prior_sigma']))
        score   = analytic_score_g_given_z(z, G, hparams)

        log_j_list.append(log_j)
        score_list.append(score)

    log_j   = torch.stack(log_j_list)           # (S,)
    scores  = torch.stack(score_list)           # (S,d,k,2)

    # softmax in log-space  -> normalised positive weights that sum to 1
    w       = torch.softmax(log_j, dim=0)       # (S,)

    # broadcast weights up to scores’ shape
    while w.dim() < scores.dim():
        w = w.unsqueeze(-1)

    grad_like = (w * scores).sum(dim=0)         # (d,k,2)

    # ---- total -----------------------------------------------------------
    return grad_prior + grad_like

def grad_z_score(z: torch.Tensor, data: Dict[str, Any], theta: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    
    # acylic + prior + likelihood

    # acylic grad
    grad_acylic = score_acyclic_constr_mc(z, hparams)
    grad_z_prior = - (z / hparams['sigma_z']**2)

    grad_prior = grad_z_prior - ( hparams['beta'] * grad_acylic)


    # likelihood grad with score estimator
    n_samples = hparams.get('n_mc_samples', 64)
    total_likelihood_score = 0
    for _ in range(n_samples):
        g_hard = torch.bernoulli(soft_gmat(z, hparams))
        ## what should go here?  the full likelihood as the reward ?
        
        log_joint_reward  = log_full_likelihood(data, g_hard, theta, hparams)  + log_theta_prior(theta * g_hard, hparams.get('theta_prior_sigma'))
        print(f"log_joint_reward: {log_joint_reward}")    
        joint_reward = torch.exp(log_joint_reward)
        print(f"joint_reward: {joint_reward}")

        score = analytic_score_g_given_z(z, g_hard, hparams)

        total_likelihood_score += joint_reward * score
    grad_likelihood = total_likelihood_score / n_samples

    total_grad = grad_prior + grad_likelihood

    return total_grad

def grad_theta_score(z: torch.Tensor, data: Dict[str, Any], theta: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    n_samples = hparams.get('n_mc_samples', 64)
    
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

def grad_theta_score_stable_sign(z:      torch.Tensor,
                                 data:   Dict[str, Any],
                                 theta:  torch.Tensor,
                                 hparams: Dict[str, Any]) -> torch.Tensor:
    """
    Sign-stable score-function estimator for ∇_Θ log p(z,Θ | D).

    * Draw K graphs   G₁…G_K  ~  p(G | z)        (hard Bernoulli here)
    * For each graph compute
        ℓ_j  = log p(D,Θ | G_j)                  (log-likelihood term)
        b_j  = ∇_Θ ℓ_j                           (gradient wrt Θ)
    * Return   E_{p(G|z,D)} [ b_j ]
              = Σ b_j · e^{ℓ_j}  /  Σ e^{ℓ_j},
      with all numerator components evaluated stably in log-space.
    """
    K = hparams.get('n_mc_samples', 64)
    d = theta.size(0)

    log_lik_list, grad_list = [], []

    for _ in range(K):
        # ---- 1. sample hard graph ----------------------------------------
        with torch.no_grad():
            G = torch.bernoulli(soft_gmat(z, hparams))

        # ---- 2. log-density  ℓ_j  ----------------------------------------
        theta_tmp = theta.clone().requires_grad_(True)
        log_j = (log_full_likelihood(data, G, theta_tmp, hparams) +
                 log_theta_prior(theta_tmp * G,
                                 hparams['theta_prior_sigma']))
        # ---- 3. gradient wrt Θ   b_j  ------------------------------------
        grad_theta_j, = torch.autograd.grad(log_j, theta_tmp,
                                            retain_graph=False,
                                            create_graph=False)
        log_lik_list.append(log_j.detach())      # scalar
        grad_list.append(grad_theta_j.detach())  # (d,d)

    log_lik   = torch.stack(log_lik_list)        # (K,)
    grads_raw = torch.stack(grad_list)           # (K,d,d)

    # ---------- sign-stable weighted average ------------------------------
    flat_dim  = grads_raw[0].numel()             # d*d
    grads_f   = grads_raw.view(K, flat_dim).T    # (flat_dim, K)
    log_b     = log_lik.unsqueeze(0)             # (1,K) broadcast

    log_num, sign = logsumexp_with_sign(log_b, grads_f, dim=1)  # per coord
    log_den       = torch.logsumexp(log_lik, dim=0)             # scalar

    grad_theta_flat = sign * torch.exp(log_num - log_den)       # N / D
    grad_theta      = grad_theta_flat.view_as(theta)            # (d,d)

    return grad_theta



def grad_log_joint(params: Dict[str, torch.Tensor], data: Dict[str, Any], hparams: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A top-level function that orchestrates the calculation of all gradients.
    """
    #grad_z = grad_z_score(params["z"], data, params["theta"].detach(), hparams)
    #grad_z = grad_z_score_softmax(params["z"], data, params["theta"].detach(), hparams)

    grad_z = grad_z_score_stable_sign(params["z"], data, params["theta"].detach(), hparams)
    #grad_z = grad_z_score_autograd_softmax(params["z"], data, params["theta"].detach(), hparams)
    #grad_z = grad_z_score_autograd_stable(params["z"], data, params["theta"].detach(), hparams)

    #grad_th = grad_theta_score(params["z"].detach(), data, params["theta"], hparams)
    grad_th = grad_theta_score_stable_sign(params["z"].detach(), data, params["theta"], hparams)
    return grad_z, grad_th

def log_joint(params: Dict[str, Any], data: Dict[str, Any], hparams: Dict[str, Any]) -> torch.Tensor:
    """
    Computes the value of the unnormalized log-joint probability for debugging.
    This function combines the different log-probability terms.
    """
    z = params["z"]
    theta = params["theta"]
    g_soft = soft_gmat(z, hparams)
    # For simplicity, we can approximate the expected likelihood with the soft graph
    log_likelihood_val = log_full_likelihood(data, g_soft, theta, hparams)
    log_theta_prior_val = log_theta_prior(theta, hparams.get('sigma_theta_prior', 1.0))
    # The acyclicity constraint is harder to evaluate without sampling, so we can skip it for a rough estimate
    # or use the soft graph version
    with torch.no_grad():
        n_samples = hparams.get('n_mc_samples', 64)
        h_vals = []
        for _ in range(n_samples):
            g_hard = torch.bernoulli(soft_gmat(z, hparams))
            d = z.shape[0]
            h_vals.append(acyclic_constr(g_hard, d))
    acyclicity_val = torch.mean(torch.stack(h_vals))

    # Prior on Z
    log_z_prior = -0.5 * torch.sum(z**2) / hparams.get('latent_prior_std', 1.0)**2
    log_joint = log_likelihood_val + log_theta_prior_val + log_z_prior - hparams['beta'] * acyclicity_val
    logging.info(f"Log-joint: {log_joint.item():.4f}, "
                 f"Log-Likelihood: {log_likelihood_val.item():.4f}, "
                 f"Log-Z-Prior: {log_z_prior.item():.4f}, "
                 f"Log-Theta-Prior: {log_theta_prior_val.item():.4f}, "
                 f"penalty acylic: {-hparams['beta'] * acyclicity_val.item():.4f}, "
                 f"Acyclicity: {acyclicity_val.item():.4f}")
    return log_joint

def update_dibs_hparams(hparams: Dict[str, Any], t: int) -> Dict[str, Any]:
    """
    Handles annealing schedules for hyperparameters.
    """
    # Simple linear annealing
    hparams['alpha'] = hparams['alpha_base'] * t *0.2 
    hparams['beta'] = hparams['beta_base'] * t
    hparams['current_t'] = t
    return hparams





# 2. DATA GENERATION
# =============================================================================

def generate_ground_truth_chain_data(num_samples, chain_length, obs_noise_std):
    """Generates data for a simple causal chain: X1 -> X2 -> ... -> Xn."""
    if chain_length < 2:
        raise ValueError("Chain length must be at least 2")
    
    G_true = torch.zeros(chain_length, chain_length, dtype=torch.float32)
    for i in range(chain_length - 1):
        G_true[i, i + 1] = 1.0

    Theta_true = torch.zeros(chain_length, chain_length, dtype=torch.float32)
    for i in range(chain_length - 1):
        if i==0:
            Theta_true[i, i + 1] = 2
        elif i==1:
            Theta_true[i, i + 1] = -1.5
        else:
            Theta_true[i, i + 1] = (torch.rand(1).item() - 0.5) * 4.0 # Random weight in [-2, 2]
    
    X_data = torch.zeros(num_samples, chain_length)
    X_data[:, 0] = torch.randn(num_samples) * obs_noise_std
    
    for i in range(1, chain_length):
        parent_value = X_data[:, i - 1]
        weight = Theta_true[i - 1, i]
        noise = torch.randn(num_samples) * obs_noise_std
        X_data[:, i] = weight * parent_value + noise
        
    return X_data, G_true, Theta_true

def generate_ground_truth_erdos_renyi_data(num_samples, n_nodes, p_edge, obs_noise_std):
    """Generates data from an Erdős-Rényi DAG structure."""
    if n_nodes < 2:
        raise ValueError("Number of nodes must be at least 2")
    
    g = ig.Graph.Erdos_Renyi(n=n_nodes, p=p_edge, directed=True)
    while not g.is_dag():
        g = ig.Graph.Erdos_Renyi(n=n_nodes, p=p_edge, directed=True)

        adj_matrix = torch.tensor(np.array(g.get_adjacency().data), dtype=torch.float32)
    G_true = adj_matrix
    
    # Initialize weight matrix
    Theta_true = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    
    # Fill random weights for existing parent-child relationships
    edge_indices = (G_true == 1).nonzero(as_tuple=True)
    num_edges = len(edge_indices[0])
    
    if num_edges > 0:
        # Generate random weights between -2 and 2 for existing edges
        edge_weights = (torch.rand(num_edges) - 0.5) * 4.0
        Theta_true[edge_indices] = edge_weights
    
    # Generate data following the DAG structure
    X_data = torch.zeros(num_samples, n_nodes)
    
    # Generate data in topological order (i before j for edges i->j)
    for j in range(n_nodes):
        # Find parent nodes (nodes with edges pointing to j)
        parent_indices = (G_true[:, j] == 1).nonzero(as_tuple=True)[0]
        noise = torch.randn(num_samples) * obs_noise_std
        
        if len(parent_indices) == 0:
            # Root node: just noise
            X_data[:, j] = noise
        else:
            # Sum weighted contributions from parents
            parent_contribution = torch.zeros(num_samples)
            for parent_idx in parent_indices:
                weight = Theta_true[parent_idx, j]
                parent_contribution += weight * X_data[:, parent_idx]
            X_data[:, j] = parent_contribution + noise
    
    return X_data, G_true, Theta_true
    
def generate_ground_truth_scale_free_data(num_samples, n_nodes, m_edges, obs_noise_std):
    """Generates data from a Scale-Free DAG structure using preferential attachment."""
    if n_nodes < 2:
        raise ValueError("Number of nodes must be at least 2")
    if m_edges >= n_nodes:
        raise ValueError("m_edges must be less than n_nodes")
    
    # Initialize adjacency and weight matrices
    G_true = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    Theta_true = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    
    # Build scale-free DAG using preferential attachment
    # Start with the first m_edges+1 nodes forming a small DAG
    for i in range(min(m_edges + 1, n_nodes - 1)):
        for j in range(i + 1, min(m_edges + 1, n_nodes)):
            if torch.rand(1).item() < 0.5:  # 50% chance for initial connections
                G_true[i, j] = 1.0
                Theta_true[i, j] = (torch.rand(1).item() - 0.5) * 4.0
    
    # Add remaining nodes with preferential attachment
    for new_node in range(m_edges + 1, n_nodes):
        # Calculate in-degrees for preferential attachment
        in_degrees = G_true.sum(dim=0)  # Sum over rows gives in-degree
        in_degrees[:new_node] += 1  # Add 1 to avoid zero probabilities
        
        # Normalize to get probabilities
        probs = in_degrees[:new_node] / in_degrees[:new_node].sum()
        
        # Select m_edges nodes to connect to (from existing nodes)
        num_connections = min(m_edges, new_node)
        
        # Sample connections based on preferential attachment
        for _ in range(num_connections):
            # Choose a node to connect from (preferential attachment)
            source_node = torch.multinomial(probs, 1, replacement=False).item()
            
            if G_true[source_node, new_node] == 0:  # Avoid duplicate edges
                G_true[source_node, new_node] = 1.0
                Theta_true[source_node, new_node] = (torch.rand(1).item() - 0.5) * 4.0
                
            # Update probabilities to reflect new connection
            probs[source_node] = 0  # Remove this node from future selection
            if probs.sum() > 0:
                probs = probs / probs.sum()  # Renormalize
    
    # Generate data following the DAG structure
    X_data = torch.zeros(num_samples, n_nodes)
    
    # Generate data in topological order
    for j in range(n_nodes):
        # Find parent nodes
        parent_indices = (G_true[:, j] == 1).nonzero(as_tuple=True)[0]
        noise = torch.randn(num_samples) * obs_noise_std
        
        if len(parent_indices) == 0:
            # Root node: just noise
            X_data[:, j] = noise
        else:
            # Sum weighted contributions from parents
            parent_contribution = torch.zeros(num_samples)
            for parent_idx in parent_indices:
                weight = Theta_true[parent_idx, j]
                parent_contribution += weight * X_data[:, parent_idx]
            X_data[:, j] = parent_contribution + noise
    
    return X_data, G_true, Theta_true

# =============================================================================
# 3. CONFIGURATION
# =============================================================================

class Config:
    seed = 31
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Data Generation ---
    # Choose data source: 'simple_chain', 'erdos_renyi', 'scale_free'
    data_source = 'simple_chain'
    
    # --- Data Parameters ---
    d_nodes = 3
    num_samples = 100
    obs_noise_std = 0.1
    
    # Parameters for 'simple_chain'
    chain_length = d_nodes
    
    # Parameters for 'erdos_renyi'
    p_edge = 0.6  # Probability of edge existence
    
    # Parameters for 'scale_free'
    m_edges = 2  # Number of edges to attach from each new node

    # --- Model ---
    k_latent = d_nodes
    alpha_base = 0.2  # Base value for annealing
    beta_base = 1.0   # Base value for annealing
    theta_prior_sigma = 1.0
    
    # --- MC Sampling ---
    n_mc_samples = 128

    # --- Training ---
    lr = 5e-3
    num_iterations = 1500
    debug_print_iter = 100

cfg = Config()

# =============================================================================
# 4. TRAINING LOOP
# =============================================================================
def debug_prediction_error(
    params: Dict[str, torch.Tensor], 
    data: Dict[str, Any], 
    hparams: Dict[str, Any],
    G_true: torch.Tensor
) -> None:
    """
    Calculates and logs key metrics related to prediction accuracy and the error penalty.
    """
    log.info("--- Prediction Error Debug ---")
    
    # Ensure we are not tracking gradients here
    with torch.no_grad():
        # --- 1. Get the model's current best estimate of the graph and weights ---
        z, theta = params["z"], params["theta"]
        x_data = data['x']
        sigma_obs = hparams['sigma_obs_noise']

        # Get the hard graph the model is currently predicting
        g_probs = soft_gmat(z, hparams)
        g_learned = torch.bernoulli(g_probs)
        
        # Calculate the effective weights based on the learned graph
        W_learned = theta * g_learned
        
        # --- 2. Make predictions based on the learned model ---
        pred_mean = torch.matmul(x_data, W_learned)
        
        # --- 3. Calculate error metrics ---
        residuals = x_data - pred_mean
        mse = torch.mean(residuals**2)  
        mae = torch.mean(torch.abs(residuals))
        
        # --- 4. Calculate the Error Penalty directly ---
        # This should match the value we derived from the log-likelihood
        total_error_penalty = torch.sum(residuals**2) / (2 * sigma_obs**2)
        
        # --- 5. Compare learned graph to ground truth ---
        shd = torch.sum(torch.abs(g_learned.cpu() - G_true.cpu()))

        log.info(f"SHD: {shd.item():.1f} | MSE: {mse.item():.6f} | MAE: {mae.item():.6f}")
        log.info(f"Target MSE (noise variance σ^2): {sigma_obs**2:.6f}")
        log.info(f"Calculated Total Error Penalty: {total_error_penalty.item():.2f}")

def main():
    # --- Setup ---
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    log.info(f"Running on device: {cfg.device}")

    # --- Data Generation ---
    log.info(f"Using data source: {cfg.data_source}")
    
    if cfg.data_source == 'simple_chain':
        data_x, G_true, Theta_true = generate_ground_truth_chain_data(
            num_samples=cfg.num_samples,
            chain_length=cfg.chain_length,
            obs_noise_std=cfg.obs_noise_std
        )
        log.info(f"Generated simple chain data with {cfg.chain_length} nodes.")
        
    elif cfg.data_source == 'erdos_renyi':
        data_x, G_true, Theta_true = generate_ground_truth_erdos_renyi_data(
            num_samples=cfg.num_samples,
            n_nodes=cfg.d_nodes,
            p_edge=cfg.p_edge,
            obs_noise_std=cfg.obs_noise_std
        )
        log.info(f"Generated Erdős-Rényi data with {cfg.d_nodes} nodes and p_edge={cfg.p_edge}.")
        
    elif cfg.data_source == 'scale_free':
        data_x, G_true, Theta_true = generate_ground_truth_scale_free_data(
            num_samples=cfg.num_samples,
            n_nodes=cfg.d_nodes,
            m_edges=cfg.m_edges,
            obs_noise_std=cfg.obs_noise_std
        )
        log.info(f"Generated Scale-Free data with {cfg.d_nodes} nodes and m_edges={cfg.m_edges}.")
        
    else:
        raise ValueError(f"Unknown data_source: {cfg.data_source}. "
                        f"Choose from: 'simple_chain', 'erdos_renyi', 'scale_free'")
    
    data = {'x': data_x.to(cfg.device)}
    log.info(f"Ground truth adjacency matrix:\n{G_true}")
    log.info(f"Ground truth weights matrix:\n{Theta_true}")
    
    # Count the number of edges in the true graph
    num_edges = (G_true > 0).sum().item()
    log.info(f"Number of edges in ground truth graph: {num_edges}")

    # --- Initialization ---
    particle = {
        "z": nn.Parameter(torch.randn(cfg.d_nodes, cfg.k_latent, 2, device=cfg.device)),
        "theta": nn.Parameter(torch.randn(cfg.d_nodes, cfg.d_nodes, device=cfg.device)),
    }
    
    sigma_z = 1.0 / torch.sqrt(torch.tensor(cfg.k_latent))
    hparams = {
        "alpha": 0.0, # Will be annealed
        "beta": 0.0,  # Will be annealed
        "alpha_base": cfg.alpha_base,
        "beta_base": cfg.beta_base,
        "sigma_z": sigma_z,
        "sigma_obs_noise": cfg.obs_noise_std,
        "theta_prior_sigma": cfg.theta_prior_sigma,
        "n_mc_samples": cfg.n_mc_samples,
        "total_steps": cfg.num_iterations
    }

    optimizer = torch.optim.RMSprop(particle.values(), lr=cfg.lr)

    # --- Gradient Ascent Loop ---
    log.info("Starting training...")
    for t in range(1, cfg.num_iterations + 1):
        optimizer.zero_grad()
        
        # Update annealed hyperparameters
        hparams = update_dibs_hparams(hparams, t)
        
        # Get gradients of the log-joint
        grad_z, grad_th = grad_log_joint(particle, data, hparams)
        
        # Assign gradients for gradient ASCENT (optimizers perform descent)
        particle['z'].grad = -grad_z
        particle['theta'].grad = -grad_th

        optimizer.step()

        # --- Logging ---
        if t % cfg.debug_print_iter == 0 or t == cfg.num_iterations:
            debug_prediction_error(particle, data, hparams, G_true) 


            with torch.no_grad():
                lj_val = log_joint(particle, data, hparams).item()
                grad_z_norm = torch.linalg.norm(grad_z).item()
                grad_theta_norm = torch.linalg.norm(grad_th).item()
                edge_probs = soft_gmat(particle['z'], hparams)

                log.info(f"--- Iter {t}/{cfg.num_iterations} ---")
                log.info(f"log_joint={lj_val:.2f} | grad_Z_norm={grad_z_norm:.2e} | grad_Theta_norm={grad_theta_norm:.2e}")
                log.info(f"Annealed: alpha={hparams['alpha']:.3f}, beta={hparams['beta']:.3f}")
                log.info(f"Current Edge Probs (rounded):\n{np.round(edge_probs.cpu().numpy(), 2)}")

    log.info("Training finished.")
    with torch.no_grad():
        final_probs = soft_gmat(particle['z'], hparams)
        
        # Create hard graph from learned probabilities (threshold at 0.5)
        learned_hard_graph = (final_probs > 0.5).float()
        
        log.info(f"\n{'='*60}")
        log.info(f"FINAL RESULTS COMPARISON")
        log.info(f"{'='*60}")
        log.info(f"Ground Truth Adjacency Matrix:\n{G_true}")
        log.info(f"Ground Truth Weights Matrix:\n{Theta_true}")
        log.info(f"\nLearned Edge Probabilities:\n{final_probs.cpu().numpy()}")
        log.info(f"Learned Hard Graph (threshold=0.5):\n{learned_hard_graph.cpu().numpy()}")
        log.info(f"Learned Theta:\n{particle['theta'].cpu().numpy()}")
        
        # Compute structural difference
        structural_diff = torch.abs(G_true - learned_hard_graph).sum().item()
        log.info(f"\nStructural Hamming Distance: {structural_diff}")
        
        # Show effective learned weights (G_learned * Theta_learned)
        effective_learned_weights = learned_hard_graph * particle['theta'].cpu()
        effective_true_weights = G_true * Theta_true
        log.info(f"\nGround Truth Effective Weights (G_true * Theta_true):\n{effective_true_weights}")
        log.info(f"Learned Effective Weights (G_learned * Theta_learned):\n{effective_learned_weights}")


if __name__ == '__main__':
    main()
