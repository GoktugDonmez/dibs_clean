import torch
from torch.distributions import Normal
import numpy as np
import logging
from typing import Dict, Any, Tuple
import torch.nn as nn
import igraph as ig  

DEBUG_PRINT_ITER = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

def acyclic_constr(g: torch.Tensor, d: int) -> torch.Tensor:
    alpha = 1.0 / d
    eye = torch.eye(d, device=g.device, dtype=g.dtype)
    m = eye + alpha * g
    # The matrix power operation is a differentiable way to check for cycles.
    return torch.trace(torch.linalg.matrix_power(m, d)) - d


def log_gaussian_likelihood(x: torch.Tensor, pred_mean: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    sigma_tensor = torch.tensor(sigma, dtype=pred_mean.dtype, device=pred_mean.device)
    residuals = x - pred_mean
    log_prob = -0.5 * (torch.log(2 * torch.pi * sigma_tensor**2)) - 0.5 * ((residuals / sigma_tensor)**2)
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


def stable_ratio(grad_samples, log_density_samples):
    eps = 1e-30
    S   = len(log_density_samples)                    # == M

    log_p   = torch.stack(log_density_samples)        # [S]
    grads   = torch.stack(grad_samples)               # [S,*]

    while log_p.dim() < grads.dim():
        log_p = log_p.unsqueeze(-1)

    log_den = torch.logsumexp(log_p, dim=0) - torch.log(torch.tensor(len(log_p), dtype=log_p.dtype, device=log_p.device))

    pos = grads >= 0
    neg = ~pos

    log_num_pos = torch.logsumexp(
        torch.where(pos,
                    torch.log(grads.abs() + eps) + log_p,
                    torch.full_like(log_p, -float('inf'))),
        dim=0) - torch.log(torch.tensor(len(log_p), dtype=log_p.dtype, device=log_p.device))

    log_num_neg = torch.logsumexp(
        torch.where(neg,
                    torch.log(grads.abs() + eps) + log_p,
                    torch.full_like(log_p, -float('inf'))),
        dim=0) - torch.log(torch.tensor(len(log_p), dtype=log_p.dtype, device=log_p.device))

    return torch.exp(log_num_pos - log_den) - torch.exp(log_num_neg - log_den)


def gumbel_acyclic_constr_mc(z: torch.Tensor, d: int, hparams: Dict[str, Any]) -> torch.Tensor:
    """
    implement the gumbel_acyclic_constr_mc later
    for now use REINFORCE estimator
    """
    return "skip for now"


def score_func_estimator_stable(data, z, hparams, f, normalized=True):
    # f is a function G, as in Equation B.8
    _n_mc = hparams["n_mc_samples"]
    _soft_gmat = soft_gmat(z, hparams)
    distr = torch.distributions.Bernoulli(_soft_gmat)
    hard_gmats = distr.sample((_n_mc,))

    # log joint densities as many Gs
    log_f = torch.func.vmap(
        f,
        randomness="different",
    )(hard_gmats)

    # grad_Z log p(G|Z) at one G
    score_func = lambda g: torch.autograd.grad(
        torch.distributions.Bernoulli(soft_gmat(z, hparams))
        .log_prob(g)
        .sum(),
        z,
        create_graph=True,
    )[0]

    scores = torch.stack([score_func(g) for g in hard_gmats])

    while log_f.dim() < scores.dim():
        log_f = log_f.unsqueeze(-1)

    log_numerator = torch.logsumexp(
        log_f
        + torch.log(torch.clamp(torch.abs(scores), min=1e-30)) * torch.sign(scores),
        dim=0,
    )
    log_numerator -= torch.log(torch.tensor(len(log_f), dtype=log_f.dtype, device=log_f.device))

    # If the score function estimator has a denominator
    if normalized:
        log_denominator = torch.logsumexp(log_f, dim=0)
        log_denominator -= torch.log(torch.tensor(len(log_f), dtype=log_f.dtype, device=log_f.device))
        result = torch.exp(log_numerator - log_denominator)
    else:
        result = torch.exp(log_numerator)

    return result


def grad_z_score_stable(z: torch.Tensor, data: Dict[str, Any], theta: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    
    grad_z_prior = - (z /hparams['sigma_z'] **2)

    def f_likelihood(g):
        return (log_full_likelihood(data, g, theta, hparams) + log_theta_prior(theta * g, hparams['theta_prior_sigma']))

    grad_likelihood = score_func_estimator_stable(
        data=data,
        z=z,
        hparams=hparams,
        f=f_likelihood,
        normalized=True
    )

    def f_acyclic(g):
        d = z.shape[0]
        return torch.log(acyclic_constr(g, d) + 1e-30)
    
    grad_acyclic = score_func_estimator_stable(
        data=data,
        z=z,
        hparams=hparams,
        f=f_acyclic,
        normalized=False
    )

    if hparams['current_t'] % DEBUG_PRINT_ITER == 0:
        print(f"grad_likelihood: {grad_likelihood}")
        print(f"grad_acyclic: {grad_acyclic}")
        print(f"grad_z_prior: {grad_z_prior}")
        print(f"acyclic  penalty: {-hparams['beta'] * grad_acyclic}")

    total_grad = grad_z_prior + grad_likelihood - hparams['beta'] * grad_acyclic

    return total_grad


def grad_theta_score_stable_ratio(z: torch.Tensor, data: Dict[str, Any], theta: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
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

    return stable_ratio(grad_samples, log_density_samples)    



def grad_log_joint(params: Dict[str, torch.Tensor], data: Dict[str, Any], hparams: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A top-level function that orchestrates the calculation of all gradients.
    """
    # Z

    grad_z = grad_z_score_stable(params["z"], data, params["theta"].detach(), hparams)     #nazaals

    # for THETA
    grad_theta = grad_theta_score_stable_ratio(params["z"].detach(), data, params["theta"], hparams)

    # printing z and theta every 10 iterations
    if hparams['current_t'] % DEBUG_PRINT_ITER == 0:
        print('--------------------------------')
        print(f"z: {params['z']}")
        print(f"gradient of z: {grad_z}")
        print(f"theta: {params['theta']}")
        print(f"gradient of theta: {grad_theta}")
        print("--------------------------------")
    return grad_z, grad_theta

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

    hparams['alpha'] = hparams['alpha_base'] * t
    hparams['beta'] = hparams['beta_base'] * t
    hparams['current_t'] = t
    return hparams

def update_dibs_hparams_v2(hparams: Dict[str, Any], t: int) -> Dict[str, Any]:
    """
    Handles annealing schedules for hyperparameters with a more balanced approach.
    """
    # Calculate the fraction of training completed
    progress = t / hparams['total_steps']

    # --- Anneal alpha ---
    # Start at 0.0 and increase to `alpha_base`
    a_final = 100
    b_final = 1000
    hparams['alpha'] = a_final * progress

    # --- Anneal beta ---
    # Introduce a burn-in period where beta is zero, then ramp up.
    # This gives the model time to learn from the likelihood first.
    burn_in_fraction = 0.25
    if progress < burn_in_fraction:
        hparams['beta'] = 0.0
    else:
        # Scale progress to be 0 -> 1 over the remainder of the training
        adjusted_progress = (progress - burn_in_fraction) / (1 - burn_in_fraction)
        hparams['beta'] = b_final * adjusted_progress
    
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
    alpha_base = 0.02  # Base value for annealing
    beta_base = 1.0   # Base value for annealing
    theta_prior_sigma = 1.0
    
    # --- MC Sampling ---
    n_mc_samples = 256

    # --- Training ---  
    lr = 5e-3
    num_iterations = 150
    debug_print_iter = DEBUG_PRINT_ITER

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
