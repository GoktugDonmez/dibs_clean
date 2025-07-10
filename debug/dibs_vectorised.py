import torch
import numpy as np
import logging
from typing import Dict, Any, Tuple
import torch.nn as nn
import igraph as ig
import torch.nn.functional as F
from torch.func import vmap, grad
import mlflow

# TODO tomorrow:
# - Try different combinations of soft hard gmats, send to triton.
# - Add reparameterization trick
#
# - Lower priority:
#
# - Add log joint function and change the logging 
# - Add mlflow logging
# - check more in detail the erdos renyi and scale free graphs

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
DEBUG_PRINT_ITER = 100

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

class Config:
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_source = 'simple_chain' # 'erdos_renyi' or  'simple_chain'
    d_nodes = 3
    num_samples = 100
    obs_noise_std = 0.1
    chain_length = d_nodes
    p_edge = 0.6
    m_edges = 2
    alpha_base = 0.01
    beta_base = 0.1
    theta_prior_sigma = 1.0
    n_mc_samples = 128
    lr = 5e-3
    num_iterations = 3000
    debug_print_iter = DEBUG_PRINT_ITER
    
    # Reparameterization trick
    #use_reparam_trick = True
    #tau = 1.0 # Temperature for Gumbel-Sigmoid

# =============================================================================
# 1. CORE COMPONENTS
# =============================================================================

def acyclic_constr(g: torch.Tensor) -> torch.Tensor:
    """Computes the acyclicity constraint h(G) = tr(e^{G◦G}) - d."""
    d = g.shape[0]
    alpha = 1.0 / d
    eye = torch.eye(d, device=g.device, dtype=g.dtype)
    m = eye + alpha * g
    return torch.trace(torch.linalg.matrix_power(m, d)) - d

def get_graph_scores(z: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    """Computes the raw scores for edges based on latent embeddings z."""
    u, v = z[..., 0], z[..., 1]
    raw_scores = hparams["alpha"] * (u @ v.T)
    d = z.shape[0]
    diag_mask = 1.0 - torch.eye(d, device=z.device, dtype=z.dtype)
    return raw_scores * diag_mask

def get_soft_gmat(z: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    """Computes the soft adjacency matrix (edge probabilities) from z."""
    scores = get_graph_scores(z, hparams)
    soft_probs = torch.sigmoid(scores)
    d = z.shape[0]
    diag_mask = 1.0 - torch.eye(d, device=z.device, dtype=z.dtype)
    return soft_probs * diag_mask

def get_gumbel_soft_gmat(z: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:

    scores = get_graph_scores(z, hparams)
    u = torch.rand_like(scores)
    L = torch.log(u) - torch.log1p(-u)
    logits = (scores + L) / hparams["tau"] #tau =1
    g_soft = torch.sigmoid(logits)
    d = g_soft.size(-1)
    mask = 1.0 - torch.eye(d, device=z.device, dtype=z.dtype)
    return g_soft * mask

# =============================================================================
# 2. LOG-PROBABILITY FUNCTIONS
# =============================================================================

def log_gaussian_likelihood(x: torch.Tensor, pred_mean: torch.Tensor, sigma: float) -> torch.Tensor:
    """Computes log p(x | pred_mean, sigma) for a Gaussian distribution."""
    sigma_tensor = torch.tensor(sigma, dtype=pred_mean.dtype, device=pred_mean.device)
    residuals = x - pred_mean
    log_prob = -0.5 * torch.log(2 * torch.pi * sigma_tensor**2) - 0.5 * ((residuals / sigma_tensor)**2)
    return torch.sum(log_prob)

def log_likelihood_given_g_and_theta(data: Dict[str, Any], g: torch.Tensor, theta: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    """Computes the log-likelihood of the data given a graph and parameters: log p(D|G,θ)."""
    x_data = data['x']
    effective_W = theta * g
    pred_mean = torch.matmul(x_data, effective_W)
    return log_gaussian_likelihood(x_data, pred_mean, sigma=hparams['sigma_obs_noise'])

def log_prior_theta(g: torch.Tensor, theta: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    """Computes the prior on theta: log p(θ|G)."""
    return log_gaussian_likelihood(theta * g, torch.zeros_like(theta), sigma=hparams['theta_prior_sigma'])

def log_prior_z(z: torch.Tensor) -> torch.Tensor:
    """Computes the regularizer part of the prior on z: log p(z)."""
    d = z.shape[0]
    variance = 1.0 / torch.sqrt(torch.tensor(d, dtype=z.dtype, device=z.device))
    dist = torch.distributions.Normal(0, torch.sqrt(variance))
    return torch.sum(dist.log_prob(z))


def log_prior_z_grad(z: torch.Tensor) -> torch.Tensor:
    """Computes the prior on z: log p(z)."""
    return -(1 / torch.sqrt(torch.tensor(z.shape[0]))) * z

# =============================================================================
# 3. GRADIENT ESTIMATION
# =============================================================================

def _calculate_weighted_score_old(grad_samples: torch.Tensor, log_density_samples: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable computation of an expectation of gradients weighted by normalized probabilities.
    Computes: E_{p(x)}[grad(x)] = sum(p_i * grad_i) / sum(p_i)
    """
    eps = 1e-30

    while log_density_samples.dim() < grad_samples.dim():
        log_density_samples = log_density_samples.unsqueeze(-1)

    log_den = torch.logsumexp(log_density_samples, dim=0) - torch.log(torch.tensor(len(log_density_samples), dtype=log_density_samples.dtype, device=log_density_samples.device))

    pos_grads = torch.where(grad_samples >= 0, grad_samples, 0.)
    neg_grads = torch.where(grad_samples < 0, -grad_samples, 0.)

    log_num_pos = torch.logsumexp(log_density_samples + torch.log(pos_grads + eps), dim=0) - torch.log(torch.tensor(len(log_density_samples), dtype=log_density_samples.dtype, device=log_density_samples.device))
    log_num_neg = torch.logsumexp(log_density_samples + torch.log(neg_grads + eps), dim=0) - torch.log(torch.tensor(len(log_density_samples), dtype=log_density_samples.dtype, device=log_density_samples.device))

    return torch.exp(log_num_pos - log_den) - torch.exp(log_num_neg - log_den)

def _calculate_weighted_score(grad_samples: torch.Tensor, log_density_samples: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable computation that preserves multi-dimensional gradient structure
    while using the positive/negative separation approach.
    """
    eps = 1e-30

    # Broadcast log_density_samples to match grad_samples dimensions
    while log_density_samples.dim() < grad_samples.dim():
        log_density_samples = log_density_samples.unsqueeze(-1)

    # Get the sample count (first dimension)
    n_samples = grad_samples.shape[0]
    
    # Calculate common denominator (only over sample dimension)
    log_denominator = torch.logsumexp(log_density_samples, dim=0) - torch.log(torch.tensor(n_samples, dtype=grad_samples.dtype, device=grad_samples.device))

    # Use torch.where to separate positive and negative gradients (preserves structure)
    pos_grads = torch.where(grad_samples >= 0, grad_samples, 0.)
    neg_grads = torch.where(grad_samples < 0, -grad_samples, 0.)  # Take absolute value
    
    # Count positive and negative elements per matrix position
    pos_mask = (grad_samples >= 0).float()
    neg_mask = (grad_samples < 0).float()
    
    pos_count = pos_mask.sum(dim=0)  # Count per matrix position
    neg_count = neg_mask.sum(dim=0)  # Count per matrix position
    total_count = torch.tensor(n_samples, dtype=grad_samples.dtype, device=grad_samples.device)
    
    # Calculate positive contribution using LSE
    pos_log_terms = torch.where(
        grad_samples >= 0,
        log_density_samples + torch.log(pos_grads + eps),
        torch.tensor(-float('inf'), device=grad_samples.device)
    )
    # Only compute LSE where we have positive elements
    log_numerator_pos = torch.where(
        pos_count > 0,
        torch.logsumexp(pos_log_terms, dim=0) - torch.log(pos_count),
        torch.tensor(-float('inf'), device=grad_samples.device)
    )
    expected_pos = torch.where(
        pos_count > 0,
        (pos_count / total_count) * torch.exp(log_numerator_pos - log_denominator),
        torch.tensor(0.0, device=grad_samples.device)
    )
    
    # Calculate negative contribution using LSE
    neg_log_terms = torch.where(
        grad_samples < 0,
        log_density_samples + torch.log(neg_grads + eps),
        torch.tensor(-float('inf'), device=grad_samples.device)
    )
    # Only compute LSE where we have negative elements
    log_numerator_neg = torch.where(
        neg_count > 0,
        torch.logsumexp(neg_log_terms, dim=0) - torch.log(neg_count),
        torch.tensor(-float('inf'), device=grad_samples.device)
    )
    expected_neg = torch.where(
        neg_count > 0,
        (neg_count / total_count) * torch.exp(log_numerator_neg - log_denominator),
        torch.tensor(0.0, device=grad_samples.device)
    )
    
    return expected_pos - expected_neg

def score_function_estimator(
    params_to_grad: Dict[str, nn.Parameter],
    log_prob_fn,
    g_samples: torch.Tensor
) -> torch.Tensor:
    """
    A general score function estimator for gradients of an expectation, vectorized with vmap.
    Computes ∇ E_{p(G|z)}[f(G, θ)]
    """
    theta = params_to_grad['theta']

    # grad(log_prob_fn, argnums=1) creates a function that computes gradient w.r.t. theta.
    # We vmap this function over g_samples to get per-sample gradients.
    grad_fn = grad(log_prob_fn, argnums=1)
    grad_samples = vmap(grad_fn, in_dims=(0, None))(g_samples, theta)

    # We vmap again to get the log probability values for each sample, detaching theta.
    log_f_samples = vmap(log_prob_fn, in_dims=(0, None))(g_samples, theta.detach())

    return _calculate_weighted_score(grad_samples, log_f_samples)

# =============================================================================
# 4. MAIN GRADIENT COMPUTATION
# =============================================================================

def compute_z_gradient(
    params: Dict[str, nn.Parameter],
    data: Dict[str, Any],
    hparams: Dict[str, Any]
) -> torch.Tensor:
    """Computes ∇_z log p(D, θ, z) using vmap for vectorization."""
    z, theta = params['z'], params['theta']
    g_samples = hparams['g_samples']

    # 1. Gradient of the prior: ∇_z log p(z)
    grad_z_prior = log_prior_z_grad(z)

    # 2. Gradient of the likelihood term: ∇_z E_{p(G|z)}[p(D,θ|G)]
    
    # Define a function to compute log p(G|z) for grad
    log_prob_g_given_z = lambda g, z_param: torch.distributions.Bernoulli(get_soft_gmat(z_param, hparams)).log_prob(g).sum()
    
    # Create a function that computes the gradient of log p(G|z) w.r.t. z
    score_fn_z = grad(log_prob_g_given_z, argnums=1)
    
    # Vectorize the score function over g_samples
    scores = vmap(score_fn_z, in_dims=(0, None))(g_samples, z)

    # Define and vectorize the log-joint function f(G) = log p(D,θ|G)
    log_joint_fn = lambda g: log_likelihood_given_g_and_theta(data, g, theta, hparams) + log_prior_theta(g, theta, hparams)
    log_f_values = vmap(log_joint_fn)(g_samples)
    
    grad_lik = _calculate_weighted_score(scores, log_f_values)

    # 3. Gradient of the acyclicity constraint: ∇_z E_{p(G|z)}[h(G)]
    h_values = vmap(acyclic_constr)(g_samples)
    # The multiplication and mean are already vectorized.
    grad_acyc = torch.mean(h_values.view(-1, 1, 1, 1) * scores, dim=0)

    total_grad = grad_z_prior + grad_lik - hparams['beta'] * grad_acyc
    
    if hparams.get('current_t', 0) % DEBUG_PRINT_ITER == 0:
        log.info(f"Z Grads | Prior: {torch.linalg.norm(grad_z_prior).item():.4f}, "
                 f"Likelihood: {torch.linalg.norm(grad_lik).item():.4f}, "
                 f"Acyclicity: {torch.linalg.norm(grad_acyc).item():.4f}")
    return total_grad

def compute_theta_gradient(
    params: Dict[str, nn.Parameter],
    data: Dict[str, Any],
    hparams: Dict[str, Any]
) -> torch.Tensor:
    """Computes ∇_θ log p(D, θ, z)."""
    log_prob_fn = lambda g, theta: log_likelihood_given_g_and_theta(data, g, theta, hparams) + log_prior_theta(g, theta, hparams)
    
    grad = score_function_estimator(
        params_to_grad={'theta': params['theta']},
        log_prob_fn=log_prob_fn,
        g_samples=hparams['g_samples']
    )
    return grad

# =============================================================================
# 5. DATA GENERATION
# =============================================================================

def generate_ground_truth_chain_data(num_samples, chain_length, obs_noise_std):
    """Generates data for a simple causal chain: X1 -> X2 -> ... -> Xn."""
    G_true = torch.zeros(chain_length, chain_length, dtype=torch.float32)
    for i in range(chain_length - 1):
        G_true[i, i + 1] = 1.0

    Theta_true = torch.zeros(chain_length, chain_length, dtype=torch.float32)
    Theta_true[0, 1] = 2.0
    if chain_length > 2:
        Theta_true[1, 2] = -1.5
    
    # SEM is X = (I - W^T)^-1 @ E
    W_true = G_true * Theta_true
    E = torch.randn(num_samples, chain_length) * obs_noise_std
    I = torch.eye(chain_length)
    X_data = torch.linalg.solve((I - W_true.T), E.T).T
        
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


# =============================================================================
# 6. TRAINING AND EVALUATION
# =============================================================================

class DIBSTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        log.info(f"Running on device: {cfg.device}")

        self.data, self.G_true, self.Theta_true = self._load_data()
        
        self.params = {
            "z": nn.Parameter(torch.randn(cfg.d_nodes, cfg.d_nodes, 2, device=cfg.device)),
            "theta": nn.Parameter(torch.randn(cfg.d_nodes, cfg.d_nodes, device=cfg.device)),
        }
        
        self.hparams = {
            "alpha": 0.0,
            "beta": 0.0,
            "alpha_base": cfg.alpha_base,
            "beta_base": cfg.beta_base,
            "sigma_z": 1.0 / np.sqrt(cfg.d_nodes),
            "sigma_obs_noise": cfg.obs_noise_std,
            "theta_prior_sigma": cfg.theta_prior_sigma,
            "n_mc_samples": cfg.n_mc_samples,
            "total_steps": cfg.num_iterations
        }
        self.optimizer = torch.optim.Adam(self.params.values(), lr=cfg.lr)

    def _load_data(self):
        log.info(f"Using data source: {self.cfg.data_source}")
        if self.cfg.data_source == 'simple_chain':
            data_x, G_true, Theta_true = generate_ground_truth_chain_data(
                num_samples=self.cfg.num_samples,
                chain_length=self.cfg.chain_length,
                obs_noise_std=self.cfg.obs_noise_std
            )
        elif self.cfg.data_source == 'erdos_renyi':
            data_x, G_true, Theta_true = generate_ground_truth_erdos_renyi_data(
                num_samples=self.cfg.num_samples,
                n_nodes=self.cfg.d_nodes,
                p_edge=self.cfg.p_edge,
                obs_noise_std=self.cfg.obs_noise_std
            )
        else:
            raise ValueError(f"Unknown data_source: {self.cfg.data_source}")
        
        data = {'x': data_x.to(self.cfg.device)}
        log.info(f"Ground truth adjacency matrix:\n{G_true}")
        log.info(f"Ground truth weights matrix:\n{Theta_true}")
        return data, G_true, Theta_true

    def _update_hparams(self, t: int):
        # Version 0 : original linear annealing
        self.hparams['alpha'] = self.hparams['alpha_base'] * t
        self.hparams['beta'] = self.hparams['beta_base'] * t

        
        # Version 1:  burn in
        #progress = t / self.hparams['total_steps']
        #self.hparams['alpha'] = 100 * progress
        
        #burn_in_fraction = 0.25
        #if progress < burn_in_fraction:
        #    self.hparams['beta'] = 0.0
        #else:
        #    adjusted_progress = (progress - burn_in_fraction) / (1 - burn_in_fraction)
        #    self.hparams['beta'] = 1000 * adjusted_progress

        
        
        self.hparams['current_t'] = t

    def train(self):
        log.info("Starting training...")
        with mlflow.start_run():
            # Log parameters from the Config object
            config_dict = {key: getattr(self.cfg, key) for key in dir(self.cfg) if not key.startswith('__') and not callable(getattr(self.cfg, key))}
            mlflow.log_params(config_dict)

            for t in range(1, self.cfg.num_iterations + 1):
                self._train_step(t)
                if t % self.cfg.debug_print_iter == 0 or t == self.cfg.num_iterations:
                    self.log_progress(t)
            log.info("Training finished.")
            self.evaluate()

    def _train_step(self, t: int):
        self.optimizer.zero_grad()
        self._update_hparams(t)
        
        g_soft = get_soft_gmat(self.params["z"], self.hparams)
        g_samples = torch.bernoulli(g_soft.expand(self.hparams['n_mc_samples'], -1, -1))
        self.hparams['g_samples'] = g_samples

        # For z gradient, theta is a constant.
        params_for_z = {'z': self.params['z'], 'theta': self.params['theta'].detach()}
        grad_z = compute_z_gradient(params_for_z, self.data, self.hparams)

        # For theta gradient, z is a constant.
        params_for_theta = {'z': self.params['z'].detach(), 'theta': self.params['theta']}
        grad_theta = compute_theta_gradient(params_for_theta, self.data, self.hparams)
        
        # Assign gradients for gradient ASCENT
        self.params['z'].grad = -grad_z
        self.params['theta'].grad = -grad_theta
        self.optimizer.step()

    def log_progress(self, t: int):
        with torch.no_grad():
            g_soft = get_soft_gmat(self.params['z'], self.hparams)
            g_hard = (g_soft > 0.5).float()
            shd = torch.sum(torch.abs(g_hard.cpu() - self.G_true.cpu()))
            
            # --- Log Joint Calculation (Vectorized) ---
            theta = self.params['theta']
            z = self.params['z']
            g_samples = self.hparams['g_samples']

            log_lik_fn = lambda g: log_likelihood_given_g_and_theta(self.data, g, theta, self.hparams)
            log_theta_prior_fn = lambda g: log_prior_theta(g, theta, self.hparams)

            log_lik_samples = vmap(log_lik_fn)(g_samples)
            log_theta_prior_samples = vmap(log_theta_prior_fn)(g_samples)
            acyclicity_samples = vmap(acyclic_constr)(g_samples)

            exp_log_lik = torch.mean(log_lik_samples)
            exp_log_theta_prior = torch.mean(log_theta_prior_samples)
            exp_acyclicity = torch.mean(acyclicity_samples)
            
            log_z_prior_reg = log_prior_z(z)
            log_z_prior_acyc = -self.hparams['beta'] * exp_acyclicity
            
            f_g_samples = log_lik_samples + log_theta_prior_samples
            exp_f_g = torch.mean(f_g_samples)
            total_objective = exp_f_g + log_z_prior_reg + log_z_prior_acyc

            # MLflow logging
            metrics_to_log = {
                "shd": shd.item(),
                "total_objective": total_objective.item(),
                "exp_log_lik": exp_log_lik.item(),
                "exp_log_theta_prior": exp_log_theta_prior.item(),
                "log_z_prior_reg": log_z_prior_reg.item(),
                "log_z_prior_acyc": log_z_prior_acyc.item(),
                "exp_acyclicity": exp_acyclicity.item(),
                "alpha": self.hparams['alpha'],
                "beta": self.hparams['beta']
            }
            mlflow.log_metrics(metrics_to_log, step=t)

            # Log g_soft as an artifact
            import tempfile
            import os
            with tempfile.TemporaryDirectory() as tmpdir:
                g_soft_path = os.path.join(tmpdir, f"g_soft_iter_{t}.npy")
                np.save(g_soft_path, g_soft.cpu().numpy())
                mlflow.log_artifact(g_soft_path, artifact_path="g_soft_history")

            log.info(f"--- Iter {t}/{self.cfg.num_iterations} ---")
            log.info(f"Annealed: alpha={self.hparams['alpha']:.3f}, beta={self.hparams['beta']:.3f}")
            log.info(f"SHD: {shd.item():.1f}")
            log.info(f"Objective Value (estimated): {total_objective.item():.4f}")
            log.info(f"  - E[Log Likelihood]: {exp_log_lik.item():.4f}")
            log.info(f"  - E[Log Prior Theta]: {exp_log_theta_prior.item():.4f}")
            log.info(f"  - Log Prior Z (Regularizer): {log_z_prior_reg.item():.4f}")
            log.info(f"  - E[Acyclicity Constraint Term]: {log_z_prior_acyc.item():.4f} (E[h(G)]={exp_acyclicity.item():.4f})")
            log.info(f"Current Edge Probs (rounded):\n{np.round(g_soft.cpu().numpy(), 2)}")

    def evaluate(self):
        with torch.no_grad():
            final_probs = get_soft_gmat(self.params['z'], self.hparams)
            learned_hard_graph = (final_probs > 0.5).float()
            
            log.info(f"\n{'='*60}\nFINAL RESULTS COMPARISON\n{'='*60}")
            log.info(f"Ground Truth Adjacency Matrix:\n{self.G_true}")
            log.info(f"Ground Truth Theta:\n{self.Theta_true}")
            log.info(f"Learned Hard Graph (threshold=0.5):\n{learned_hard_graph.cpu().numpy()}")
            log.info(f"Learned Theta:\n{self.params['theta'].cpu().numpy()}")
            log.info(f"Learned causal graph:\n{learned_hard_graph.cpu().numpy() * self.params['theta'].cpu().numpy()}")
            
            shd = torch.abs(self.G_true - learned_hard_graph).sum().item()
            log.info(f"\nStructural Hamming Distance: {shd}")
            mlflow.log_metric("final_shd", shd)

            # Log artifacts
            import tempfile
            import os
            with tempfile.TemporaryDirectory() as tmpdir:
                g_true_path = os.path.join(tmpdir, "g_true.txt")
                theta_true_path = os.path.join(tmpdir, "theta_true.txt")
                learned_g_path = os.path.join(tmpdir, "learned_g.txt")
                learned_theta_path = os.path.join(tmpdir, "learned_theta.txt")
                
                np.savetxt(g_true_path, self.G_true.cpu().numpy(), fmt='%.2f')
                np.savetxt(theta_true_path, self.Theta_true.cpu().numpy(), fmt='%.2f')
                np.savetxt(learned_g_path, learned_hard_graph.cpu().numpy(), fmt='%.2f')
                np.savetxt(learned_theta_path, self.params['theta'].cpu().numpy(), fmt='%.2f')
                
                mlflow.log_artifacts(tmpdir, artifact_path="final_matrices")


def main():
    log.info("--- Running VECTORIZED version ---")
    mlflow.set_experiment("DiBS Vectorised Experiments")
    cfg = Config()
    trainer = DIBSTrainer(cfg)
    trainer.train()

if __name__ == '__main__':
    main()
