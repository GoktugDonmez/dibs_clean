import torch
from torch.distributions import Bernoulli
import numpy as np
import logging
from typing import Dict, Any, Tuple
import torch.nn as nn
import igraph as ig

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

# -----------------------------------------------------------------------------
# Core helpers (mostly identical to original, but with device-aware tensor
# creation)
# -----------------------------------------------------------------------------

def acyclic_constr(g: torch.Tensor, d: int) -> torch.Tensor:
    """ NOTE: supports batched `g` of shape (S,d,d) or single (d,d). """
    alpha = 1.0 / d
    eye = torch.eye(d, device=g.device, dtype=g.dtype)
    m = eye + alpha * g
    return torch.trace(torch.linalg.matrix_power(m, d)) - d if g.dim() == 2 else (
        # vmap over first dim when batched
        torch.func.vmap(lambda A: torch.trace(torch.linalg.matrix_power(eye + alpha * A, d)) - d)(g)
    )

def log_gaussian_likelihood(x: torch.Tensor, pred_mean: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    sigma_tensor = torch.tensor(sigma, dtype=pred_mean.dtype, device=pred_mean.device)
    residuals = x - pred_mean
    log_prob = -0.5 * (torch.log(2 * torch.pi * sigma_tensor**2)) - 0.5 * ((residuals / sigma_tensor) ** 2)
    return torch.sum(log_prob)

def scores(z: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    u, v = z[..., 0], z[..., 1]
    raw_scores = hparams["alpha"] * torch.einsum("...ik,...jk->...ij", u, v)
    _, d = z.shape[:-1]
    diag_mask = 1.0 - torch.eye(d, device=z.device, dtype=z.dtype)
    return raw_scores * diag_mask

def soft_gmat(z: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    raw_scores = scores(z, hparams)
    edge_probs = torch.sigmoid(raw_scores)
    d = z.shape[0]
    diag_mask = 1.0 - torch.eye(d, device=z.device, dtype=z.dtype)
    return edge_probs * diag_mask

def log_full_likelihood(data: Dict[str, Any], g_soft: torch.Tensor, theta: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    x_data = data["x"]
    effective_W = theta * g_soft
    pred_mean = torch.matmul(x_data, effective_W)
    sigma_obs = hparams.get("sigma_obs_noise", 0.1)
    return log_gaussian_likelihood(x_data, pred_mean, sigma=sigma_obs)

def log_theta_prior(theta_effective: torch.Tensor, sigma: float) -> torch.Tensor:
    return log_gaussian_likelihood(theta_effective, torch.zeros_like(theta_effective), sigma=sigma)

# -----------------------------------------------------------------------------
# Numerical stable ratio helper (unchanged)
# -----------------------------------------------------------------------------

def stable_ratio(grad_samples, log_density_samples):
    eps = 1e-30
    log_p = torch.stack(log_density_samples)
    grads = torch.stack(grad_samples)
    while log_p.dim() < grads.dim():
        log_p = log_p.unsqueeze(-1)
    log_grads_abs = torch.log(grads.abs() + eps) + log_p
    pos_mask = grads >= 0
    neg_mask = grads < 0
    n_pos = pos_mask.sum().clamp(min=1)
    n_neg = neg_mask.sum().clamp(min=1)
    log_grads_abs_pos = log_grads_abs.masked_fill(~pos_mask, float("-inf"))
    log_grads_abs_neg = log_grads_abs.masked_fill(~neg_mask, float("-inf"))
    log_num_pos = torch.logsumexp(log_grads_abs_pos, dim=0) - torch.log(n_pos.float())
    log_num_neg = torch.logsumexp(log_grads_abs_neg, dim=0) - torch.log(n_neg.float())
    log_den = torch.logsumexp(log_p, dim=0) - torch.log(torch.tensor(len(log_p), dtype=log_p.dtype, device=log_p.device))
    return torch.exp(log_num_pos - log_den) - torch.exp(log_num_neg - log_den)

# -----------------------------------------------------------------------------
# Vectorised / GPU-friendly Monte-Carlo estimators
# -----------------------------------------------------------------------------

def score_func_estimator_stable(data, z, hparams, f, normalized=True):
    """Vectorised version without Python for-loops."""
    S = hparams["n_mc_samples"]
    g_probs = soft_gmat(z, hparams)
    hard_gmats = Bernoulli(g_probs).sample((S,))  # (S,d,d) on same device

    # 1)   log_f(G)
    log_f = torch.func.vmap(f, randomness="different")(hard_gmats)  # (S,)

    # 2)   score function ∇_z log p(G|Z)
    def _score_single(g):
        return torch.autograd.grad(Bernoulli(soft_gmat(z, hparams)).log_prob(g).sum(), z, create_graph=True)[0]

    scores = torch.func.vmap(_score_single)(hard_gmats)  # (S, *z.shape)

    while log_f.dim() < scores.dim():
        log_f = log_f.unsqueeze(-1)

    log_num = torch.logsumexp(log_f + torch.log(torch.clamp(torch.abs(scores), min=1e-30)) * torch.sign(scores), dim=0)
    if normalized:
        log_denom = torch.logsumexp(log_f, dim=0)
        return torch.exp(log_num - log_denom)
    return torch.exp(log_num)


def grad_theta_score(z: torch.Tensor, data: Dict[str, Any], theta: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    """Fully vectorised Monte-Carlo gradient wrt theta."""
    S = hparams.get("n_mc_samples", 64)
    g_hard = Bernoulli(soft_gmat(z, hparams)).sample((S,))  # (S,d,d)

    # Expand theta to batch & keep track of grads via functorch
    theta_batched = theta.expand(S, *theta.shape).clone()

    def log_density_single(th, g):
        return (
            log_full_likelihood(data, g, th, hparams)
            + log_theta_prior(th * g, hparams.get("theta_prior_sigma", 1.0))
        )

    # Vectorised log-density
    log_densities = torch.func.vmap(log_density_single)(theta_batched, g_hard)  # (S,)

    # Vectorised gradient w.r.t first argument (theta)
    grad_fn = torch.func.grad(log_density_single)
    grad_samples = torch.func.vmap(grad_fn)(theta_batched, g_hard)  # (S,d,d)

    return stable_ratio(grad_samples, log_densities)

# -----------------------------------------------------------------------------
# Gradients for z (uses the new score func estimator)
# -----------------------------------------------------------------------------

def grad_z_score_stable(z: torch.Tensor, data: Dict[str, Any], theta: torch.Tensor, hparams: Dict[str, Any]) -> torch.Tensor:
    grad_z_prior = -(z / hparams["sigma_z"] ** 2)

    def f_likelihood(g):
        return log_full_likelihood(data, g, theta, hparams) + log_theta_prior(theta * g, hparams["theta_prior_sigma"])

    grad_lik = score_func_estimator_stable(data, z, hparams, f_likelihood, normalized=True)

    def f_acyclic(g):
        d = z.shape[0]
        return acyclic_constr(g, d)

    grad_acyc = score_func_estimator_stable(data, z, hparams, f_acyclic, normalized=False)

    return grad_z_prior + grad_lik - hparams["beta"] * grad_acyc

# -----------------------------------------------------------------------------
# Top-level joint gradient & (optional) compile for PyTorch ≥ 2.0
# -----------------------------------------------------------------------------

def grad_log_joint(params: Dict[str, torch.Tensor], data: Dict[str, Any], hparams: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_z = grad_z_score_stable(params["z"], data, params["theta"].detach(), hparams)
    grad_theta = grad_theta_score(params["z"].detach(), data, params["theta"], hparams)
    return grad_z, grad_theta

# Try TorchDynamo / Triton compile if available (PyTorch ≥ 2.0)
try:
    grad_log_joint = torch.compile(grad_log_joint)  # type: ignore
    log.info("Using torch.compile for grad_log_joint speed-up.")
except Exception as e:
    log.info(f"torch.compile unavailable or failed ({e}); continuing without it.")

# -----------------------------------------------------------------------------
# Fast version of log_joint (batch acyclicity)
# -----------------------------------------------------------------------------

def log_joint(params: Dict[str, Any], data: Dict[str, Any], hparams: Dict[str, Any]) -> torch.Tensor:
    z, theta = params["z"], params["theta"]
    g_soft = soft_gmat(z, hparams)
    ll = log_full_likelihood(data, g_soft, theta, hparams)
    lp_theta = log_theta_prior(theta, hparams.get("sigma_theta_prior", 1.0))

    with torch.no_grad():
        S = hparams.get("n_mc_samples", 64)
        g_hard = Bernoulli(g_soft).sample((S,))
        d = z.shape[0]
        h_vals = acyclic_constr(g_hard, d)  # vectorised call -> (S,)
        acyclicity_val = h_vals.mean()

    lp_z = -0.5 * torch.sum(z ** 2) / hparams.get("latent_prior_std", 1.0) ** 2
    return ll + lp_theta + lp_z - hparams["beta"] * acyclicity_val

# -----------------------------------------------------------------------------
# Data generation helpers (device-aware)
# -----------------------------------------------------------------------------

def generate_chain(num_samples, chain_length, obs_noise_std, device):
    G_true = torch.zeros(chain_length, chain_length, dtype=torch.float32, device=device)
    for i in range(chain_length - 1):
        G_true[i, i + 1] = 1.0
    Theta_true = torch.zeros_like(G_true)
    Theta_true[:-1, 1:] = torch.tensor([2.0, -1.5], device=device).unsqueeze(1)[: chain_length - 1]
    X_data = torch.zeros(num_samples, chain_length, device=device)
    X_data[:, 0] = torch.randn(num_samples, device=device) * obs_noise_std
    for i in range(1, chain_length):
        noise = torch.randn(num_samples, device=device) * obs_noise_std
        X_data[:, i] = Theta_true[i - 1, i] * X_data[:, i - 1] + noise
    return X_data, G_true, Theta_true

# -----------------------------------------------------------------------------
# Config and training loop (AMP enabled)
# -----------------------------------------------------------------------------

class Config:
    seed = 31
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_nodes = 3
    num_samples = 1000
    obs_noise_std = 0.1
    k_latent = d_nodes
    alpha_base = 0.2
    beta_base = 1.0
    theta_prior_sigma = 1.0
    n_mc_samples = 128
    lr = 5e-3
    num_iterations = 1500
    debug_print_iter = 100

cfg = Config()

def update_dibs_hparams(hparams: Dict[str, Any], t: int) -> Dict[str, Any]:
    hparams["alpha"] = hparams["alpha_base"] * t * 0.02
    hparams["beta"] = hparams["beta_base"] * t * 0.1
    hparams["current_t"] = t
    return hparams

# ----------------------------------------------------------------------------
# Training utilities (unchanged apart from device placement)
# ----------------------------------------------------------------------------

def debug_prediction_error(params, data, hparams, G_true):
    with torch.no_grad():
        z, theta = params["z"], params["theta"]
        x_data = data["x"]
        sigma_obs = hparams["sigma_obs_noise"]
        g_probs = soft_gmat(z, hparams)
        g_learned = torch.bernoulli(g_probs)
        pred_mean = torch.matmul(x_data, theta * g_learned)
        residuals = x_data - pred_mean
        mse = torch.mean(residuals ** 2)
        mae = torch.mean(torch.abs(residuals))
        shd = torch.sum(torch.abs(g_learned.cpu() - G_true.cpu()))
        log.info(f"SHD: {shd.item():.1f} | MSE: {mse.item():.6f} | MAE: {mae.item():.6f}")

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    log.info(f"Running on device: {cfg.device}")

    # --- Data ---
    X, G_true, Theta_true = generate_chain(cfg.num_samples, cfg.d_nodes, cfg.obs_noise_std, cfg.device)
    data = {"x": X}

    # --- Parameters ---
    particle = {
        "z": nn.Parameter(torch.randn(cfg.d_nodes, cfg.k_latent, 2, device=cfg.device)),
        "theta": nn.Parameter(torch.randn(cfg.d_nodes, cfg.d_nodes, device=cfg.device)),
    }

    hparams = {
        "alpha": 0.0,
        "beta": 0.0,
        "alpha_base": cfg.alpha_base,
        "beta_base": cfg.beta_base,
        "sigma_z": 1.0 / np.sqrt(cfg.k_latent),
        "sigma_obs_noise": cfg.obs_noise_std,
        "theta_prior_sigma": cfg.theta_prior_sigma,
        "n_mc_samples": cfg.n_mc_samples,
    }

    optimizer = torch.optim.RMSprop(particle.values(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device == "cuda"))

    log.info("Starting training ...")
    for t in range(1, cfg.num_iterations + 1):
        optimizer.zero_grad()
        update_dibs_hparams(hparams, t)
        with torch.cuda.amp.autocast(enabled=(cfg.device == "cuda")):
            grad_z, grad_th = grad_log_joint(particle, data, hparams)
        # Assign ascent gradients
        particle["z"].grad = -grad_z
        particle["theta"].grad = -grad_th
        scaler.step(optimizer) if cfg.device == "cuda" else optimizer.step()
        if cfg.device == "cuda":
            scaler.update()
        if t % cfg.debug_print_iter == 0:
            debug_prediction_error(particle, data, hparams, G_true)

    log.info("Training finished.")

if __name__ == "__main__":
    main() 