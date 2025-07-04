import matplotlib.pyplot as plt
import numpy as np
import torch

# --------------------------------------------------
#  Device selection
# --------------------------------------------------
# Runs on the first CUDA‑capable GPU if available, otherwise CPU.
# Everything created below inherits the device from existing tensors or uses
# the global ``device`` helper.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
#  Helper functions
# --------------------------------------------------

def loglik_gaussian(x, pred_mean, sigma):
    """Log‑likelihood of a spherical Gaussian."""
    assert sigma.shape[0] == pred_mean.shape[-1]
    distr = torch.distributions.Normal(pred_mean, sigma)
    return torch.sum(distr.log_prob(x))


def loglik_bernoulli(y, pred_p, rho):
    """Smoothed Bernoulli log‑likelihood used for expert feedback."""
    jitter = 1e-5
    p_tilde = rho + pred_p - 2.0 * rho * pred_p
    return torch.sum(
        y * torch.log(1.0 - p_tilde + jitter) + (1.0 - y) * torch.log(p_tilde + jitter)
    )


# ---- Graph samplers --------------------------------------------------------

def soft_gmat_gumbel(z, hparams):
    """Soft adjacency matrix via Gumbel‑sigmoid reparametrisation."""
    d = z.shape[0]
    cycle_mask = 1.0 - torch.eye(d, device=z.device)

    u = torch.rand((d, d), device=z.device)
    ls = torch.log(u) - torch.log(1.0 - u)
    inner_product_scores = torch.einsum("ik, jk -> ij", z[:, :, 0], z[:, :, 1])
    sampled_scores = ls + (hparams["alpha"] * inner_product_scores)
    soft_gmat_unmasked = torch.sigmoid(hparams["tau"] * sampled_scores)
    return cycle_mask * soft_gmat_unmasked


def hard_gmat(z):
    """Deterministic hard thresholding at 0.0 (sign of the inner product)."""
    d = z.shape[0]
    cycle_mask = 1.0 - torch.eye(d, device=z.device)
    inner_product_scores = torch.einsum("ik, jk -> ij", z[:, :, 0], z[:, :, 1])
    return cycle_mask * (inner_product_scores > 0.0)


def soft_gmat(z, hparams):
    """Deterministic soft adjacency without stochasticity."""
    d = z.shape[0]
    cycle_mask = 1.0 - torch.eye(d, device=z.device)
    inner_product_scores = torch.einsum("ik, jk -> ij", z[:, :, 0], z[:, :, 1])
    soft_gmat_unmasked = torch.sigmoid(hparams["alpha"] * inner_product_scores)
    return cycle_mask * soft_gmat_unmasked


# ---- Acyclicity constraint --------------------------------------------------

def acyclic_constr_mc_gumbel(z, hparams):
    """Monte‑Carlo estimate of the NOTEARS acyclicity constraint."""
    soft_gmats_gumbel = torch.func.vmap(
        lambda _: soft_gmat_gumbel(z, hparams), randomness="different"
    )(torch.arange(hparams["n_gumbel_mc_samples"], device=z.device))

    hard_gmats = torch.distributions.Bernoulli(soft_gmats_gumbel).sample()
    return torch.mean(torch.func.vmap(acyclic_constr)(hard_gmats))


def acyclic_constr(g_mat):
    d = g_mat.shape[0]
    alpha = 1.0 / d
    M = torch.eye(d, device=g_mat.device) + alpha * g_mat
    M_mult = torch.linalg.matrix_power(M, d)
    h = torch.trace(M_mult) - d
    return h


# ---- Misc. utilities --------------------------------------------------------

def stable_mean(fxs):
    """Numerically stable mean of a 1‑D tensor that may contain small values."""
    jitter = 1e-30

    stable_mean_psve_only = lambda fs, n: torch.exp(
        torch.logsumexp(torch.log(fs + jitter), dim=1) - torch.log(torch.tensor(n, device=fs.device) + jitter)
    )

    f_xs_psve = fxs * (fxs > 0.0)
    f_xs_ngve = -fxs * (fxs < 0.0)
    n_psve = torch.sum((fxs > 0.0))
    n_ngve = fxs.numel() - n_psve
    avg_psve = stable_mean_psve_only(f_xs_psve, n_psve)
    avg_ngve = stable_mean_psve_only(f_xs_ngve, n_ngve)

    return (n_psve / fxs.numel()) * avg_psve - (n_ngve / fxs.numel()) * avg_ngve


# ---- Log‑joint --------------------------------------------------------------

def log_joint(data, params, hparams):
    """Log‑joint p(D, Z, Θ, G)."""
    d = data["x"].shape[-1]
    gmat = params["hard_gmat"]

    pred_mean_x = lambda x_, ps: x_ @ (
        soft_gmat(params["z"], hparams) * ps["theta"]
    )

    loglik_x = loglik_gaussian(data["x"], pred_mean_x(data["x"], params), hparams["sigma"])

    # Expert feedback likelihood currently disabled
    loglik_y = torch.tensor(0.0, device=device)

    log_prob_g_given_z = torch.sum(
        torch.distributions.Bernoulli(soft_gmat(params["z"], hparams)).log_prob(gmat.round())
    )

    log_prior_z_constraint = -hparams["beta"] * acyclic_constr_mc_gumbel(params["z"], hparams)

    log_prior_z_regularizer = torch.sum(
        torch.distributions.Normal(
            torch.zeros_like(params["z"]),
            torch.ones_like(params["z"]) / torch.sqrt(torch.tensor(d, device=params["z"].device)),
        ).log_prob(params["z"])
    )

    log_prior_z = log_prior_z_constraint + log_prior_z_regularizer

    log_prior_theta = torch.sum(
        torch.distributions.Normal(
            torch.zeros_like(params["theta"]),
            torch.ones_like(params["theta"]),
        ).log_prob(params["theta"] * gmat)
    )

    return loglik_x + loglik_y + log_prob_g_given_z + log_prior_z + log_prior_theta


# ---- Score‑function & gradient estimators -----------------------------------

def score_func_estimator_stable(params, hparams, log_f, normalized=True):
    hard_gmats = params["hard_gmats"]
    _n_mc = hard_gmats.shape[0]

    log_fs = torch.func.vmap(log_f, randomness="different")(hard_gmats)

    score_func = lambda g: torch.autograd.grad(
        torch.distributions.Bernoulli(soft_gmat(params["z"], hparams)).log_prob(g).sum(),
        params["z"],
        create_graph=True,
    )[0]

    scores = torch.stack([score_func(g) for g in hard_gmats])

    while log_fs.dim() < scores.dim():
        log_fs = log_fs.unsqueeze(-1)

    log_numerator = torch.logsumexp(log_fs + torch.log(torch.abs(scores)) * torch.sign(scores), dim=0)

    if normalized:
        log_denominator = torch.logsumexp(log_fs, dim=0)
        result = torch.exp(log_numerator - log_denominator)
    else:
        result = torch.exp(log_numerator - torch.log(torch.tensor(_n_mc, device=scores.device)))

    return result


def grad_z_neg_log_joint(data, params, hparams):
    _n_mc = params["hard_gmats"].shape[0]
    d = params["z"].shape[0]

    h_grad = (
        lambda g: acyclic_constr(g) * torch.autograd.grad(
            torch.distributions.Bernoulli(soft_gmat(params["z"], hparams)).log_prob(g).sum(),
            params["z"],
            create_graph=True,
        )[0]
    )

    h_grads = torch.stack([h_grad(params["hard_gmats"][i]) for i in range(_n_mc)])
    grad_z_log_prior_acyclic_constr = -hparams["beta"] * torch.mean(h_grads, dim=0)
    grad_log_prior_z_regularizer = -(1 / torch.tensor(d, device=params["z"].device)) * params["z"]
    grad_z_log_prior_z = grad_z_log_prior_acyclic_constr + grad_log_prior_z_regularizer

    grad_z_log_likelihood = score_func_estimator_stable(
        params, hparams, lambda g: log_joint(data, params | {"hard_gmat": g}, hparams)
    )
    return -(grad_z_log_prior_z + grad_z_log_likelihood)


def grad_theta_neg_log_joint(data, params, hparams):
    hard_gmats = params["hard_gmats"]
    _n_mc = hard_gmats.shape[0]

    log_fs = torch.tensor(
        [log_joint(data, {**params, "hard_gmat": hard_gmats[i]}, hparams) for i in range(_n_mc)],
        device=device,
    )

    score_func = lambda g: torch.autograd.grad(
        log_joint(data, {**params, "hard_gmat": g}, hparams),
        params["theta"],
        create_graph=True,
    )[0]

    scores = torch.clamp(torch.stack([score_func(g) for g in hard_gmats]), min=1e-32)

    while log_fs.dim() < scores.dim():
        log_fs = log_fs.unsqueeze(-1)

    log_numerator = torch.logsumexp(log_fs + torch.log(torch.abs(scores)) * torch.sign(scores), dim=0)
    log_denominator = torch.logsumexp(log_fs, dim=0)

    return -torch.exp(log_numerator - log_denominator)


# --------------------------------------------------
#  Main experiment
# --------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # ---------------------  synthetic data  ---------------------------------
    N, d = 100, 3
    sigma = 0.1 * torch.ones((d,), device=device)

    gt_gmat = torch.diag(torch.ones(d - 1, device=device), diagonal=1)
    gt_theta = 5 * torch.rand((d, d), device=device)

    exog_noise = torch.normal(
        torch.zeros((N, d), device=device), sigma * torch.ones((N, d), device=device)
    )
    x = exog_noise @ (gt_gmat * gt_theta)

    data = {"x": x, "y": {}}

    # ---------------------  hyper‑parameters  --------------------------------
    hparams = {
        "lr": 0.001,
        "sigma": sigma,
        "alpha": 0.01,
        "beta": 1,
        "tau": 1.0,
        "n_gumbel_mc_samples": 128,
        "n_grad_mc_samples": 16,
        "rho": 1.0,
        "minibatch_size": N,
        "n_score_func_mc_samples": 1024,
    }

    update_hparams = lambda hps, t: {
        **hps,
        "alpha": hps["alpha"] * (t + 1) / t,
        "beta": hps["beta"] * (t + 1) / t,
        # "tau": hps["tau"] * (t + 1) / t,
    }

    # ---------------------  parameters & optimiser  -------------------------
    params = {
        "z": torch.randn((d, d, 2), device=device, requires_grad=True),
        "theta": torch.randn((d, d), device=device, requires_grad=True),
    }
    optimizer = torch.optim.RMSprop(list(params.values()), lr=hparams["lr"])

    # ---------------------  gradient ascent ----------------------------------
    iters = 501
    for t in range(1, iters):
        optimizer.zero_grad()

        _soft_gmat = soft_gmat(params["z"], hparams)
        distr = torch.distributions.Bernoulli(_soft_gmat)
        hard_gmats = distr.sample((hparams["n_score_func_mc_samples"],))

        # Detach to block gradients where appropriate
        grad_z = grad_z_neg_log_joint(
            data,
            {**params, "hard_gmats": hard_gmats, "theta": params["theta"].detach()},
            hparams,
        )
        grad_theta = grad_theta_neg_log_joint(
            data,
            {**params, "hard_gmats": hard_gmats, "z": params["z"].detach()},
            hparams,
        )

        # Manually assign grads
        params["z"].grad = grad_z
        params["theta"].grad = grad_theta

        optimizer.step()
        hparams = update_hparams(hparams, t)

        # ---- logging every 10 iters
        if t % 10 == 0:
            grad_z_norm = grad_z.abs().mean().detach().cpu().item()
            grad_theta_norm = grad_theta.abs().mean().detach().cpu().item()
            print(f"t={t:04d} | grad_z_norm={grad_z_norm:.3e} | grad_theta_norm={grad_theta_norm:.3e}")

            current_params = {**params, "hard_gmat": hard_gmat(params["z"])}
            lj = log_joint(data, current_params, hparams).detach().cpu().item()
            print(f"           log_joint={lj:.2f}")
            print(f"soft_gmat={soft_gmat(params['z'], hparams)}")

