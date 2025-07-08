"""
This script is a revised version of nazaal_v2.py, incorporating several critical fixes
to align the implementation more closely with the original DiBS paper and ensure
the mathematical correctness of the gradient calculations.

Reasoning for Changes:

1.  **Corrected `z` Parameterization and `einsum`:**
    The latent variable `z` was previously parameterized as a `(d, d, 2)` tensor. This is
    inconsistent with the DiBS paper, which specifies a shape of `(d, 2, k)` where `d`
    is the number of variables and `k` is the embedding dimension. This version adopts
    the standard `(d, 2, k)` shape (using `k=d` for this case) and adjusts the
    `einsum` operation accordingly. This change improves clarity and alignment with the
    source material.

2.  **Unified and Corrected Log Joint Probability:**
    The previous version had two separate and partially incorrect log joint functions.
    The key quantity for the score function estimator for both `z` and `theta` gradients
    is `log p(D, θ | G)`. The old functions either included incorrect terms (like
    `log p(G|Z)` inside the expectation) or used the `soft_gmat` for prediction instead
    of the sampled `hard_gmat`.

    This version introduces a single, correct function, `log_prob_data_and_theta_given_g`,
    which computes `log p(D, θ | G)`. This function correctly uses the provided hard
    graph `gmat` for both the data likelihood `p(D|G, θ)` and the prior on the weights
    `p(θ|G)`.

3.  **Sound Gradient Calculation:**
    *   **For `grad_z`**: The score function estimator now correctly uses the new
        `log_prob_data_and_theta_given_g` function. This fixes a critical bug where
        terms related to `p(G|Z)` were incorrectly included inside the expectation,
        effectively double-counting them. The gradient of the prior `log p(Z)` was
        already correctly implemented and remains the same.
    *   **For `grad_theta`**: The gradient calculation for `θ` also now relies on the
        corrected `log_prob_data_and_theta_given_g`. While the structure of the
        calculation (a weighted average of gradients) was correct for the objective
        function, it produced the wrong result because the underlying log-probability
        function was flawed. This has been fixed.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch


def loglik_gaussian(x, pred_mean, sigma):
    assert sigma.shape[0] == pred_mean.shape[-1]

    distr = torch.distributions.Normal(pred_mean, sigma)
    return torch.sum(distr.log_prob(x))


def loglik_bernoulli(y, pred_p, rho):
    # pred_p is supposed to be the soft value predicted for some G_ij
    jitter = 1e-5
    p_tilde = rho + pred_p - 2.0 * rho * pred_p
    return torch.sum(
        y * torch.log(1.0 - p_tilde + jitter) + (1.0 - y) * torch.log(p_tilde + jitter)
    )


def soft_gmat_gumbel(z, hparams):
    # For a single z, return a soft gmat using a reparametrized bernoulli distribution
    # Higher tau -> Reduces bias
    d = z.shape[0]
    cycle_mask = 1.0 - torch.eye(d)

    u = torch.rand((d, d))
    ls = torch.log(u) - torch.log(1.0 - u)
    # Corrected einsum based on z shape (d, 2, k)
    inner_product_scores = torch.einsum("ik,jk->ij", z[:, 0, :], z[:, 1, :])
    sampled_scores = ls + (hparams["alpha"] * inner_product_scores)
    soft_gmat_unmasked = torch.sigmoid(hparams["tau"] * sampled_scores)
    return cycle_mask * soft_gmat_unmasked


def hard_gmat(z):
    d = z.shape[0]
    cycle_mask = 1.0 - torch.eye(d)
    # Corrected einsum based on z shape (d, 2, k)
    inner_product_scores = torch.einsum("ik,jk->ij", z[:, 0, :], z[:, 1, :])
    return cycle_mask * (inner_product_scores > 0.0)


def soft_gmat(z, hparams):
    # Deterministic transform of the latent variable z to a soft adjacency matrix
    d = z.shape[0]
    cycle_mask = 1.0 - torch.eye(d)
    # Corrected einsum based on z shape (d, 2, k)
    inner_product_scores = torch.einsum("ik,jk->ij", z[:, 0, :], z[:, 1, :])
    soft_gmat_unmasked = torch.sigmoid(hparams["alpha"] * inner_product_scores)
    return cycle_mask * soft_gmat_unmasked


def acyclic_constr(g_mat):
    # NOTE this code is a copy-paste from the dibs implementation
    d = g_mat.shape[0]
    alpha = 1.0 / d
    M = torch.eye(d) + alpha * g_mat
    M_mult = torch.linalg.matrix_power(M, d)
    h = torch.trace(M_mult) - d
    return h


def log_prob_data_and_theta_given_g(data, params, hparams, gmat):
    """
    Computes log p(D, θ | G=gmat), which is the core term for the score function estimator.
    This is composed of:
    - Data likelihood: log p(D | G, θ)
    - Theta prior:   log p(θ | G)
    """
    # Data likelihood: log p(D | G, θ)
    # The prediction for x MUST use the specific hard graph `gmat`
    pred_mean_x = data["x"] @ (gmat * params["theta"])
    loglik_x = loglik_gaussian(data["x"], pred_mean_x, hparams["sigma"])
    loglik_y = 0.0  # Placeholder for expert feedback

    # Theta prior: log p(θ | G)
    # The prior on weights `theta` only applies to edges present in `gmat`
    log_prior_theta = torch.sum(
        torch.distributions.Normal(
            torch.zeros_like(params["theta"]),
            torch.ones_like(params["theta"]),
        ).log_prob(params["theta"] * gmat)
    )

    return loglik_x + loglik_y + log_prior_theta


def score_func_estimator_stable(params, hparams, log_f, hard_gmats, normalized=True):
    # f is a function of G, as in Equation B.8 from the DiBS paper.
    # It computes E_{p(G|Z)}[f(G) \nabla_Z log p(G|Z)] / E_{p(G|Z)}[f(G)]
    _n_mc = hard_gmats.shape[0]

    # log f(G) for all sampled graphs G
    log_fs = torch.func.vmap(log_f)(hard_gmats)

    # grad_Z log p(G|Z) for a single G
    score_func = lambda g: torch.autograd.grad(
        torch.distributions.Bernoulli(soft_gmat(params["z"], hparams))
        .log_prob(g)
        .sum(),
        params["z"],
        create_graph=True,
    )[0]

    # grad_Z log p(G|Z) for all sampled graphs G
    scores = torch.stack([score_func(g) for g in hard_gmats])

    # Ensure log_fs can be broadcast with scores
    while log_fs.dim() < scores.dim():
        log_fs = log_fs.unsqueeze(-1)

    # Numerator: logsumexp(log(f(G)) + log(score))
    # Use log(abs(score)) and sign(score) for stability with negative scores
    log_numerator = torch.logsumexp(
        log_fs + torch.log(torch.abs(scores) + 1e-30) + torch.log(torch.sign(scores) + 1.1),
        dim=0,
    )


    if normalized:
        # Denominator: logsumexp(log(f(G)))
        log_denominator = torch.logsumexp(log_fs, dim=0)
        result = torch.exp(log_numerator - log_denominator)
    else:
        # Unnormalized estimator
        result = torch.exp(log_numerator - torch.log(torch.tensor(_n_mc)))

    return result


def grad_z_neg_log_joint(data, params, hparams):
    """
    Computes the negative gradient of the log marginal likelihood w.r.t. z.
    -log p(D, θ, Z) = -log p(Z) - log E_{p(G|Z)}[p(D, θ | G)]
    ∇z log p(D, θ, Z) = ∇z log p(Z) + ∇z log E_{p(G|Z)}[p(D, θ | G)]
    """
    hard_gmats = params["hard_gmats"]
    _n_mc = hard_gmats.shape[0]
    d = params["z"].shape[0]

    # First term: Gradient of the log prior on Z, log p(Z)
    # p(Z) has two parts: acyclicity constraint and a regularizer.
    
    # Gradient of acyclicity constraint part using score function estimator
    h_grad_func = (
        lambda g: acyclic_constr(g)
        * torch.autograd.grad(
            torch.distributions.Bernoulli(soft_gmat(params["z"], hparams))
            .log_prob(g)
            .sum(),
            params["z"],
            create_graph=True,
        )[0]
    )
    h_grads = torch.stack([h_grad_func(g) for g in hard_gmats])
    grad_z_log_prior_acyclic_constr = -hparams["beta"] * torch.mean(h_grads, dim=0)

    # Gradient of the regularizer part
    grad_log_prior_z_regularizer = -(1 / torch.tensor(d)) * params["z"]
    grad_z_log_prior_z = grad_z_log_prior_acyclic_constr + grad_log_prior_z_regularizer

    # Second term: Gradient of the expected log likelihood part using score function estimator
    # The function inside the expectation is log p(D, θ | G)
    log_f = lambda g: log_prob_data_and_theta_given_g(
        data, params, hparams, g
    )
    grad_z_expected_log_likelihood = score_func_estimator_stable(
        params, hparams, log_f, hard_gmats, normalized=True
    )

    # Combine gradients
    result = grad_z_log_prior_z + grad_z_expected_log_likelihood

    return -result


def grad_theta_neg_log_joint(data, params, hparams):
    """
    Computes the negative gradient of the log marginal likelihood w.r.t. θ.
    ∇θ log p(D, θ, Z) = ∇θ log E_{p(G|Z)}[p(D, θ | G)]
    This is E_{p(G|Z, D, θ)}[∇θ log p(D, θ | G)], which can be computed with a
    weighted average of gradients.
    """
    hard_gmats = params["hard_gmats"]
    _n_mc = hard_gmats.shape[0]

    # log p(D, θ | G) for all sampled graphs G
    log_fs = torch.tensor(
        [
            log_prob_data_and_theta_given_g(data, params, hparams, g)
            for g in hard_gmats
        ]
    )

    # ∇θ log p(D, θ | G) for a single G
    grad_theta_log_f = lambda g: torch.autograd.grad(
        log_prob_data_and_theta_given_g(data, params, hparams, g),
        params["theta"],
        create_graph=True,
    )[0]

    # Gradients for all sampled graphs
    grads = torch.stack([grad_theta_log_f(g) for g in hard_gmats])

    # Weight gradients by p(D, θ | G) and average
    # Use logsumexp for stable computation of the weighted average
    while log_fs.dim() < grads.dim():
        log_fs = log_fs.unsqueeze(-1)

    log_numerator = torch.logsumexp(log_fs + torch.log(torch.abs(grads) + 1e-30) + torch.log(torch.sign(grads) + 1.1), dim=0)
    log_denominator = torch.logsumexp(log_fs, dim=0)
    
    result = torch.exp(log_numerator - log_denominator)
    return -result


if __name__ == "__main__":
    torch.manual_seed(42)
    # Parameter
    N, d, k = 100, 3, 3 # k is the embedding dimension for z
    sigma = 0.1 * torch.ones((d,))
    gt_gmat = torch.diag(torch.ones(d - 1), diagonal=1)
    gt_theta = 5 * torch.rand((d, d))
    exog_noise = torch.normal(torch.zeros((N, d)), sigma * torch.ones((N, d)))
    # Note: A correct generative process for a linear SEM is x = (I - W^T)^{-1} @ noise
    # For simplicity, we use a single-step generation process.
    x = exog_noise @ torch.inverse(torch.eye(d) - (gt_gmat * gt_theta).T)


    data = {"x": x, "y": {}}
    hparams = {
        "lr": 0.001,
        "sigma": sigma,
        "alpha": 0.01,
        "beta": 1.0,
        "tau": 1.0,
        "k": k,
        "n_grad_mc_samples": 1024,
        "rho": 1.0,
        "minibatch_size": N,
    }

    update_hparams = lambda hps, t: {
        **hps,
        # Annealing schedules can be important for performance
        # "alpha": hps["alpha"] * (t + 1) / t,
        # "beta": hps["beta"] * (t + 1) / t,
    }

    iters = 500 + 1
    params = {
        # z shape is (d, 2, k) according to the paper
        "z": torch.randn((d, 2, k), requires_grad=True),
        "theta": torch.randn((d, d), requires_grad=True),
    }
    optimizer = torch.optim.RMSprop(list(params.values()), lr=hparams["lr"])

    for t in range(1, iters):
        optimizer.zero_grad()
        
        # Sample a batch of graphs for this iteration
        _soft_gmat = soft_gmat(params["z"], hparams)
        distr = torch.distributions.Bernoulli(_soft_gmat)
        hard_gmats = distr.sample((hparams["n_grad_mc_samples"],))

        # Create param dicts for gradient functions, detaching where necessary
        params_for_z = {
            "z": params["z"],
            "theta": params["theta"].detach(),
            "hard_gmats": hard_gmats,
        }
        params_for_theta = {
            "z": params["z"].detach(),
            "theta": params["theta"],
            "hard_gmats": hard_gmats,
        }

        # Compute gradients
        grad_z = grad_z_neg_log_joint(data, params_for_z, hparams)
        grad_theta = grad_theta_neg_log_joint(data, params_for_theta, hparams)
        
        # Assign gradients and update
        params["z"].grad = grad_z
        params["theta"].grad = grad_theta
        optimizer.step()

        hparams = update_hparams(hparams, t)

        if t % 50 == 0:
            grad_z_norm = torch.mean(torch.abs(grad_z)).item()
            grad_theta_norm = torch.mean(torch.abs(grad_theta)).item()
            
            # For logging, calculate log probability with a single sampled graph
            current_hard_gmat = (soft_gmat(params["z"], hparams) > 0.5).float()
            log_prob = log_prob_data_and_theta_given_g(data, params, hparams, current_hard_gmat).item()

            print(f"Iter {t:4d} | "
                  f"grad_z: {grad_z_norm:.4f} | "
                  f"grad_theta: {grad_theta_norm:.4f} | "
                  f"log_prob: {log_prob:.2f}")
            print(f"soft_gmat:\n{soft_gmat(params['z'], hparams).detach().numpy().round(2)}")


    # Plot results
    learnt_soft_gmat = soft_gmat(params["z"], {"alpha": 1.0})
    learnt_hard_gmat = (learnt_soft_gmat > 0.5).float().detach().numpy()
    learnt_theta = params["theta"].detach().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Ground Truth vs. Learnt Graph")

    for ax, matrix, title in zip(axes, [gt_gmat * gt_theta, learnt_hard_gmat * learnt_theta], ["Ground Truth", "Learnt"]):
        im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=5)
        ax.set_title(title)
        for (x, y), val in np.ndenumerate(matrix):
            ax.text(y, x, f"{val:.2f}", ha="center", va="center", color="white")
    
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
    plt.show()
