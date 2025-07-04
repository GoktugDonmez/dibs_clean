import matplotlib.pyplot as plt
import numpy as np
import torch


def loglik_gaussian(x, pred_mean, sigma):
    assert sigma.shape[0] == pred_mean.shape[-1]

    distr = torch.distributions.Normal(pred_mean, sigma)
    return torch.sum(distr.log_prob(x))


def loglik_bernoulli(y, pred_p, rho):
    # TODO Have a proper look at this again
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
    inner_product_scores = torch.einsum("ik, jk -> ij", z[:, :, 0], z[:, :, 1])
    sampled_scores = ls + (hparams["alpha"] * inner_product_scores)
    soft_gmat_unmasked = torch.sigmoid(hparams["tau"] * sampled_scores)
    return cycle_mask * soft_gmat_unmasked


def grad_z_log_joint_gumbel(data, params, hparams):
    # TODO

    # 1 sample for now
    soft_gmat = soft_gmat_gumbel(params["z"], hparams)

    # grad_g_f = torch.autograd.grad(
    #     log_joint(data, {**params, "soft_gmat": soft_gmat}, hparams), soft_gmat
    # )

    # grad_z_g = torch.autograd.grad(soft_gmat, params["z"])

    grad_z_log_joint = torch.autograd.grad(
        log_joint(data, {**params, "soft_gmat": soft_gmat}, hparams), params["z"]
    )
    return grad_z_log_joint


def hard_gmat(z):
    d = z.shape[0]
    cycle_mask = 1.0 - torch.eye(d)
    inner_product_scores = torch.einsum("ik, jk -> ij", z[:, :, 0], z[:, :, 1])
    return cycle_mask * (inner_product_scores > 0.0)


def soft_gmat(z, hparams):
    # Deterministic transform of the latent variable z to a soft adjacency matrix
    d = z.shape[0]
    cycle_mask = 1.0 - torch.eye(d)
    inner_product_scores = torch.einsum("ik, jk -> ij", z[:, :, 0], z[:, :, 1])
    soft_gmat_unmasked = torch.sigmoid(hparams["alpha"] * inner_product_scores)
    return cycle_mask * soft_gmat_unmasked


def acyclic_constr_mc_gumbel(z, hparams):
    # Monte-carlo estimate for h(G) and denominator computation
    soft_gmats_gumbel = torch.func.vmap(
        lambda _: soft_gmat_gumbel(z, hparams), randomness="different"
    )(torch.arange(hparams["n_gumbel_mc_samples"]))
    # hard_gmats = soft_gmats_gumbel
    hard_gmats = torch.distributions.Bernoulli(soft_gmats_gumbel).sample()
    return torch.mean(torch.func.vmap(acyclic_constr)(hard_gmats))


def acyclic_constr(g_mat):
    # NOTE this code is a copy-paste from the dibs implementation

    d = g_mat.shape[0]
    alpha = 1.0 / d
    M = torch.eye(d) + alpha * g_mat
    M_mult = torch.linalg.matrix_power(M, d)
    h = torch.trace(M_mult) - d
    return h


def stable_mean(fxs):
    # assumes fs are only positive
    jitter = 1e-30

    # Taking n separately we need non-zero
    stable_mean_psve_only = lambda fs, n: torch.exp(
        torch.logsumexp(torch.log(fs), dim=1) - torch.log(n + jitter)
    )

    f_xs_psve = fxs * (fxs > 0.0)
    f_xs_ngve = -fxs * (fxs < 0.0)
    n_psve = torch.sum((fxs > 0.0))
    n_ngve = fxs.size - n_psve
    avg_psve = stable_mean_psve_only(f_xs_psve, n_psve)
    avg_ngve = stable_mean_psve_only(f_xs_ngve, n_ngve)

    return (n_psve / fxs.size) * avg_psve - (n_ngve / fxs.size) * avg_ngve


def log_joint(data, params, hparams):
    # log joint as a function of Z, Theta which are both in the params dict

    d = data["x"].shape[-1]

    # TODO May want to think of the soft_gmat when gradients aren't being
    # passed around nicely, e.g. wrt theta
    gmat = params["hard_gmat"]

    pred_mean_x = lambda x, ps: x @ (
        soft_gmat(params["z"], hparams) * ps["theta"]
    )  # Note: Will change to use a neural network later
    #
    loglik_x = loglik_gaussian(
        data["x"], pred_mean_x(data["x"], params), hparams["sigma"]
    )

    # pred_mean_y = (
    #     lambda y, ps: 1.0
    # )  # Only get G_ij values for edges i,j involved in data y

    # Expert feedback likelihood
    # loglik_y = loglik_bernoulli(
    #     data["y"], pred_mean_y(data["y"], params), hparams["rho"]
    # )
    loglik_y = 0.0

    log_prob_g_given_z = torch.sum(
        torch.distributions.Bernoulli(soft_gmat(params["z"], hparams)).log_prob(
            gmat.round()
        )
    )

    log_prior_z_constraint = -hparams["beta"] * acyclic_constr_mc_gumbel(
        params["z"], hparams
    )

    log_prior_z_regularizer = torch.sum(
        torch.distributions.Normal(
            torch.zeros_like(params["z"]),
            torch.ones_like(params["z"]) / torch.sqrt(torch.tensor(d)),
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


def update_dibs_hparams(hparams, t):
    # NOTE these values become tracers, so the hparam dict used for
    # inference cannot be used later in utility computations
    # way to get around this is to update d.copy()
    if not isinstance(hparams, dict):
        # Conversion needed since _partial_ instantiation in hydra
        # makes hparams an OmegaConf.dictconf
        pass
        # hparams = OmegaConf.to_container(hparams, resolve=True)

    updated_hparams = hparams.copy()
    updated_hparams["tau"] = hparams["tau"]  # hparams["tau"] * (t + 1 / t)
    updated_hparams["alpha"] = hparams["alpha"] * (t + 1 / t)
    updated_hparams["beta"] = hparams["beta"] * (t + 1 / t)
    return updated_hparams

def stable_ratio(grad_samples, log_density_samples):
    eps = 1e-30
    S   = len(log_density_samples)                    # == M

    if not isinstance(log_density_samples, torch.Tensor):
        log_p   = torch.stack(log_density_samples)        # [S]
    else:
        log_p = log_density_samples
    if not isinstance(grad_samples, torch.Tensor):
        grads   = torch.stack(grad_samples)               # [S,*]
    else:
        grads = grad_samples

    while log_p.dim() < grads.dim():
        log_p = log_p.unsqueeze(-1)

    log_den = torch.logsumexp(log_p, dim=0) - torch.log(torch.tensor(len(log_p), dtype=log_p.dtype, device=log_p.device))

    pos = grads >= 0
    neg = ~pos

    log_num_pos = torch.logsumexp(
        torch.where(pos,
                    torch.log(grads + eps) + log_p,
                    torch.full_like(log_p, -float('inf'))),
        dim=0) - torch.log(torch.tensor(len(log_p), dtype=log_p.dtype, device=log_p.device))

    log_num_neg = torch.logsumexp(
        torch.where(neg,
                    torch.log(grads.abs() + eps) + log_p,
                    torch.full_like(log_p, -float('inf'))),
        dim=0) - torch.log(torch.tensor(len(log_p), dtype=log_p.dtype, device=log_p.device))

    return torch.exp(log_num_pos - log_den) - torch.exp(log_num_neg - log_den)



def score_func_estimator_stable(params, hparams, log_f, normalized=True):
    """
    Numerically stable score function estimator using the log-sum-exp trick.
    It computes E_p(G|Z)[f(G) * score(G)], where score(G) = grad_Z log p(G|Z).
    
    The expectation is split into positive and negative parts for stability:
    E[f*score] = E[f*score | score > 0] - E[f*|score| | score < 0]
    """
    _n_mc = hparams["n_score_func_mc_samples"]
    _soft_gmat = soft_gmat(params["z"], hparams)
    distr = torch.distributions.Bernoulli(_soft_gmat)
    hard_gmats = distr.sample((_n_mc,))

    # log densities of f(G), for all sampled Gs
    log_fs = torch.func.vmap(
        log_f,
        randomness="different",
    )(hard_gmats)

    # The score function: grad_Z log p(G|Z)
    score_func = lambda g: torch.autograd.grad(
        torch.distributions.Bernoulli(soft_gmat(params["z"], hparams))
        .log_prob(g)
        .sum(),
        params["z"],
        create_graph=True,
    )[0]

    # Compute scores for all sampled Gs
    scores = torch.stack([score_func(g) for g in hard_gmats])
    
    # Ensure log_fs is broadcastable with scores
    while log_fs.dim() < scores.dim():
        log_fs = log_fs.unsqueeze(-1)
        
    eps = 1e-30  # Epsilon for numerical stability in log

    # --- Start of the corrected logic ---
    
    # Separate positive and negative scores
    pos_mask = scores >= 0
    neg_mask = ~pos_mask

    # Log of the positive term of the sum: log(sum(f(G) * score(G))) for score(G) >= 0
    # Calculated as logsumexp(log(f(G)) + log(score(G)))
    log_numerator_pos = torch.logsumexp(
        torch.where(pos_mask,
                    log_fs + torch.log(scores + eps),
                    torch.full_like(log_fs, -float('inf'))),
        dim=0
    )

    # Log of the negative term of the sum: log(sum(f(G) * |score(G)|)) for score(G) < 0
    # Calculated as logsumexp(log(f(G)) + log(|score(G)|))
    log_numerator_neg = torch.logsumexp(
        torch.where(neg_mask,
                    log_fs + torch.log(torch.abs(scores) + eps),
                    torch.full_like(log_fs, -float('inf'))),
        dim=0
    )

    if normalized:
        # For a normalized estimator, divide by E[f(G)], which is sum(f(G)) / N.
        # In log space, this corresponds to subtracting log(sum(f(G))).
        log_denominator = torch.logsumexp(log_fs, dim=0)
        
        term_pos = torch.exp(log_numerator_pos - log_denominator)
        term_neg = torch.exp(log_numerator_neg - log_denominator)
    else:
        # For an unnormalized estimator, we just take the mean, dividing by N.
        # In log space, this corresponds to subtracting log(N).
        log_n_mc = torch.log(torch.tensor(_n_mc, dtype=log_fs.dtype, device=log_fs.device))
        
        term_pos = torch.exp(log_numerator_pos - log_n_mc)
        term_neg = torch.exp(log_numerator_neg - log_n_mc)
        
    result = term_pos - term_neg
    return result


def grad_z_neg_log_joint(data, params, hparams):
    d = params["z"].shape[0]
    #grad_z_log_prior_acyclic_constr = -hparams["beta"] * score_func_estimator_stable(
    #    params,
    #    hparams,
    #    lambda g: torch.log(torch.clamp(acyclic_constr(g), 1e-32)),
    #    normalized=False,
    #)
    grad_log_prior_z_regularizer = -(1 / torch.sqrt(torch.tensor(d))) * params["z"]
    #grad_z_log_prior_z = grad_z_log_prior_acyclic_constr + grad_log_prior_z_regularizer

    #grad_z_log_likelihood = score_func_estimator_stable(
    #    params,
    #    hparams,
    #    lambda g: log_joint(data, params | {"hard_gmat": g}, hparams),
    #    normalized=True,
    #)
    # calculate the likelihood with stable ratio
    log_fs = []
    scores = []
    grad_prior_acyclic_constr_total = 0.0
    for i in range(hparams["n_score_func_mc_samples"]):
        hard_gmat = torch.distributions.Bernoulli(soft_gmat(params["z"], hparams)).sample()
        log_f = log_joint(data, params | {"hard_gmat": hard_gmat}, hparams)
        grad = torch.autograd.grad(log_f, params["z"], create_graph=True)[0]
        log_fs.append(log_f)
        scores.append(grad)

        grad_prior_acyclic_constr_total +=  acyclic_constr(hard_gmat)*grad
    grad_prior_acyclic_constr_total /= hparams["n_score_func_mc_samples"]
    grad_prior_acyclic_constr = -hparams["beta"] * grad_prior_acyclic_constr_total

    grad_z_log_likelihood = -stable_ratio(scores, log_fs)
    grad_z_log_prior_z = grad_prior_acyclic_constr + grad_log_prior_z_regularizer
      
    result = grad_z_log_prior_z + grad_z_log_likelihood

    return -result


def grad_theta_neg_log_joint(data, params, hparams):
    _n_mc = hparams["n_score_func_mc_samples"]
    distr = torch.distributions.Bernoulli(soft_gmat(params["z"], hparams))
    hard_gmats = distr.sample((_n_mc,))

    log_fs = torch.tensor(
        [
            log_joint(data, {**params, "hard_gmat": hard_gmats[i]}, hparams)
            for i in range(_n_mc)
        ]
    )
    
    # Compute gradients for each graph individually
    scores = []
    for i in range(_n_mc):
        grad = torch.autograd.grad(
            log_joint(data, {**params, "hard_gmat": hard_gmats[i]}, hparams),
            params["theta"],
            create_graph=True,
        )[0]
        scores.append(grad)
    
    grads = stable_ratio(scores, log_fs)
 
    return -grads


if __name__ == "__main__":
    torch.manual_seed(42)
    # Parameter
    N, d = 100, 5
    sigma = 0.1 * torch.ones((d,))
    gt_gmat = torch.diag(torch.ones(d - 1), diagonal=1)
    gt_theta = 5 * torch.rand((d, d))
    exog_noise = torch.normal(torch.zeros((N, d)), sigma * torch.ones((N, d)))
    x = exog_noise @ (gt_gmat * gt_theta)

    data = {"x": x, "y": {}}
    hparams = {
        "lr": 0.01,
        "sigma": sigma,
        "alpha": 0.01,
        "beta": 0.1,
        "tau": 1.0,
        "n_gumbel_mc_samples": 128,
        "n_grad_mc_samples": 16,
        "rho": 1.0,
        "minibatch_size": N,
        "n_score_func_mc_samples": 512,
    }

    update_hparams = lambda hps, t: {
        **hps,
        "alpha": hps["alpha"] * (t + 1) / t,
        "beta": hps["beta"] * (t + 1) / t,
        # "tau": hps["tau"] * (t + 1) / t,
    }

    iters = 500 + 1
    params = {
        "z": torch.randn((d, d, 2), requires_grad=True),
        "theta": torch.randn((d, d), requires_grad=True),
    }
    optimizer = torch.optim.RMSprop(list(params.values()), lr=hparams["lr"])
    for t in range(1, iters):
        optimizer.zero_grad()

        # Detaching params not relevant to the gradients being computed
        grad_z = grad_z_neg_log_joint(
            data, {**params, "theta": params["theta"].detach()}, hparams
        )
        grad_theta = grad_theta_neg_log_joint(
            data, {**params, "z": params["z"].detach()}, hparams
        )
        grads = {"z": grad_z, "theta": grad_theta}
        for name, param in params.items():
            param.grad = grads[name]

        optimizer.step()

        hparams = update_hparams(hparams, t)

        if t % 10 == 0:
            _params = {**params, "hard_gmat": hard_gmat(params["z"])}
            print(f"log_joint = {log_joint(data, _params, hparams)}")
            print(f"grad_z = {grad_z_neg_log_joint(data, _params, hparams)}")
            print(f"grad_theta = {grad_theta_neg_log_joint(data, _params, hparams)}")
            print(f"gmat = {_params['hard_gmat']}")

    # Plot results
    learnt_soft_gmat = soft_gmat(params["z"], {"alpha": 1.0})
    learnt_hard_gmat = (learnt_soft_gmat > 0.5).float().detach().numpy()
    learnt_theta = params["theta"].detach().numpy()
    for i, matrix in enumerate(
        [gt_gmat * gt_theta, learnt_hard_gmat * learnt_theta], 1
    ):
        plt.subplot(1, 2, i)
        plt.imshow(matrix, cmap="viridis")
        for (x, y), val in np.ndenumerate(matrix):
            plt.text(y, x, f"{val:.2f}", ha="center", va="center", color="white")
    plt.show()
