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


def log_joint_z(data, params, hparams):
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
    if params["theta"].requires_grad:
        return loglik_x + loglik_y + log_prior_theta

    return loglik_x + loglik_y + log_prob_g_given_z + log_prior_z + log_prior_theta

def log_joint_theta(data, params, hparams):
    """
    Computes the log of p(D, Theta | G), which is composed of:
    - Data likelihood: log p(D | G, Theta)
    - Theta prior:   log p(Theta | G)
    This is the required term for calculating the gradient with respect to Theta.
    """
    gmat = params["hard_gmat"]
    
    # Data likelihood: log p(D | G, Theta)
    # Using soft_gmat for the prediction but hard gmat for the prior on theta
    pred_mean_x = lambda x, ps: x @ (soft_gmat(params["z"], hparams) * ps["theta"])
    loglik_x = loglik_gaussian(
        data["x"], pred_mean_x(data["x"], params), hparams["sigma"]
    )
    loglik_y = 0.0 # Placeholder for expert feedback

    # Theta prior: log p(Theta | G)
    log_prior_theta = torch.sum(
        torch.distributions.Normal(
            torch.zeros_like(params["theta"]),
            torch.ones_like(params["theta"]),
        ).log_prob(params["theta"] * gmat) # The prior only applies to weights for existing edges
    )

    return loglik_x + loglik_y + log_prior_theta


def score_func_estimator_stable(params, hparams, log_f, normalized=True):
    # f is a function G, as in Equation B.8
    # _n_mc = hparams["n_score_func_mc_samples"]
    # _soft_gmat = soft_gmat(params["z"], hparams)
    # distr = torch.distributions.Bernoulli(_soft_gmat)
    # hard_gmats = distr.sample((_n_mc,))
    hard_gmats = params["hard_gmats"]
    _n_mc = hard_gmats.shape[0]

    # log joint densities, for all Gs
    log_fs = torch.func.vmap(
        log_f,
        randomness="different",
    )(hard_gmats)

    # grad_Z log p(G|Z) at one G
    score_func = lambda g: torch.autograd.grad(
        torch.distributions.Bernoulli(soft_gmat(params["z"], hparams))
        .log_prob(g)
        .sum(),
        params["z"],
        create_graph=True,
    )[0]

    # grad_Z log p(G|Z) for all Gs
    scores = torch.stack([score_func(g) for g in hard_gmats])

    while log_fs.dim() < scores.dim():
        log_fs = log_fs.unsqueeze(-1)

    log_numerator = torch.logsumexp(
        log_fs + torch.log(torch.abs(scores)) * torch.sign(scores),
        dim=0,
    )

    if normalized:
        # If the score function estimator has a denominator
        log_denominator = torch.logsumexp(log_fs, dim=0)
        result = torch.exp(log_numerator - log_denominator)
    else:
        result = torch.exp(log_numerator - torch.log(torch.tensor(_n_mc)))

    return result


def grad_z_neg_log_joint(data, params, hparams):
    _n_mc = params["hard_gmats"].shape[0]
    d = params["z"].shape[0]
    h_grad = (
        lambda g: acyclic_constr(g)
        * torch.autograd.grad(
            torch.distributions.Bernoulli(soft_gmat(params["z"], hparams))
            .log_prob(g)
            .sum(),
            params["z"],
            create_graph=True,
        )[0]
    )
    # Important to compute gradients of acyclicity constraint without logsumexp,
    # since the h(G) function can often be 0.
    h_grads = torch.stack([h_grad(params["hard_gmats"][i]) for i in range(_n_mc)])
    grad_z_log_prior_acyclic_constr = -hparams["beta"] * torch.mean(
        h_grads,
        dim=0,
    )
    grad_log_prior_z_regularizer = -(1 / torch.tensor(d)) * params["z"]
    grad_z_log_prior_z = grad_z_log_prior_acyclic_constr + grad_log_prior_z_regularizer

    grad_z_log_likelihood = score_func_estimator_stable(
        params,
        hparams,
        lambda g: log_joint_z(data, params | {"hard_gmat": g}, hparams),
        normalized=True,
    )
    result = grad_z_log_prior_z + grad_z_log_likelihood

    return -result


def grad_z_neg_log_joint_reparam(data, params, hparams):
    pass


def grad_theta_neg_log_joint(data, params, hparams):
    # _n_mc = hparams["n_score_func_mc_samples"]
    # distr = torch.distributions.Bernoulli(soft_gmat(params["z"], hparams))
    # hard_gmats = distr.sample((_n_mc,))
    # print(f"theta_hard_gmats={torch.mean(hard_gmats, dim=0)}")
    hard_gmats = params["hard_gmats"]
    soft_gmat = params["soft_gmat"]
    _n_mc = hard_gmats.shape[0]

    log_fs = torch.tensor(
        [
            log_joint_theta(data, {**params, "hard_gmat": hard_gmats[i]}, hparams)
            for i in range(_n_mc)
        ]
    )
    score_func = lambda g: torch.autograd.grad(
        log_joint_theta(data, {**params, "hard_gmat": g}, hparams),
        params["theta"],
        create_graph=True,
    )[0]

    # grad_Theta log p(Theta, D|G) for all Gs
    scores = torch.clamp(torch.stack([score_func(g) for g in hard_gmats]), min=1e-32)

    while log_fs.dim() < scores.dim():
        log_fs = log_fs.unsqueeze(-1)

    log_numerator = torch.logsumexp(
        log_fs + torch.log(torch.abs(scores)) * torch.sign(scores),
        dim=0,
    )
    log_denominator = torch.logsumexp(log_fs, dim=0)
    result = torch.exp(log_numerator - log_denominator)
    return -result


if __name__ == "__main__":
    torch.manual_seed(42)
    # Parameter
    N, d = 100, 3
    sigma = 0.1 * torch.ones((d,))
    gt_gmat = torch.diag(torch.ones(d - 1), diagonal=1)
    gt_theta = 5 * torch.rand((d, d))
    exog_noise = torch.normal(torch.zeros((N, d)), sigma * torch.ones((N, d)))
    x = exog_noise @ (gt_gmat * gt_theta)

    data = {"x": x, "y": {}}
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

    iters = 500 + 1
    params = {
        "z": torch.randn((d, d, 2), requires_grad=True),
        "theta": torch.randn((d, d), requires_grad=True),
    }
    optimizer = torch.optim.RMSprop(list(params.values()), lr=hparams["lr"])
    for t in range(1, iters):
        optimizer.zero_grad()
        _soft_gmat = soft_gmat(params["z"], hparams)
        distr = torch.distributions.Bernoulli(_soft_gmat)
        hard_gmats = distr.sample((hparams["n_score_func_mc_samples"],))

        # Detaching params not relevant to the gradients being computed
        grad_z = grad_z_neg_log_joint(
            data,
            {**params, "hard_gmats": hard_gmats, "soft_gmat": _soft_gmat, "theta": params["theta"].detach()},
            hparams,
        )
        grad_theta = grad_theta_neg_log_joint(
            data,
            {**params, "hard_gmats": hard_gmats, "soft_gmat": _soft_gmat, "z": params["z"].detach()},
            hparams,
        )
        grads = {"z": grad_z, "theta": grad_theta}
        for name, param in params.items():
            param.grad = grads[name]

        optimizer.step()

        hparams = update_hparams(hparams, t)

        if t % 10 == 0:
            grad_z_norm = torch.mean(torch.abs(grads["z"])).float().detach().numpy()
            grad_theta_norm = (
                torch.mean(torch.abs(grads["theta"])).float().detach().numpy()
            )
            print(f"{grad_z_norm=}, {grad_theta_norm=}")
            _params = {**params, "hard_gmat": hard_gmat(params["z"])}
            print(f"{log_joint_z(data, _params, hparams)=}")
            print(f"soft_gmat={soft_gmat(params['z'], hparams)}")
            print(f"alpha={hparams['alpha']}, beta={hparams['beta']}")

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
