# models/utils.py
import torch

def acyclic_constr(g: torch.Tensor, d: int) -> torch.Tensor:
    """H(G) from NOTEARS (Zheng et al.) with a series fallback for large *d*."""
    alpha = 1.0 / d
    eye = torch.eye(d, device=g.device, dtype=g.dtype)
    m = eye + alpha * g

    if d <= 10:
        return torch.trace(torch.linalg.matrix_power(m, d)) - d

    try:
        eigvals = torch.linalg.eigvals(m)
        return torch.sum(torch.real(eigvals ** d)) - d
    except RuntimeError:
        trace, p = torch.tensor(0.0, device=g.device, dtype=g.dtype), g.clone()
        for k in range(1, min(d + 1, 20)):
            trace += (alpha ** k) * torch.trace(p) / k
            if k < 19:
                p = p @ g
        return trace

def stable_mean(x: torch.Tensor, dim: int = 0, keepdim: bool = False) -> torch.Tensor:
    """Numerically stable mean for tensors spanning many orders of magnitude."""
    jitter = 1e-30
    if not x.is_floating_point():
        x = x.float()

    pos, neg = x.clamp(min=0), (-x).clamp(min=0)
    sum_pos = torch.exp(torch.logsumexp(torch.log(pos + jitter), dim=dim, keepdim=True))
    sum_neg = torch.exp(torch.logsumexp(torch.log(neg + jitter), dim=dim, keepdim=True))

    n = torch.tensor(x.shape[dim] if dim is not None else x.numel(), dtype=x.dtype, device=x.device)
    mean = (sum_pos - sum_neg) / (n + jitter)
    return mean if keepdim else mean.squeeze(dim)



# NUMERICAL STABILITY FUNCTIONS ARCHIVE

def logsumexp_v1(log_tensor: torch.Tensor) -> torch.Tensor:


    M = log_tensor.shape[0]
    logM = torch.log(torch.tensor(M, dtype=log_tensor.dtype, device=log_tensor.device))

    
    log_sum_exp = torch.logsumexp(log_tensor, dim=0)

    total = log_sum_exp - logM
    return total # torch.exp(total)

def manual_stable_gradient(log_p_tensor: torch.Tensor, grad_p_tensor: torch.Tensor) -> torch.Tensor:
# uses the logsumexp_v1 function to compute the stable gradient

    print(f'log density values and shape: {log_p_tensor}, {log_p_tensor.shape}')
    log_density_lse = torch.exp(torch.logsumexp(log_p_tensor, dim=0) - log_p_tensor.shape[0])  # logsumexp_v1(log_p_tensor)
    # logsumexp_v1(log_p_tensor)
    print(f'log density lse value: {log_density_lse}, shape: {log_density_lse.shape}')

    print('-' * 50)
    print(f'grad density values and shape: {grad_p_tensor}, {grad_p_tensor.shape}')
    grad_lse = logsumexp_v1(grad_p_tensor)
    print(f'grad density lse value: {grad_lse}, shape: {grad_lse.shape}')

    return torch.exp(logsumexp_v1(grad_p_tensor) - logsumexp_v1(log_p_tensor)) # grad_lse / log_density_lse[:, None]

