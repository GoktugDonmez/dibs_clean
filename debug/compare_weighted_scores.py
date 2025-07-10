import torch


def _calculate_weighted_score_old(grad_samples: torch.Tensor, log_density_samples: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable computation of an expectation of gradients weighted by normalized probabilities.
    Computes: E_{p(x)}[grad(x)] = sum(p_i * grad_i) / sum(p_i)
    """
    eps = 1e-30
    print(f'OLD VERSION -----------------------------')
    while log_density_samples.dim() < grad_samples.dim():
        log_density_samples = log_density_samples.unsqueeze(-1)
    log_den = torch.logsumexp(log_density_samples, dim=0) - torch.log(torch.tensor(len(log_density_samples), dtype=log_density_samples.dtype, device=log_density_samples.device))
    print(f"log_den old: {log_den}")

    pos_grads = torch.where(grad_samples >= 0, grad_samples, 0.)
    print(f"pos_grads: {pos_grads}")
    neg_grads = torch.where(grad_samples < 0, -grad_samples, 0.)
    print(f"neg_grads: {neg_grads}")

    log_num_pos = torch.logsumexp(log_density_samples + torch.log(pos_grads + eps), dim=0) - torch.log(torch.tensor(len(log_density_samples), dtype=log_density_samples.dtype, device=log_density_samples.device))
    log_num_neg = torch.logsumexp(log_density_samples + torch.log(neg_grads + eps), dim=0) - torch.log(torch.tensor(len(log_density_samples), dtype=log_density_samples.dtype, device=log_density_samples.device))
    print(f"log_num_pos: {log_num_pos}")
    print(f"log_nxum_neg: {log_num_neg}")

    total_pos = torch.exp(log_num_pos - log_den)
    print(f"expected positive after exponentiation: {total_pos}")
    total_neg = torch.exp(log_num_neg - log_den)
    print(f"expected negative after exponentiation: {total_neg}")
    return total_pos - total_neg

def _calculate_weighted_score_new(grad_samples: torch.Tensor, log_density_samples: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable computation that preserves multi-dimensional gradient structure
    while using the positive/negative separation approach.
    """
    eps = 1e-30
    print(f'NEW VERSION -----------------------------')
    # Broadcast log_density_samples to match grad_samples dimensions
    while log_density_samples.dim() < grad_samples.dim():
        log_density_samples = log_density_samples.unsqueeze(-1)

    # Get the sample count (first dimension)
    n_samples = grad_samples.shape[0]
    
    # Calculate common denominator (only over sample dimension)
    log_denominator = torch.logsumexp(log_density_samples, dim=0) - torch.log(torch.tensor(n_samples, dtype=grad_samples.dtype, device=grad_samples.device))
    print(f"log_denominator: {log_denominator}")
    # Use torch.where to separate positive and negative gradients (preserves structure)
    pos_grads = torch.where(grad_samples >= 0, grad_samples, 0.)
    print(f"pos_grads: {pos_grads}")
    neg_grads = torch.where(grad_samples < 0, -grad_samples, 0.)  # Take absolute value
    print(f"neg_grads: {neg_grads}")
    # Count positive and negative elements per matrix position
    pos_mask = (grad_samples >= 0).float()
    neg_mask = (grad_samples < 0).float()
    print(f"pos_mask: {pos_mask}")
    print(f"neg_mask: {neg_mask}")
    pos_count = pos_mask.sum(dim=0)  # Count per matrix position
    neg_count = neg_mask.sum(dim=0)  # Count per matrix position
    total_count = torch.tensor(n_samples, dtype=grad_samples.dtype, device=grad_samples.device)
    print(f"pos_count: {pos_count}")
    print(f"neg_count: {neg_count}")
    print(f"total_count: {total_count}")
    # Calculate positive contribution using LSE
    pos_log_terms = torch.where(
        grad_samples >= 0,
        log_density_samples + torch.log(pos_grads + eps),
        torch.tensor(-float('inf'), device=grad_samples.device)
    )
    print(f"pos_log_terms: {pos_log_terms}")
    # Only compute LSE where we have positive elements
    log_numerator_pos = torch.where(
        pos_count > 0,
        torch.logsumexp(pos_log_terms, dim=0) - torch.log(pos_count),
        torch.tensor(-float('inf'), device=grad_samples.device)
    )
    print(f"log_numerator_pos: {log_numerator_pos}")
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
    print(f"neg_log_terms: {neg_log_terms}")
    # Only compute LSE where we have negative elements
    log_numerator_neg = torch.where(
        neg_count > 0,
        torch.logsumexp(neg_log_terms, dim=0) - torch.log(neg_count),
        torch.tensor(-float('inf'), device=grad_samples.device)
    )
    print(f"log_numerator_neg: {log_numerator_neg}")
    expected_neg = torch.where(
        neg_count > 0,
        (neg_count / total_count) * torch.exp(log_numerator_neg - log_denominator),
        torch.tensor(0.0, device=grad_samples.device)
    )

    print(f"expected value of positive and negative are same after exponentiation")
    print(f"expected_pos: {expected_pos}")
    print(f"expected_neg: {expected_neg}")
    return expected_pos - expected_neg

def soft_max_check(grad_samples: torch.Tensor, log_density_samples: torch.Tensor) -> torch.Tensor:

    weights = torch.softmax(log_density_samples, dim=0) 
    print(f"weights: {weights}")

    weighted_grads = weights * grad_samples
    print(f"weighted_grads: {weighted_grads}")
    print(f"sum of weighted_grads: {torch.sum(weighted_grads)}")
    return torch.sum(weighted_grads)


grads = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0], dtype=torch.float32)
log_density_samples = torch.tensor([1.0, 2.0, -3.0, 4.0, 5.0], dtype=torch.float32)
print(f"grads: {grads}")
print(f"log_density_samples: {log_density_samples}")

res_soft_max = soft_max_check(grads, log_density_samples)
res_old = _calculate_weighted_score_old(grads, log_density_samples)
res_new = _calculate_weighted_score_new(grads, log_density_samples)

print(f"res_soft_max: {res_soft_max}")
print(f"res_old: {res_old}")
print(f"res_new: {res_new}")

