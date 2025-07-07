#!/usr/bin/env python3

import numpy as np
from typing import Dict, Any, List

def sigmoid(x):
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

def acyclic_constraint_numpy(g: np.ndarray) -> float:
    """
    Compute acyclicity constraint h(G) = tr((I + αG)^d) - d
    """
    d = g.shape[0]
    alpha = 1.0 / d
    eye = np.eye(d)
    m = eye + alpha * g
    
    if d <= 10:
        # Use matrix power for small graphs
        return np.trace(np.linalg.matrix_power(m, d)) - d
    else:
        # Use eigenvalues for larger graphs
        eigvals = np.linalg.eigvals(m)
        return np.sum(np.real(eigvals ** d)) - d

def soft_adjacency_numpy(z: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute soft adjacency matrix from latent variables.
    z: (d, k, 2) array
    """
    u, v = z[:, :, 0], z[:, :, 1]  # (d, k)
    raw_scores = alpha * np.einsum('ik,jk->ij', u, v)  # (d, d)
    
    # Mask diagonal
    d = z.shape[0]
    diag_mask = 1.0 - np.eye(d)
    masked_scores = raw_scores * diag_mask
    
    return sigmoid(masked_scores)

def score_function_numpy(z: np.ndarray, g: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute ∇_z log q(G|z) analytically.
    """
    g_soft = soft_adjacency_numpy(z, alpha)
    diff = g - g_soft  # (d, d)
    u, v = z[:, :, 0], z[:, :, 1]  # (d, k)
    
    # Gradients w.r.t. u and v
    grad_u = alpha * np.einsum('ij,jk->ik', diff, v)  # (d, k)
    grad_v = alpha * np.einsum('ij,ik->jk', diff, u)  # (d, k)
    
    return np.stack([grad_u, grad_v], axis=-1)  # (d, k, 2)

def stable_ratio_numpy(grad_samples: List[np.ndarray], log_density_samples: List[float]) -> np.ndarray:
    """
    Stable ratio estimator for gradients.
    """
    eps = 1e-30
    
    # Convert to arrays
    log_p = np.array(log_density_samples)  # (M,)
    grads = np.stack(grad_samples)  # (M, ...)
    
    # Expand log_p to match grads dimensions
    while log_p.ndim < grads.ndim:
        log_p = np.expand_dims(log_p, -1)
    
    # Compute stable softmax weights
    log_p_max = np.max(log_p, axis=0, keepdims=True)
    log_p_shifted = log_p - log_p_max
    weights = np.exp(log_p_shifted)
    weights = weights / (np.sum(weights, axis=0, keepdims=True) + eps)
    
    # Separate positive and negative gradients for stability
    pos_mask = grads >= 0
    neg_mask = grads < 0
    
    # Weighted sum for positive gradients
    pos_grads = np.where(pos_mask, grads, 0)
    pos_weighted = np.sum(weights * pos_grads, axis=0)
    
    # Weighted sum for negative gradients
    neg_grads = np.where(neg_mask, -grads, 0)
    neg_weighted = np.sum(weights * neg_grads, axis=0)
    
    return pos_weighted - neg_weighted

def test_gradient_computation():
    """
    Test the gradient computation logic.
    """
    print("Testing DiBS Gradient Computation Logic")
    print("="*50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test parameters
    d = 4  # 4 nodes
    k = 4  # latent dimension
    alpha = 1.0
    num_samples = 10
    
    # Initialize latent variables
    z = np.random.randn(d, k, 2) * 0.1
    
    print(f"Testing with {d} nodes, latent dim {k}")
    
    # Test 1: Soft adjacency computation
    g_soft = soft_adjacency_numpy(z, alpha)
    print(f"\nTest 1 - Soft Adjacency Matrix:")
    print(f"Shape: {g_soft.shape}")
    print(f"Range: [{g_soft.min():.3f}, {g_soft.max():.3f}]")
    print(f"Diagonal should be zero: {np.allclose(np.diag(g_soft), 0)}")
    
    # Test 2: Acyclicity constraint
    h_val = acyclic_constraint_numpy(g_soft)
    print(f"\nTest 2 - Acyclicity Constraint:")
    print(f"h(G_soft) = {h_val:.6f}")
    
    # Test 3: Score function computation
    # Sample a few hard graphs
    hard_graphs = []
    scores = []
    log_densities = []
    
    for i in range(num_samples):
        # Sample hard graph
        g_hard = (np.random.rand(d, d) < g_soft).astype(float)
        np.fill_diagonal(g_hard, 0)  # Ensure no self-loops
        
        # Compute score function
        score = score_function_numpy(z, g_hard, alpha)
        
        # Dummy log density (in real implementation this would be log likelihood)
        log_density = -np.sum((g_hard - g_soft)**2)  # Simplified
        
        hard_graphs.append(g_hard)
        scores.append(score)
        log_densities.append(log_density)
    
    print(f"\nTest 3 - Score Function:")
    print(f"Generated {num_samples} samples")
    print(f"Score shape: {scores[0].shape}")
    print(f"Score range: [{np.min([s.min() for s in scores]):.3f}, {np.max([s.max() for s in scores]):.3f}]")
    
    # Test 4: Stable ratio estimator
    stable_grad = stable_ratio_numpy(scores, log_densities)
    print(f"\nTest 4 - Stable Ratio Estimator:")
    print(f"Gradient shape: {stable_grad.shape}")
    print(f"Gradient norm: {np.linalg.norm(stable_grad):.6f}")
    print(f"Gradient finite: {np.all(np.isfinite(stable_grad))}")
    
    # Test 5: Compare with simple average
    simple_avg = np.mean(scores, axis=0)
    print(f"\nTest 5 - Comparison with Simple Average:")
    print(f"Stable ratio norm: {np.linalg.norm(stable_grad):.6f}")
    print(f"Simple average norm: {np.linalg.norm(simple_avg):.6f}")
    print(f"Relative difference: {np.linalg.norm(stable_grad - simple_avg) / (np.linalg.norm(simple_avg) + 1e-8):.6f}")
    
    # Test 6: Numerical stability
    print(f"\nTest 6 - Numerical Stability:")
    
    # Test with extreme log densities
    extreme_log_densities = [-1000.0, -500.0, 0.0, -800.0, -300.0]
    extreme_scores = scores[:5]  # Use first 5 scores
    
    try:
        extreme_grad = stable_ratio_numpy(extreme_scores, extreme_log_densities)
        print(f"Stable with extreme values: {np.all(np.isfinite(extreme_grad))}")
        print(f"Extreme gradient norm: {np.linalg.norm(extreme_grad):.6f}")
    except:
        print("Failed with extreme values")
    
    # Test 7: Chain graph learning simulation
    print(f"\nTest 7 - Chain Graph Simulation:")
    
    # True chain graph: 0 -> 1 -> 2 -> 3
    G_true = np.zeros((d, d))
    for i in range(d-1):
        G_true[i, i+1] = 1
    
    print(f"True chain graph:\n{G_true}")
    
    # Simulate learning iterations
    learning_rate = 0.01
    for iteration in range(50):
        # Get current soft graph
        g_soft_current = soft_adjacency_numpy(z, alpha)
        
        # Sample hard graphs and compute gradients
        sample_scores = []
        sample_densities = []
        
        for _ in range(5):  # Small number for demo
            g_hard = (np.random.rand(d, d) < g_soft_current).astype(float)
            np.fill_diagonal(g_hard, 0)
            
            # Score function
            score = score_function_numpy(z, g_hard, alpha)
            
            # Simplified likelihood: encourage edges that match true graph
            log_density = -10 * np.sum(np.abs(g_hard - G_true))
            
            sample_scores.append(score)
            sample_densities.append(log_density)
        
        # Compute gradient
        grad = stable_ratio_numpy(sample_scores, sample_densities)
        
        # Prior gradient (regularization)
        grad_prior = -z / 1.0
        
        # Update
        z += learning_rate * (grad + grad_prior)
        
        if iteration % 10 == 0:
            current_soft = soft_adjacency_numpy(z, alpha)
            mse = np.mean((current_soft - G_true)**2)
            print(f"Iteration {iteration}: MSE = {mse:.6f}")
    
    final_soft = soft_adjacency_numpy(z, alpha)
    final_hard = (final_soft > 0.5).astype(float)
    
    print(f"\nFinal Results:")
    print(f"True graph:\n{G_true}")
    print(f"Final soft probabilities:\n{np.round(final_soft, 3)}")
    print(f"Final hard graph:\n{final_hard}")
    print(f"Structural Hamming Distance: {np.sum(np.abs(final_hard - G_true))}")
    
    success = np.allclose(final_hard, G_true)
    print(f"Learning successful: {success}")
    
    return success

if __name__ == "__main__":
    success = test_gradient_computation()
    
    print(f"\n{'='*50}")
    if success:
        print("✅ GRADIENT LOGIC TEST PASSED!")
        print("The fixed DiBS implementation should work correctly.")
    else:
        print("⚠️  GRADIENT LOGIC TEST SHOWS ISSUES")
        print("There may still be some problems to address.")
    print(f"{'='*50}")