#!/usr/bin/env python3

import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
import igraph as ig
from typing import Dict, Any, Tuple
import argparse
import os

# Import the fixed DiBS implementation
from models.dibs_fixed import (
    DiBSFixed, update_hparams, log_joint, acyclic_constr, soft_gmat
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def generate_erdos_renyi_data(n_nodes: int, p_edge: float, num_samples: int, obs_noise_std: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate data from an Erdős-Rényi DAG.
    
    Args:
        n_nodes: Number of nodes
        p_edge: Edge probability
        num_samples: Number of data samples
        obs_noise_std: Observation noise standard deviation
        
    Returns:
        X_data: (num_samples, n_nodes) observed data
        G_true: (n_nodes, n_nodes) true adjacency matrix
        Theta_true: (n_nodes, n_nodes) true weight matrix
    """
    # Generate random DAG using igraph
    attempts = 0
    max_attempts = 100
    
    while attempts < max_attempts:
        try:
            g = ig.Graph.Erdos_Renyi(n=n_nodes, p=p_edge, directed=True)
            if g.is_dag() and len(g.es) > 0:  # Ensure it's a DAG with at least one edge
                break
        except:
            pass
        attempts += 1
    
    if attempts >= max_attempts:
        log.warning(f"Failed to generate connected DAG after {max_attempts} attempts, using chain graph")
        # Fallback to chain graph
        G_true = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
        for i in range(n_nodes - 1):
            G_true[i, i + 1] = 1.0
    else:
        # Convert to adjacency matrix
        adj_matrix = torch.tensor(np.array(g.get_adjacency().data), dtype=torch.float32)
        G_true = adj_matrix
    
    # Generate random weights for existing edges
    Theta_true = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    edge_indices = (G_true == 1).nonzero(as_tuple=True)
    
    if len(edge_indices[0]) > 0:
        # Random weights between -2 and 2
        edge_weights = (torch.rand(len(edge_indices[0])) - 0.5) * 4.0
        Theta_true[edge_indices] = edge_weights
    
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

def generate_chain_data(n_nodes: int, num_samples: int, obs_noise_std: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate data from a simple chain: X1 -> X2 -> ... -> Xn
    """
    G_true = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    Theta_true = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    
    # Create chain structure
    for i in range(n_nodes - 1):
        G_true[i, i + 1] = 1.0
        # Random weight between -2 and 2
        Theta_true[i, i + 1] = (torch.rand(1).item() - 0.5) * 4.0
    
    # Generate data
    X_data = torch.zeros(num_samples, n_nodes)
    X_data[:, 0] = torch.randn(num_samples) * obs_noise_std
    
    for i in range(1, n_nodes):
        parent_value = X_data[:, i - 1]
        weight = Theta_true[i - 1, i]
        noise = torch.randn(num_samples) * obs_noise_std
        X_data[:, i] = weight * parent_value + noise
    
    return X_data, G_true, Theta_true

def compute_metrics(learned_adj: torch.Tensor, true_adj: torch.Tensor) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    """
    # Structural Hamming Distance
    shd = torch.sum(torch.abs(learned_adj - true_adj)).item()
    
    # True/False Positives/Negatives
    tp = torch.sum((learned_adj == 1) & (true_adj == 1)).item()
    fp = torch.sum((learned_adj == 1) & (true_adj == 0)).item()
    fn = torch.sum((learned_adj == 0) & (true_adj == 1)).item()
    tn = torch.sum((learned_adj == 0) & (true_adj == 0)).item()
    
    # Precision, Recall, F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'shd': shd,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

def train_dibs(config: Dict[str, Any]) -> Tuple[DiBSFixed, Dict[str, Any]]:
    """
    Train the fixed DiBS model.
    """
    # Set seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = config['device']
    log.info(f"Training on device: {device}")
    
    # Generate data
    if config['graph_type'] == 'erdos_renyi':
        X_data, G_true, Theta_true = generate_erdos_renyi_data(
            config['n_nodes'], config['p_edge'], config['num_samples'], config['obs_noise_std']
        )
        log.info(f"Generated Erdős-Rényi graph with {config['n_nodes']} nodes, p_edge={config['p_edge']}")
    elif config['graph_type'] == 'chain':
        X_data, G_true, Theta_true = generate_chain_data(
            config['n_nodes'], config['num_samples'], config['obs_noise_std']
        )
        log.info(f"Generated chain graph with {config['n_nodes']} nodes")
    else:
        raise ValueError(f"Unknown graph_type: {config['graph_type']}")
    
    # Move data to device
    X_data = X_data.to(device)
    G_true = G_true.to(device)
    Theta_true = Theta_true.to(device)
    
    data = {'x': X_data}
    
    log.info(f"Ground truth adjacency matrix:\n{G_true.cpu().numpy()}")
    log.info(f"Ground truth weights matrix:\n{Theta_true.cpu().numpy()}")
    log.info(f"Number of edges: {(G_true > 0).sum().item()}")
    
    # Initialize model
    model = DiBSFixed(config['n_nodes'], config['latent_dim'], device=device)
    
    # Initialize hyperparameters
    sigma_z = 1.0 / np.sqrt(config['latent_dim'])
    hparams = {
        'alpha': 0.2,  # Will be annealed
        'beta': 1.0,   # Will be annealed
        'alpha_base': config['alpha_base'],
        'beta_base': config['beta_base'],
        'sigma_z': sigma_z,
        'sigma_obs': config['obs_noise_std'],
        'theta_prior_sigma': config['theta_prior_sigma'],
        'n_mc_samples': config['n_mc_samples'],
        'total_steps': config['num_iterations']
    }
    
    # Setup optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config['lr'])
    
    # Training loop
    log.info("Starting training...")
    log_joint_history = []
    shd_history = []
    
    for t in range(1, config['num_iterations'] + 1):
        optimizer.zero_grad()
        
        # Update hyperparameters
        hparams = update_hparams(hparams, t)
        
        # Forward pass
        outputs = model(data, hparams)
        
        # Set gradients for gradient ascent (optimizer does descent)
        model.z.grad = -outputs['grad_z']
        model.theta.grad = -outputs['grad_theta']
        
        # Optimizer step
        optimizer.step()
        
        # Logging and evaluation
        if t % config['print_interval'] == 0 or t == config['num_iterations']:
            with torch.no_grad():
                log_joint_val = outputs['log_joint'].item()
                grad_z_norm = torch.norm(outputs['grad_z']).item()
                grad_theta_norm = torch.norm(outputs['grad_theta']).item()
                
                # Get current learned graph
                learned_soft = outputs['soft_adj']
                print(f"learned_soft:\n{learned_soft}")
                learned_hard = model.get_hard_adjacency(hparams, threshold=0.5)
                
                # Compute metrics
                #metrics = compute_metrics(learned_hard.cpu(), G_true.cpu())
                
                # Acyclicity check
                acyc_val = acyclic_constr(learned_soft, config['n_nodes']).item()
                
                log.info(f"--- Iteration {t}/{config['num_iterations']} ---")
                log.info(f"Log-joint: {log_joint_val:.2f}")
                log.info(f"Grad norms - Z: {grad_z_norm:.2e}, Theta: {grad_theta_norm:.2e}")
                log.info(f"Hyperparams - Alpha: {hparams['alpha']:.3f}, Beta: {hparams['beta']:.3f}")
                #log.info(f"Metrics - SHD: {metrics['shd']:.0f}, F1: {metrics['f1']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
                log.info(f"Acyclicity constraint: {acyc_val:.3f}")
                log.info(f"Edge prob range: [{learned_soft.min().item():.3f}, {learned_soft.max().item():.3f}]")
                
                # Store history
                log_joint_history.append(log_joint_val)
                #shd_history.append(metrics['shd'])
    
    # Final evaluation
    with torch.no_grad():
        learned_soft_final = soft_gmat(model.z, hparams['alpha'])
        learned_hard_final = model.get_hard_adjacency(hparams, threshold=0.5)
        final_metrics = compute_metrics(learned_hard_final.cpu(), G_true.cpu())
        
        log.info(f"\n{'='*60}")
        log.info(f"FINAL RESULTS")
        log.info(f"{'='*60}")
        log.info(f"Ground Truth Adjacency:\n{G_true.cpu().numpy()}")
        log.info(f"Learned Soft Probabilities:\n{learned_soft_final.cpu().numpy()}")
        log.info(f"Learned Hard Adjacency:\n{learned_hard_final.cpu().numpy()}")
        log.info(f"Final Metrics: {final_metrics}")
        
        # Check if learning was successful
        success = final_metrics['shd'] == 0
        log.info(f"Learning Success: {success}")
    
    results = {
        'model': model,
        'G_true': G_true.cpu(),
        'Theta_true': Theta_true.cpu(),
        'G_learned_soft': learned_soft_final.cpu(),
        'G_learned_hard': learned_hard_final.cpu(),
        'final_metrics': final_metrics,
        'log_joint_history': log_joint_history,
        'shd_history': shd_history,
        'success': success
    }
    
    return model, results

def plot_results(results: Dict[str, Any], save_path: str = None):
    """
    Plot training results and learned graphs.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Training curves
    axes[0, 0].plot(results['log_joint_history'])
    axes[0, 0].set_title('Log Joint Probability')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Log Joint')
    
    axes[0, 1].plot(results['shd_history'])
    axes[0, 1].set_title('Structural Hamming Distance')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('SHD')
    
    # Plot 2: Ground truth adjacency
    im1 = axes[0, 2].imshow(results['G_true'].numpy(), cmap='Blues', vmin=0, vmax=1)
    axes[0, 2].set_title('Ground Truth Adjacency')
    plt.colorbar(im1, ax=axes[0, 2])
    
    # Plot 3: Learned soft probabilities
    im2 = axes[1, 0].imshow(results['G_learned_soft'].numpy(), cmap='Blues', vmin=0, vmax=1)
    axes[1, 0].set_title('Learned Soft Probabilities')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Plot 4: Learned hard adjacency
    im3 = axes[1, 1].imshow(results['G_learned_hard'].numpy(), cmap='Blues', vmin=0, vmax=1)
    axes[1, 1].set_title('Learned Hard Adjacency')
    plt.colorbar(im3, ax=axes[1, 1])
    
    # Plot 5: Difference
    diff = torch.abs(results['G_true'] - results['G_learned_hard'])
    im4 = axes[1, 2].imshow(diff.numpy(), cmap='Reds', vmin=0, vmax=1)
    axes[1, 2].set_title('Difference (Errors)')
    plt.colorbar(im4, ax=axes[1, 2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.info(f"Results saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train Fixed DiBS on Causal Discovery')
    parser.add_argument('--n_nodes', type=int, default=4, choices=[3, 4, 5, 6], help='Number of nodes')
    parser.add_argument('--graph_type', type=str, default='erdos_renyi', choices=['erdos_renyi', 'chain'], help='Graph type')
    parser.add_argument('--p_edge', type=float, default=0.4, help='Edge probability for Erdős-Rényi')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of data samples')
    parser.add_argument('--num_iterations', type=int, default=2000, help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--alpha_base', type=float, default=1.0, help='Base value for alpha annealing')
    parser.add_argument('--beta_base', type=float, default=10.0, help='Base value for beta annealing')
    parser.add_argument('--n_mc_samples', type=int, default=128, help='Number of Monte Carlo samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_results', type=str, default=None, help='Path to save results plot')
    
    args = parser.parse_args()
    
    config = {
        'n_nodes': args.n_nodes,
        'graph_type': args.graph_type,
        'p_edge': args.p_edge,
        'num_samples': args.num_samples,
        'obs_noise_std': 0.1,
        'latent_dim': args.n_nodes,  # Set k = d for sufficient capacity
        'alpha_base': args.alpha_base,
        'beta_base': args.beta_base,
        'theta_prior_sigma': 1.0,
        'n_mc_samples': args.n_mc_samples,
        'num_iterations': args.num_iterations,
        'lr': args.lr,
        'print_interval': max(1, args.num_iterations // 20),  # Print 20 times during training
        'seed': args.seed,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    log.info(f"Configuration: {config}")
    
    # Train model
    model, results = train_dibs(config)
    
    # Plot results
    if args.save_results:
        save_dir = os.path.dirname(args.save_results)
        if save_dir:  # Only create directory if there is one
            os.makedirs(save_dir, exist_ok=True)
    plot_results(results, args.save_results)
    
    # Success message
    if results['success']:
        log.info("🎉 SUCCESS: Model correctly learned the ground truth graph!")
    else:
        log.info(f"❌ Model did not perfectly recover the graph. SHD: {results['final_metrics']['shd']}")

if __name__ == '__main__':
    main()