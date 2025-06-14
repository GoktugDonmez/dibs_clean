# data/graph_data.py
import torch
import igraph as ig
import numpy as np
import logging
from typing import Callable, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _topological_sort(g: ig.Graph) -> list:
    return g.topological_sorting(mode='out')

def _permute_adjacency(adj_matrix: torch.Tensor, perm: list) -> torch.Tensor:
    perm_tensor = torch.LongTensor(perm)
    return adj_matrix[perm_tensor][:, perm_tensor]

def linear_functional_relationship(parents: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum('bp,p->b', parents, weights)

def generate_erdos_renyi_dag(n_nodes: int, p_edge: float) -> Tuple[torch.Tensor, torch.Tensor]:
    adj_matrix = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    weights = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if torch.rand(1).item() < p_edge:
                adj_matrix[i, j] = 1.0
                weights[i, j] = torch.randn(1).item() * 2.0
    logging.info("Generated a valid Erdős-Rényi DAG constructively.")
    return adj_matrix, weights

def generate_scale_free_dag(n_nodes: int, m_edges: int) -> Tuple[torch.Tensor, torch.Tensor]:
    while True:
        g = ig.Graph.Barabasi(n=n_nodes, m=m_edges, directed=True)
        if g.is_dag():
            logging.info("Generated a valid Scale-Free DAG.")
            topo_order = _topological_sort(g)
            adj_matrix = torch.tensor(np.array(g.get_adjacency().data), dtype=torch.float32)
            adj_matrix = _permute_adjacency(adj_matrix, topo_order)
            
            weights = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
            edge_indices = adj_matrix.nonzero(as_tuple=True)
            num_edges = len(edge_indices[0])
            edge_weights = torch.randn(num_edges) * 2.0
            weights[edge_indices] = edge_weights
            
            return adj_matrix, weights

def sample_from_graph(adj_matrix: torch.Tensor, weights: torch.Tensor, n_samples: int, noise_std: float = 0.1) -> torch.Tensor:
    n_nodes = adj_matrix.shape[0]
    data = torch.zeros(n_samples, n_nodes)
    
    for i in range(n_nodes):
        parent_indices = (adj_matrix[:, i] == 1).nonzero(as_tuple=True)[0]
        noise = torch.randn(n_samples) * noise_std
        
        if len(parent_indices) == 0:
            node_values = noise
        else:
            parents_data = data[:, parent_indices]
            parent_weights = weights[parent_indices, i]
            deterministic_part = linear_functional_relationship(parents_data, parent_weights)
            node_values = deterministic_part + noise
            
        data[:, i] = node_values
        
    return data

def generate_synthetic_data(n_samples: int, n_nodes: int, graph_type: str, graph_params: dict, noise_std: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if graph_type == 'erdos-renyi':
        if 'p_edge' not in graph_params:
            raise ValueError("Missing 'p_edge' for 'erdos-renyi' graph.")
        adj_matrix, weights = generate_erdos_renyi_dag(n_nodes, graph_params['p_edge'])
    elif graph_type == 'scale-free':
        if 'm_edges' not in graph_params:
            raise ValueError("Missing 'm_edges' for 'scale-free' graph.")
        adj_matrix, weights = generate_scale_free_dag(n_nodes, graph_params['m_edges'])
    else:
        raise ValueError(f"Unsupported graph_type: {graph_type}")
        
    data = sample_from_graph(adj_matrix, weights, n_samples, noise_std)
    return adj_matrix, weights, data