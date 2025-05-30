import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import (
    add_self_loops, 
    remove_self_loops, 
    to_undirected,
    subgraph,
    degree
)
import random
import numpy as np
from typing import Optional, Union, Tuple, List


class GraphAugmentation:
    """
    Graph augmentation class for PyTorch Geometric edge-only graphs (graphs without node features).
    Supports multiple augmentation strategies for graph neural network training on structural data.
    """
    
    def __init__(self, 
                 edge_drop_prob: float = 0.1,
                 node_drop_prob: float = 0.1,
                 edge_attr_noise_std: float = 0.1,
                 subgraph_ratio: float = 0.8):
        """
        Initialize augmentation parameters for edge-only graphs.
        
        Args:
            edge_drop_prob: Probability of dropping edges
            node_drop_prob: Probability of dropping nodes
            edge_attr_noise_std: Standard deviation for edge attribute noise
            subgraph_ratio: Ratio of nodes to keep in subgraph sampling
        """
        self.edge_drop_prob = edge_drop_prob
        self.node_drop_prob = node_drop_prob
        self.edge_attr_noise_std = edge_attr_noise_std
        self.subgraph_ratio = subgraph_ratio
    
    def _get_num_nodes(self, data: Data) -> int:
        """
        Infer number of nodes from edge_index since x is not available.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Number of nodes in the graph
        """
        if data.edge_index.numel() == 0:
            return 0
        return int(data.edge_index.max().item()) + 1
    
    def edge_dropout(self, data: Data, p: Optional[float] = None) -> Data:
        """
        Randomly drop edges from the graph.
        
        Args:
            data: PyTorch Geometric Data object (edge-only)
            p: Probability of dropping edges (uses self.edge_drop_prob if None)
        
        Returns:
            Augmented Data object with dropped edges
        """
        if p is None:
            p = self.edge_drop_prob
            
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        
        if num_edges == 0:
            return data.clone()
        
        # Get device from edge_index
        device = edge_index.device
        
        # Create mask for edges to keep (on same device)
        mask = torch.rand(num_edges, device=device) > p
        
        # Ensure at least one edge remains if possible
        if not mask.any() and num_edges > 0:
            mask[0] = True
        
        # Apply mask to edge_index
        new_edge_index = edge_index[:, mask]
        
        # Create new data object
        new_data = Data(
            edge_index=new_edge_index,
            y=getattr(data, 'y', None),
            batch=getattr(data, 'batch', None)
        )
        
        # Handle edge attributes if they exist
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            new_data.edge_attr = data.edge_attr[mask]
            
        return new_data
    
    def node_dropout(self, data: Data, p: Optional[float] = None) -> Data:
        """
        Randomly drop nodes and their associated edges.
        
        Args:
            data: PyTorch Geometric Data object (edge-only)
            p: Probability of dropping nodes (uses self.node_drop_prob if None)
        
        Returns:
            Augmented Data object with dropped nodes
        """
        if p is None:
            p = self.node_drop_prob
            
        num_nodes = self._get_num_nodes(data)
        
        if num_nodes == 0:
            return data.clone()
        
        # Get device from edge_index
        device = data.edge_index.device
        
        # Create mask for nodes to keep (on same device)
        keep_mask = torch.rand(num_nodes, device=device) > p
        keep_nodes = torch.where(keep_mask)[0]
        
        if len(keep_nodes) == 0:  # Ensure at least one node remains
            keep_nodes = torch.tensor([0], device=device)
        
        # Extract subgraph
        edge_index, edge_attr = subgraph(
            keep_nodes, 
            data.edge_index, 
            edge_attr=getattr(data, 'edge_attr', None),
            relabel_nodes=True,
            num_nodes=num_nodes
        )
        
        # Create new data object
        new_data = Data(
            edge_index=edge_index,
            y=getattr(data, 'y', None),
            batch=getattr(data, 'batch', None)
        )
        
        if edge_attr is not None:
            new_data.edge_attr = edge_attr
            
        return new_data
    
    def add_edge_noise(self, data: Data, edge_noise_std: Optional[float] = None) -> Data:
        """
        Add Gaussian noise to edge attributes (only applicable if edge attributes exist).
        
        Args:
            data: PyTorch Geometric Data object (edge-only)
            edge_noise_std: Standard deviation for edge attribute noise
        
        Returns:
            Augmented Data object with noisy edge attributes
        """
        if edge_noise_std is None:
            edge_noise_std = self.edge_attr_noise_std
            
        new_data = data.clone()
        
        # Add noise to edge attributes if they exist
        if hasattr(new_data, 'edge_attr') and new_data.edge_attr is not None:
            edge_noise = torch.randn_like(new_data.edge_attr) * edge_noise_std
            new_data.edge_attr = new_data.edge_attr + edge_noise
        
        return new_data
    
    def random_walk_subgraph(self, data: Data, 
                           walk_length: int = 10, 
                           num_walks: int = 5) -> Data:
        """
        Sample subgraph using random walks starting from random nodes.
        
        Args:
            data: PyTorch Geometric Data object (edge-only)
            walk_length: Length of each random walk
            num_walks: Number of random walks to perform
        
        Returns:
            Augmented Data object with subgraph
        """
        num_nodes = self._get_num_nodes(data)
        edge_index = data.edge_index
        
        if num_nodes == 0 or edge_index.size(1) == 0:
            return data.clone()
        
        # Get device from edge_index
        device = edge_index.device
        
        # Convert to adjacency list for efficient random walks
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].append(dst)
        
        visited_nodes = set()
        
        # Perform random walks
        for _ in range(num_walks):
            start_node = random.randint(0, num_nodes - 1)
            current_node = start_node
            visited_nodes.add(current_node)
            
            for _ in range(walk_length):
                if len(adj_list[current_node]) == 0:
                    break
                next_node = random.choice(adj_list[current_node])
                visited_nodes.add(next_node)
                current_node = next_node
        
        # Convert to tensor (on same device)
        keep_nodes = torch.tensor(list(visited_nodes), dtype=torch.long, device=device)
        
        if len(keep_nodes) == 0:
            keep_nodes = torch.tensor([0], device=device)
        
        # Extract subgraph
        edge_index, edge_attr = subgraph(
            keep_nodes,
            data.edge_index,
            edge_attr=getattr(data, 'edge_attr', None),
            relabel_nodes=True,
            num_nodes=num_nodes
        )
        
        new_data = Data(
            edge_index=edge_index,
            y=getattr(data, 'y', None),
            batch=getattr(data, 'batch', None)
        )
        
        if edge_attr is not None:
            new_data.edge_attr = edge_attr
            
        return new_data
    
    def edge_perturbation(self, data: Data, add_prob: float = 0.05) -> Data:
        """
        Add random edges to the graph (edge perturbation/addition).
        
        Args:
            data: PyTorch Geometric Data object (edge-only)
            add_prob: Probability of adding new edges
        
        Returns:
            Augmented Data object with additional edges
        """
        num_nodes = self._get_num_nodes(data)
        edge_index = data.edge_index
        
        if num_nodes <= 1:
            return data.clone()
        
        # Get device from edge_index
        device = edge_index.device
        
        # Calculate number of edges to add
        num_possible_edges = num_nodes * (num_nodes - 1)
        num_edges_to_add = int(add_prob * num_possible_edges)
        
        # Generate random edges
        new_edges = []
        existing_edges = set()
        
        # Store existing edges
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            existing_edges.add((src, dst))
        
        # Add random edges
        attempts = 0
        while len(new_edges) < num_edges_to_add and attempts < num_edges_to_add * 10:
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            
            if src != dst and (src, dst) not in existing_edges:
                new_edges.append([src, dst])
                existing_edges.add((src, dst))
            
            attempts += 1
        
        if new_edges:
            new_edge_tensor = torch.tensor(new_edges, dtype=torch.long, device=device).T
            combined_edge_index = torch.cat([edge_index, new_edge_tensor], dim=1)
        else:
            combined_edge_index = edge_index
        
        new_data = Data(
            edge_index=combined_edge_index,
            y=getattr(data, 'y', None),
            batch=getattr(data, 'batch', None)
        )
        
        # Handle edge attributes - pad with zeros for new edges
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if new_edges:
                new_edge_attrs = torch.zeros(len(new_edges), data.edge_attr.size(1), device=device)
                new_data.edge_attr = torch.cat([data.edge_attr, new_edge_attrs], dim=0)
            else:
                new_data.edge_attr = data.edge_attr
        
        return new_data
    
    def degree_based_node_sampling(self, data: Data, ratio: Optional[float] = None) -> Data:
        """
        Sample nodes based on their degree (preferentially keep high-degree nodes).
        
        Args:
            data: PyTorch Geometric Data object (edge-only)
            ratio: Ratio of nodes to keep (uses self.subgraph_ratio if None)
        
        Returns:
            Augmented Data object with degree-based subgraph
        """
        if ratio is None:
            ratio = self.subgraph_ratio
            
        num_nodes = self._get_num_nodes(data)
        
        if num_nodes == 0:
            return data.clone()
            
        node_degrees = degree(data.edge_index[0], num_nodes=num_nodes)
        
        # Number of nodes to keep
        num_keep = max(1, int(ratio * num_nodes))
        
        # Sample based on degree distribution
        # Add small epsilon to avoid zero probabilities
        probabilities = F.softmax(node_degrees.float() + 1e-8, dim=0)
        keep_nodes = torch.multinomial(probabilities, num_keep, replacement=False)
        
        # Extract subgraph
        edge_index, edge_attr = subgraph(
            keep_nodes,
            data.edge_index,
            edge_attr=getattr(data, 'edge_attr', None),
            relabel_nodes=True,
            num_nodes=num_nodes
        )
        
        new_data = Data(
            edge_index=edge_index,
            y=getattr(data, 'y', None),
            batch=getattr(data, 'batch', None)
        )
        
        if edge_attr is not None:
            new_data.edge_attr = edge_attr
            
        return new_data
    
    def add_self_loops_augmentation(self, data: Data) -> Data:
        """
        Add self-loops to all nodes as an augmentation strategy.
        
        Args:
            data: PyTorch Geometric Data object (edge-only)
        
        Returns:
            Augmented Data object with self-loops added
        """
        edge_index = data.edge_index
        num_nodes = self._get_num_nodes(data)
        
        if num_nodes == 0:
            return data.clone()
        
        # Add self-loops
        edge_index_with_loops, edge_attr_with_loops = add_self_loops(
            edge_index, 
            edge_attr=getattr(data, 'edge_attr', None),
            num_nodes=num_nodes
        )
        
        new_data = Data(
            edge_index=edge_index_with_loops,
            y=getattr(data, 'y', None),
            batch=getattr(data, 'batch', None)
        )
        
        if edge_attr_with_loops is not None:
            new_data.edge_attr = edge_attr_with_loops
            
        return new_data
    
    def remove_self_loops_augmentation(self, data: Data) -> Data:
        """
        Remove self-loops as an augmentation strategy.
        
        Args:
            data: PyTorch Geometric Data object (edge-only)
        
        Returns:
            Augmented Data object with self-loops removed
        """
        edge_index = data.edge_index
        
        # Remove self-loops
        edge_index_no_loops, edge_attr_no_loops = remove_self_loops(
            edge_index, 
            edge_attr=getattr(data, 'edge_attr', None)
        )
        
        new_data = Data(
            edge_index=edge_index_no_loops,
            y=getattr(data, 'y', None),
            batch=getattr(data, 'batch', None)
        )
        
        if edge_attr_no_loops is not None:
            new_data.edge_attr = edge_attr_no_loops
            
        return new_data
    
    def augment(self, data: Data, 
                augmentation_types: List[str] = ['edge_dropout', 'node_dropout', 'edge_noise']) -> Data:
        """
        Apply multiple augmentations randomly.
        
        Args:
            data: PyTorch Geometric Data object (edge-only)
            augmentation_types: List of augmentation types to apply
        
        Returns:
            Augmented Data object
        """
        augmented_data = data.clone()
        
        # Randomly select and apply augmentations
        selected_aug = random.choice(augmentation_types)
        
        if selected_aug == 'edge_dropout':
            augmented_data = self.edge_dropout(augmented_data)
        elif selected_aug == 'node_dropout':
            augmented_data = self.node_dropout(augmented_data)
        elif selected_aug == 'edge_noise':
            augmented_data = self.add_edge_noise(augmented_data)
        elif selected_aug == 'random_walk':
            augmented_data = self.random_walk_subgraph(augmented_data)
        elif selected_aug == 'edge_perturbation':
            augmented_data = self.edge_perturbation(augmented_data)
        elif selected_aug == 'degree_sampling':
            augmented_data = self.degree_based_node_sampling(augmented_data)
        elif selected_aug == 'add_self_loops':
            augmented_data = self.add_self_loops_augmentation(augmented_data)
        elif selected_aug == 'remove_self_loops':
            augmented_data = self.remove_self_loops_augmentation(augmented_data)
        
        return augmented_data


# Example usage and utility functions
def create_edge_augmentation_pipeline(data: Data, num_augmentations: int = 5) -> List[Data]:
    """
    Create multiple augmented versions of an edge-only graph.
    
    Args:
        data: Original PyTorch Geometric Data object (edge-only)
        num_augmentations: Number of augmented versions to create
    
    Returns:
        List of augmented Data objects
    """
    augmenter = EdgeGraphAugmentation()
    
    augmentation_types = [
        'edge_dropout', 'node_dropout', 'edge_noise',
        'random_walk', 'edge_perturbation', 'degree_sampling',
        'add_self_loops', 'remove_self_loops'
    ]
    
    augmented_graphs = []
    for _ in range(num_augmentations):
        aug_data = augmenter.augment(data, augmentation_types)
        augmented_graphs.append(aug_data)
    
    return augmented_graphs


# Example usage
if __name__ == "__main__":
    # Create sample edge-only graph data
    num_nodes = 100
    edge_index = torch.randint(0, num_nodes, (2, 200))  # Random edges
    edge_attr = torch.randn(200, 4)  # Optional edge attributes
    
    # Edge-only graph (no node features)
    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    
    # Initialize augmenter
    augmenter = EdgeGraphAugmentation(
        edge_drop_prob=0.2,
        node_drop_prob=0.1,
        edge_attr_noise_std=0.05
    )
    
    # Infer number of nodes from edge_index
    num_nodes_inferred = augmenter._get_num_nodes(data)
    
    # Apply different augmentations
    print(f"Original graph: {num_nodes_inferred} nodes (inferred), {data.num_edges} edges")
    
    aug1 = augmenter.edge_dropout(data)
    print(f"After edge dropout: {augmenter._get_num_nodes(aug1)} nodes, {aug1.num_edges} edges")
    
    aug2 = augmenter.node_dropout(data)
    print(f"After node dropout: {augmenter._get_num_nodes(aug2)} nodes, {aug2.num_edges} edges")
    
    aug3 = augmenter.add_edge_noise(data)
    print(f"After adding edge noise: {augmenter._get_num_nodes(aug3)} nodes, {aug3.num_edges} edges")
    
    aug4 = augmenter.random_walk_subgraph(data)
    print(f"After random walk: {augmenter._get_num_nodes(aug4)} nodes, {aug4.num_edges} edges")
    
    aug5 = augmenter.add_self_loops_augmentation(data)
    print(f"After adding self-loops: {augmenter._get_num_nodes(aug5)} nodes, {aug5.num_edges} edges")
    
    # Example without edge attributes
    data_no_attr = Data(edge_index=edge_index)
    aug6 = augmenter.edge_perturbation(data_no_attr)
    print(f"After edge perturbation (no attr): {augmenter._get_num_nodes(aug6)} nodes, {aug6.num_edges} edges")
    
    # Create augmentation pipeline
    augmented_graphs = create_edge_augmentation_pipeline(data, num_augmentations=3)
    print(f"\nCreated {len(augmented_graphs)} augmented versions")
    for i, aug_graph in enumerate(augmented_graphs):
        print(f"Augmented graph {i+1}: {augmenter._get_num_nodes(aug_graph)} nodes, {aug_graph.num_edges} edges")