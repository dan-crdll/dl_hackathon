from torch_geometric.data import Batch
from torch_geometric.utils import dropout_edge
import torch
import random


def graph_augmentation(data, drop_edge_prob=0.2, edge_noise_std=0.1):
    """
    Applica augmentazioni random a un singolo grafo.
    """
    # DropEdge: rimuove casualmente un sottoinsieme di archi
    edge_index, edge_attr = data.edge_index, data.edge_attr
    if drop_edge_prob > 0:
        edge_index, edge_attr = dropout_edge(edge_index, edge_attr, p=drop_edge_prob, training=True)
    
    data.edge_index = edge_index
    data.edge_attr = edge_attr

    # Feature Noise: aggiunge rumore gaussiano alle edge features
    if edge_attr is not None and edge_noise_std > 0:
        noise = torch.randn_like(edge_attr) * edge_noise_std
        data.edge_attr = edge_attr + noise

    return data


def collate_fn_with_augmentation(batch_list, drop_edge_prob=0.2, edge_noise_std=0.1):
    """
    Collate_fn per DataLoader che applica data augmentation su ogni grafo.
    """
    augmented_batch = []
    for data in batch_list:
        data_aug = graph_augmentation(data, drop_edge_prob=drop_edge_prob, edge_noise_std=edge_noise_std)
        augmented_batch.append(data_aug)
    
    return Batch.from_data_list(augmented_batch)