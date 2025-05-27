from torch import nn 
import torch 
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, GraphNorm


class EdgeEncoder(MessagePassing):
    def __init__(self, edge_dim, hidden_dim):
        super(EdgeEncoder, self).__init__(aggr='add')  
        self.node_mlp = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, hidden_dim),
            torch.nn.LeakyReLU(0.15),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = GraphNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_emb = self.edge_mlp(edge_attr)
        out = self.propagate(edge_index, x=x, edge_attr=edge_emb)
        out = self.norm(out + x, batch)
        return out 

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, edge_attr], dim=1)  
        return self.node_mlp(z)

    def update(self, aggr_out):
        return aggr_out
    

class Encoder(nn.Module):
    def __init__(self, edge_dim=7, hidden_dim=256, layers=10):
        super().__init__()

        self.node_embedding = nn.Embedding(1, hidden_dim)

        self.convs = nn.ModuleList()
        for i in range(layers):
            self.convs.append(EdgeEncoder(edge_dim, hidden_dim))

    def forward(self, data):
        edge_index, edge_attr, num_nodes, batch = data.edge_index, data.edge_attr, data.num_nodes, data.batch

        x = self.node_embedding(torch.zeros(num_nodes, dtype=torch.long, device=edge_attr.device))
        
        z = x 
        for conv in self.convs:
            z = conv(z, edge_index, edge_attr, batch)

        z = global_mean_pool(z, batch)
        return z
    

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, z):
        y = self.mlp(z)
        return y 
    