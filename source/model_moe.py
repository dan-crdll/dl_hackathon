import lightning as L
from torchmetrics import Accuracy
from source.loss_fn import SCELoss, FocalLoss, GCODLoss
import torch
import torch.nn.functional as F
from source.conv import GNN
from torch import nn
from source.layers import Encoder, Classifier

class LitClassifier(L.LightningModule):
    def __init__(self, split, num_experts=10, k=4):
        super().__init__()
        self.num_experts = num_experts
        self.k = k  # Numero di esperti da selezionare (top-k)
        
        self.gate_module = GNN(num_experts, 2, 256)
        self.experts = nn.ModuleList([
            Encoder(layers=2) for _ in range(num_experts)
        ])
        self.classifier = Classifier(256, 1024, 6, 0.1)
        self.loss_fn = GCODLoss(8)
        self.acc_fn = Accuracy('multiclass', num_classes=6)
        
        # Load balancing parameters
        self.load_balancing_weight = 0.01  # Weight per la loss di bilanciamento
        
        # Logging
        self.h = []
        self.loss_epoch = []
        self.acc_epoch = []
        self.split = split
        self.loss = []
        self.acc = []
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)
    
    def compute_load_balancing_loss(self, gate_scores, top_k_indices):
        """Calcola la loss per bilanciare l'uso degli esperti"""
        batch_size = gate_scores.size(0)
        
        # Conta l'uso di ogni esperto
        expert_usage = torch.zeros(self.num_experts, device=gate_scores.device)
        for idx in top_k_indices.flatten():
            expert_usage[idx] += 1
        
        # Normalizza per il numero totale di selezioni
        expert_usage = expert_usage / (batch_size * self.k)
        
        # Target uniforme
        target_usage = 1.0 / self.num_experts
        
        # Loss di bilanciamento (varianza dall'uso uniforme)
        load_loss = torch.sum((expert_usage - target_usage) ** 2)
        return load_loss
    
    def forward(self, data):
        batch_size = data.batch.max().item() + 1 if hasattr(data, 'batch') else data.x.size(0)
        
        # 1. Compute gating scores
        gate_scores = self.gate_module(data)  # [BSZ, num_experts]
        
        # 2. Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)  # [BSZ, k]
        
        # 3. Trova tutti gli esperti unici necessari per questo batch
        unique_experts = torch.unique(top_k_indices)
        
        # 4. Pre-calcola gli output solo per gli esperti necessari
        expert_cache = {}
        for expert_idx in unique_experts:
            expert_output = self.experts[expert_idx](data)  # [BSZ, hidden_dim]
            expert_cache[expert_idx.item()] = expert_output
        
        # 5. Assembla gli output per ogni sample
        final_outputs = []
        for i in range(batch_size):
            sample_outputs = []
            sample_weights = top_k_weights[i]  # [k]
            
            for j in range(self.k):
                expert_idx = top_k_indices[i, j].item()
                expert_output = expert_cache[expert_idx][i]  # [hidden_dim]
                weighted_output = expert_output * sample_weights[j]
                sample_outputs.append(weighted_output)
            
            # Somma pesata degli output degli esperti selezionati
            final_output = torch.stack(sample_outputs).sum(dim=0)  # [hidden_dim]
            final_outputs.append(final_output)
        
        z = torch.stack(final_outputs)  # [BSZ, hidden_dim]
        
        # 6. Final classification
        y = self.classifier(z)
        
        # Store per la load balancing loss
        self.last_gate_scores = gate_scores
        self.last_top_k_indices = top_k_indices
        
        return y
    
    def compute_total_loss(self, pred, y):
        """Calcola la loss totale includendo load balancing"""
        # Loss principale
        main_loss = self.loss_fn(pred, y)
        
        # Load balancing loss
        if hasattr(self, 'last_gate_scores') and hasattr(self, 'last_top_k_indices'):
            load_loss = self.compute_load_balancing_loss(
                self.last_gate_scores, 
                self.last_top_k_indices
            )
            total_loss = main_loss + self.load_balancing_weight * load_loss
            
            # Log delle loss separate
            self.log('main_loss', main_loss, prog_bar=False)
            self.log('load_loss', load_loss, prog_bar=False)
        else:
            total_loss = main_loss
            
        return total_loss
    
    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % 10 == 0:
            loss = sum(self.loss) / len(self.loss)
            acc = sum(self.acc) / len(self.acc)
            self.h.append({
                'train_loss': loss,
                'train_accuracy': acc
            })
            
            # Log expert usage statistics
            self.log_expert_usage()
    
    def log_expert_usage(self):
        """Log statistiche sull'uso degli esperti"""
        if hasattr(self, 'last_top_k_indices'):
            expert_counts = torch.zeros(self.num_experts)
            for idx in self.last_top_k_indices.flatten():
                expert_counts[idx] += 1
            
            # Log l'esperto più e meno usato
            most_used = expert_counts.max().item()
            least_used = expert_counts.min().item()
            
            self.log('expert_usage_max', most_used, prog_bar=False)
            self.log('expert_usage_min', least_used, prog_bar=False)
            self.log('expert_usage_ratio', most_used / (least_used + 1e-8), prog_bar=False)
    
    def training_step(self, batch):
        y = batch.y
        pred = self.forward(batch)
        
        # Use total loss with load balancing
        loss = self.compute_total_loss(pred, y)
        acc = self.acc_fn(pred, y)
        
        self.log_dict({
            'train_loss': loss,
            'train_accuracy': acc
        }, prog_bar=True, on_epoch=True, on_step=False, batch_size=pred.shape[0])
        
        self.loss_epoch.append(loss.detach().cpu().item())
        self.acc_epoch.append(acc.cpu().item())
        self.loss.append(loss.detach().cpu().item())
        self.acc.append(acc.cpu().item())
        
        return loss
    
    def validation_step(self, batch):
        y = batch.y
        pred = self.forward(batch)
        
        # Use total loss with load balancing anche per validation
        loss = self.compute_total_loss(pred, y)
        acc = self.acc_fn(pred, y)
        
        self.log_dict({
            'val_loss': loss,
            'val_accuracy': acc
        }, prog_bar=True, on_epoch=True, on_step=False, batch_size=pred.shape[0])
        
        self.loss_epoch.append(loss.detach().cpu().item())
        self.acc_epoch.append(acc.cpu().item())
        self.loss.append(loss.detach().cpu().item())
        self.acc.append(acc.cpu().item())
        
        return loss
    
    def get_expert_usage_stats(self):
        """Ritorna statistiche sull'uso degli esperti"""
        if hasattr(self, 'last_top_k_indices'):
            expert_counts = torch.zeros(self.num_experts)
            for idx in self.last_top_k_indices.flatten():
                expert_counts[idx] += 1
            
            return {
                'expert_counts': expert_counts.tolist(),
                'most_used_expert': expert_counts.argmax().item(),
                'least_used_expert': expert_counts.argmin().item(),
                'usage_std': expert_counts.std().item()
            }
        return None