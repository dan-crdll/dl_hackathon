import torch 
from torch import nn 
import torch.nn.functional as F 

class SCELoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=10):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, target):
        # Cross-Entropy standard
        ce = F.cross_entropy(logits, target)

        # Reverse Cross-Entropy
        probs = F.softmax(logits, dim=1)
        # One hot labels
        with torch.no_grad():
            target_onehot = torch.zeros_like(probs).scatter_(1, target.view(-1,1), 1)
        rce = -torch.sum(probs * torch.log(target_onehot + 1e-7), dim=1).mean()

        loss = self.alpha * ce + self.beta * rce
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = nn.Parameter(alpha.float(), requires_grad=False)
        self.gamma = gamma
        self.reduction = reduction 

    def forward(self, logits, target):
        logpt = F.log_softmax(logits, dim=1)
        pt = torch.exp(logpt)
        at = self.alpha.gather(0, target)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -at * (1 - pt) ** self.gamma * logpt 

        return loss.mean()