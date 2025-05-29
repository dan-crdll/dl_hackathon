import torch 
from torch import nn 
import torch.nn.functional as F 
from torchmetrics import Accuracy

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
    
class GCODLoss(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(batch_size), requires_grad=True)
        self.acc = Accuracy('multiclass', num_classes=6)
        self.epsilon = 0.1
        self.eta = 1e-3  # step size for manual update

    def forward(self, pred, target):
        target_one_hot = F.one_hot(target, 6).to(target.device).squeeze().float()
        soft_target = (1 - self.epsilon) * target_one_hot + self.epsilon / 6
        acc = self.acc(pred, target.squeeze())

        pred_label = torch.argmax(pred, -1).to(pred.device)
        pred_one_hot = F.one_hot(pred_label, 6).to(pred.device).squeeze().float()

        U = torch.diag(self.u)

        # l1 term (uses U.detach to avoid influencing u)
        l1 = F.cross_entropy(pred + acc * U.detach() @ target_one_hot, soft_target)

        # l2 term (needs grad w.r.t. u)
        l2 = 1 / 6 * torch.norm(pred_one_hot.detach() + U @ target_one_hot - target_one_hot) ** 2

        # l3 term (KL divergence)
        L = torch.log(F.sigmoid(torch.diagonal(pred @ target_one_hot.T)))
        L = F.softmax(L, -1)
        Q = F.softmax(-torch.log(self.u.detach() + 1e-8), -1)
        D = (L * torch.log(L / Q)).sum(-1)
        l3 = (1 - acc) * D

        return l1 + l3 + l2
