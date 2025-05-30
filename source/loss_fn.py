import torch 
from torch import nn 
import torch.nn.functional as F 
from torchmetrics import Accuracy

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


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1: Tensor of shape (bsz, hidden_dim)
        z2: Tensor of shape (bsz, hidden_dim)
        """
        bsz = z1.size(0)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        logits = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(bsz, device=z1.device)
        loss = F.cross_entropy(logits, labels)
        return loss