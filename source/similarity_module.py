import torch
from torch import nn
from torch.nn import functional as F

class RobustSimilarityModule(nn.Module):
    def __init__(self, hidden_dim, num_classes=6, momentum=0.9, temp=0.1, confidence_threshold=0.8):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.temp = temp
        self.confidence_threshold = confidence_threshold
        
        # Running statistics
        self.register_buffer('means', torch.randn(num_classes, hidden_dim))
        self.register_buffer('vars', torch.ones(num_classes, hidden_dim))
        self.register_buffer('samples', torch.zeros(num_classes))
        self.register_buffer('confidence_scores', torch.zeros(num_classes))
    
    def forward(self, z, return_confidence=False):
        # Normalize features
        z_norm = F.normalize(z, p=2, dim=1)
        means_norm = F.normalize(self.means, p=2, dim=1)
        
        # Compute similarities with temperature scaling
        similarities = torch.matmul(z_norm, means_norm.t()) / self.temp
        
        # Convert to probabilities
        soft_labels = F.softmax(similarities, dim=1)
        
        if return_confidence:
            # Compute confidence as negative entropy
            entropy = -(soft_labels * torch.log(soft_labels + 1e-8)).sum(dim=1)
            confidence = 1 - entropy / torch.log(torch.tensor(self.num_classes))
            return soft_labels, confidence
        
        return soft_labels
    
    def update_prototypes(self, z, labels, confidence=None):
        """Update prototypes with confidence weighting and momentum"""
        # If confidence not provided, use prediction confidence
        if confidence is None:
            with torch.no_grad():
                soft_labels, confidence = self.forward(z, return_confidence=True)
                labels = torch.argmax(soft_labels, dim=-1)
        
        # Only update with high-confidence samples
        mask = confidence > self.confidence_threshold
        if not mask.any():
            return
        
        z_filtered = z[mask]
        labels_filtered = labels[mask]
        confidence_filtered = confidence[mask]
        
        # Update statistics for each class
        for c in range(self.num_classes):
            class_mask = labels_filtered == c
            if not class_mask.any():
                continue
            
            # Get class samples and their confidences
            class_samples = z_filtered[class_mask]
            class_confidences = confidence_filtered[class_mask]
            
            # Weighted mean
            weights = class_confidences / class_confidences.sum()
            new_mean = (class_samples * weights.unsqueeze(1)).sum(dim=0)
            
            # Update with momentum - FIXED: Use .data to avoid in-place operations on computational graph
            if self.samples.data[c] > 0:
                self.means.data[c] = self.momentum * self.means.data[c] + (1 - self.momentum) * new_mean
            else:
                self.means.data[c] = new_mean
            
            # Update sample count - FIXED: Use .data
            self.samples.data[c] += class_mask.sum()
            
            # Track average confidence per class - FIXED: Use .data
            self.confidence_scores.data[c] = (
                self.confidence_scores.data[c] * 0.9 +
                class_confidences.mean() * 0.1
            )