import lightning as L 
from torchmetrics import Accuracy
from source.loss_fn import GCODLoss, InfoNCELoss
from source.augmentation import GraphAugmentation
import torch 
from torch import nn 


class LitClassifier(L.LightningModule):
    def __init__(self, split, encoder, similarity_module, embed_dim=256, hidden_dim=1024, num_classes=6):
        super().__init__()
        self.encoder = encoder 
        self.similarity_module = similarity_module

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes)
        )

        self.loss_fn_prev = GCODLoss(8)
        self.infonce = InfoNCELoss(0.05)

        self.augmenter = GraphAugmentation(0.2, 0.2)

        self.acc_fn = Accuracy('multiclass', num_classes=6)
        self.h = []
        self.loss_epoch = []
        self.acc_epoch = []
        self.split = split

        self.loss = []
        self.acc = []

        self.h = []
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)
    
    def forward(self, data, weight_soft=1):
        z1 = self.encoder(data)
        with torch.no_grad():
            data_aug = self.augmenter.augment(data)
            z2 = self.encoder(data_aug)
        
        contrastive_loss = self.infonce(z1, z2)

        soft_labels, confidence = self.similarity_module(z1, return_confidence=True)

        y = self.classifier(z1)
        y = y + weight_soft * soft_labels

        return y, (z1, contrastive_loss, confidence)
    
    def on_train_epoch_end(self):
        if (self.current_epoch + 1)  % 10 == 0:
            loss = sum(self.loss) / len(self.loss)
            acc = sum(self.acc) / len(self.acc)
            self.h.append({
                'loss': loss,
                'accuracy': acc
            })
    
    def weight_soft(self, epoch):
        if epoch > 15:
            return 1.0
        else:
            return 1 / epoch
    
    def training_step(self, batch):
        # batch = self.undersample_class2(batch)
        y = batch.y 
        
        pred, (z, contrastive_loss, confidence) = self.forward(batch, self.weight_soft(self.current_epoch))

        self.similarity_module.update_prototypes(z, y, confidence)

        loss = self.loss_fn_prev(pred, y) + contrastive_loss
        acc = self.acc_fn(pred, y)

        self.log_dict({
            'loss': loss,
            'accuracy': acc 
        }, prog_bar=True, on_epoch=True, on_step=False, batch_size=pred.shape[0])

        self.loss_epoch.append(loss.detach().cpu().item())
        self.acc_epoch.append(acc.cpu().item())

        self.loss.append(loss.detach().cpu().item())
        self.acc.append(acc.cpu().item())

        return loss 

    