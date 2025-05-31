import lightning as L 
from torchmetrics import Accuracy
from source.loss_fn import SCELoss, FocalLoss, GCODLoss
import torch 


class LitClassifier(L.LightningModule):
    def __init__(self, encoder, alpha, split):
        super().__init__()
        self.encoder = encoder 
        # self.classifier = classifier

        self.loss_fn = GCODLoss(32)
        # self.focal_loss = FocalLoss(alpha)
        self.acc_fn = Accuracy('multiclass', num_classes=6)
        self.undersample = torch.argmin(alpha).int().item()
        self.h = []
        self.val_h = []
        self.val_loss = []
        self.val_acc = []
        self.split = split

        self.loss = []
        self.acc = []

        self.h = []
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)
    
    def forward(self, data):
        # z = self.encoder(data)
        # y = self.classifier(z)
        y = self.encoder(data)
        return y 
    
    def on_train_epoch_end(self):
        if (self.current_epoch + 1)  % 10 == 0:
            loss = sum(self.loss) / len(self.loss)
            acc = sum(self.acc) / len(self.acc)
            self.h.append({
                'loss': loss,
                'accuracy': acc
            })

        self.loss.clear()
        self.acc.clear()
    
    def training_step(self, batch):
        # batch = self.undersample_class2(batch)
        y = batch.y 
        
        pred = self.forward(batch)

        loss = self.loss_fn(pred, y)
        acc = self.acc_fn(pred, y)

        self.log_dict({
            'loss': loss,
            'accuracy': acc 
        }, prog_bar=True, on_epoch=True, on_step=False, batch_size=pred.shape[0])


        self.loss.append(loss.detach().cpu().item())
        self.acc.append(acc.cpu().item())

        return loss 

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1)  % 10 == 0:
            loss = sum(self.val_loss) / len(self.val_loss)
            acc = sum(self.val_acc) / len(self.val_acc)
            self.val_h.append({
                'loss': loss,
                'accuracy': acc
            })
        self.val_loss.clear()
        self.val_acc.clear()

    
    def validation_step(self, batch):
        # batch = self.undersample_class2(batch)
        y = batch.y 
        
        pred = self.forward(batch)

        loss = self.loss_fn(pred, y)
        acc = self.acc_fn(pred, y)

        self.log_dict({
            'val_loss': loss,
            'val_accuracy': acc 
        }, prog_bar=True, on_epoch=True, on_step=False, batch_size=pred.shape[0])

        self.val_loss.append(loss.detach().cpu().item())
        self.val_acc.append(acc.cpu().item())

        return loss 
