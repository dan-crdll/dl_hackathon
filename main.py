import torch 
import numpy as np  
import random
import argparse
from source.layers import Classifier, Encoder
from source.model import LitClassifier
from source.load_data import GraphDataset
from source.loss_fn import FocalLoss
from torch_geometric.data import DataLoader
from tqdm.auto import tqdm
import lightning as L
import os
import pandas as pd
import matplotlib.pyplot as plt


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(train_path=None, test_path=None, epochs=None):
    seed_everything()

    num_layers = 2
    hidden_dim = 128
    dropout = 0.1
    batch_size = 8
    if epochs is None:
        epochs = 30

    n = torch.zeros(6)

    encoder = Encoder(7, hidden_dim, num_layers)
    classifier = Classifier(hidden_dim, hidden_dim, 6, dropout)
    alpha = torch.ones(6)
    model = LitClassifier(encoder, classifier, alpha, split=None)

    if train_path is not None:
        train_ds = GraphDataset(train_path)
        dataloader = DataLoader(train_ds, shuffle=False, batch_size=1)

        for d in tqdm(dataloader):
            n[d.y] = n[d.y] + 1
        del dataloader 
        alpha = n / n.sum()
        alpha = 1.0 - alpha
        model.focal_loss = FocalLoss(alpha)
        train_dl = DataLoader(train_ds, shuffle=False, batch_size=batch_size)

        split = train_path.split("/")[-2]
        model.split = split


        trainer = L.Trainer(max_epochs=epochs, gradient_clip_val=1)
        trainer.fit(model, train_dataloaders=train_dl)

        history = model.h
        epochs = [i * 10 for i in range(len(history))]
        losses = [entry["loss"] for entry in history]
        accuracies = [entry["accuracy"] for entry in history]

        df = pd.DataFrame({
            "epoch": epochs,
            "loss": losses,
            "accuracy": accuracies
        })
        csv_path = f"logs/metric_{split}.csv"
        df.to_csv(csv_path, index=False)

        plt.figure()
        plt.plot(epochs, losses, label="Loss")
        plt.plot(epochs, accuracies, label="Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.title("Loss and Accuracy")
        plt.savefig(f"logs/metric_{split}.png")
        plt.close()

        del train_ds 
    
    if test_path is not None:
        test_ds = GraphDataset(test_path)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=1)

        split = test_path.split("/")[-2]
        model.split = split 
        if train_path is None:
            checkpoints = os.listdir('./checkpoints')

            max_epoch = -1
            best_ckpt = None
            for f in checkpoints:
                parts = f.split("_")
                if len(parts) < 3 or parts[-3] != split:
                    continue
                try:
                    epoch_num = int(parts[-1].split(".")[0])
                except Exception:
                    continue
                if epoch_num > max_epoch:
                    max_epoch = epoch_num
                    best_ckpt = f
            if best_ckpt is not None:
                print(f"Loading checkpoints {best_ckpt}")
                model.load_state_dict(torch.load(f'./checkpoints/{best_ckpt}'))
        model.eval()
        results = []
        with torch.no_grad():
            for data in tqdm(test_dl, total=len(test_dl)):
                pred = model(data)
                label = torch.argmax(pred, -1).cpu().item()
                results.append(label)

        submission_df = pd.DataFrame({
            "id": list(range(len(results))),
            "pred": results
        })
        submission_path = f"submission/testset_{split}.csv"
        submission_df.to_csv(submission_path, index=False)


        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-path', type=str, required=False, help='Path to the test data')
    parser.add_argument('--train-path', type=str, required=False, help='Path to the train data')
    parser.add_argument('--epochs', type=str, required=False, help='Training epochs')
    args = parser.parse_args()

    main(args.train_path, args.test_path, args.epochs)