import torch 
import numpy as np  
import random
import argparse
from source.layers import Classifier, Encoder
from source.model import LitClassifier
from source.load_data import GraphDataset
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


def main(train_path=None, test_path=None):
    seed_everything()

    num_layers = 2
    hidden_dim = 1024
    embed_dim = 256
    dropout = 0.1
    batch_size = 8
    epochs = 50

    n = torch.zeros(6)

    encoder = Encoder(7, hidden_dim, num_layers)
    classifier = Classifier(embed_dim, hidden_dim, 6, dropout)
    model = LitClassifier(encoder, classifier, alpha, split=split)

    if train_path is not None:
        train_ds = GraphDataset(train_path)
        dataloader = DataLoader(train_ds, shuffle=False, batch_size=1)

        for d in tqdm(dataloader):
            n[d.y] = n[d.y] + 1
        del dataloader 
        alpha = n / n.sum()
        alpha = 1.0 - alpha
        train_dl = DataLoader(train_ds, shuffle=False, batch_size=batch_size)

        split = train_path.split("/")[-1]


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

        del dataset 
    
    if test_path is not None:
        test_ds = GraphDataset(test_path)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=1)

        split = test_path.split("/")[-1]

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
    args = parser.parse_args()

    main(args.train_path, args.test_path)