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


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(train_path, test_path=None):
    seed_everything()

    num_layers = 10
    hidden_dim = 1024
    embed_dim = 256
    dropout = 0.1
    batch_size = 8
    epochs = 50

    n = torch.zeros(6)

    train_ds = GraphDataset(train_path)
    dataloader = DataLoader(train_ds, shuffle=False, batch_size=1)

    for d in tqdm(dataloader):
        n[d.y] = n[d.y] + 1
    del dataloader 
    alpha = n / n.sum()
    alpha = 1.0 - alpha


    encoder = Encoder(7, hidden_dim, num_layers)
    classifier = Classifier(embed_dim, hidden_dim, 6, dropout)

    train_dl = DataLoader(train_ds, shuffle=False, batch_size=batch_size)

    split = train_path.split("/")[-1]

    model = LitClassifier(encoder, classifier, alpha, split=split)

    trainer = L.Trainer(max_epochs=epochs, gradient_clip_val=1)
    trainer.fit(model, train_dataloaders=train_dl)

    # TODO: Save model history logs and loss/accuracy plot in logs folder
    # TODO: Test part with dataset loading and result printing in submission

        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-path', type=str, required=False, help='Path to the test data')
    parser.add_argument('--train-path', type=str, required=True, help='Path to the train data')
    args = parser.parse_args()

    main(args.train_path, args.test_path)