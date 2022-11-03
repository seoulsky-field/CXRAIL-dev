# hydara
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from hydra.utils import instantiate

# ray
from ray import air, tune
from ray.air.checkpoint import Checkpoint
from ray.air import session

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from data_loader import dataset_CheXpert # Load our custom loader
#from data_loader.dataset_CheXpert import *


best_val_roc_auc = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(dataloader, val_loader, model, loss_f, optimizer, cfg):
    size = len(dataloader.dataset)
    global best_val_roc_auc
    
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        data_X, label_y = X.to(device), y.to(device)
        pred = model(data_X)
        pred = torch.softmax(pred, dim=1) # for multi-label
        loss = loss_f(pred, label_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 500 == 0:
            loss, current = loss.item(), batch * len(data_X)
            val_loss, val_roc_auc = val(val_loader, model, loss_f)
            
            tune.report(loss =val_loss,  accuracy = val_roc_auc)

            if best_val_roc_auc < val_roc_auc:
                best_val_roc_auc = val_roc_auc
                torch.save(model.state_dict(), cfg.ckpt_name)
                print("Best model saved.")
            print(f"Batch ID: {batch}, loss: {loss:>4f}, val_loss = {val_loss:>4f}, val_roc_auc: {val_roc_auc:>4f}, Best_val_score: {best_val_roc_auc:>4f}, [{current:>5d}/{size:>5d}]")
        
            

def val(dataloader, model, loss_f):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss = 0
    model.eval()
    with torch.no_grad():
        val_pred = []
        val_true = []
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = torch.softmax(pred, dim=1) # for multi-label
            val_loss += loss_f(pred, y).item()
            val_pred.append(pred.cpu().detach().numpy())
            val_true.append(y.cpu().numpy())
            
    val_true = np.concatenate(val_true)
    val_pred = np.concatenate(val_pred)
    val_roc_auc = roc_auc_score(val_true, val_pred, average='macro', multi_class='ovr')
    
    val_loss /= num_batches
    return val_loss, val_roc_auc



def trainval(config, hydra_cfg):
    train_dataset = dataset_CheXpert.ChexpertDataset('train', **hydra_cfg.Dataset)
    val_dataset = dataset_CheXpert.ChexpertDataset('valid', **hydra_cfg.Dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, **hydra_cfg.Dataloader.train)
    val_loader = torch.utils.data.DataLoader(val_dataset, **hydra_cfg.Dataloader.test)
    
    model = instantiate(hydra_cfg.model)
    model = model.to(device)
    loss_f = instantiate(hydra_cfg.loss)
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-8)


    print (device)
#    for t in range(hydra_cfg.epochs):
    for t in range(1):
        print(f"Epoch {t+1}")
        train(train_loader, val_loader, model, loss_f, optimizer, hydra_cfg)
        #test(val_dataloader, model, loss_f)
        print("---------------------------------")

    print("Done!")

if __name__ == "__main__":
    trainval()
