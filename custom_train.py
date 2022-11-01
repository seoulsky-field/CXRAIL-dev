import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

import timm

from data_loader import dataset_CheXpert # Load our custom loader
from data_loader.dataset_CheXpert import *
#from cfg import cfg (outdated as we use hydra)


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
            if best_val_roc_auc < val_roc_auc:
                best_val_roc_auc = val_roc_auc
                torch.save(model.state_dict(), cfg.ckpt_name)
                print("Best model saved.")
            print(f"Batch ID: {batch}, loss: {loss:>7f}, val_loss = {val_loss:>7f}, val_roc_auc: {val_roc_auc:>4f}, Best_val_score: {best_val_roc_auc:>4f}, [{current:>5d}/{size:>5d}]")
            
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
    #print(f"Batch ID: {idx}, Accuracy: {(100*correct):>0.1f}, loss: {loss:>7f}, [{current:>5d}/{size:>5d}], val_roc_auc: {val_roc_auc:>4f}, Best_val_score: {best_val_roc_auc:>4f}, [{current:>5d}/{size:>5d}]")
    return val_loss, val_roc_auc


@hydra.main(
    version_base = None, 
    config_path='./hydra_ref/config', 
    config_name = 'basic_config'
    )
def trainval(cfg: DictConfig):  
    chexpert_cfg = dataset_CheXpert.Config() # default chexpert sample configuration
    train_dataset = dataset_CheXpert.ChexpertDataset(
        chexpert_cfg.root_path, 
        chexpert_cfg.small_dir, 
        chexpert_cfg.mode, 
        chexpert_cfg.use_frontal, 
        chexpert_cfg.train_cols,
        chexpert_cfg.use_enhancement, 
        chexpert_cfg.enhance_cols, 
        chexpert_cfg.enhance_time, 
        chexpert_cfg.flip_label, 
        chexpert_cfg.shuffle,
        chexpert_cfg.seed, 
        chexpert_cfg.image_size, 
        chexpert_cfg.verbose
        )
    val_dataset = dataset_CheXpert.ChexpertDataset(
        chexpert_cfg.root_path, 
        chexpert_cfg.small_dir, 
        'valid', 
        chexpert_cfg.use_frontal, 
        chexpert_cfg.train_cols,
        chexpert_cfg.use_enhancement, 
        chexpert_cfg.enhance_cols, 
        chexpert_cfg.enhance_time, 
        chexpert_cfg.flip_label, 
        chexpert_cfg.shuffle,
        chexpert_cfg.seed, 
        chexpert_cfg.image_size, 
        chexpert_cfg.verbose
        )

    train_loader = DataLoader(train_dataset,
                          batch_size=cfg.batch_size,
                          num_workers=cfg.num_workers,
                          shuffle=True,
                          drop_last=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers,
                            shuffle=False,
                            drop_last=False)
    
    model = timm.create_model(model_name=cfg.model_name, pretrained=cfg.pretrained, num_classes=cfg.num_classes).to(device)
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, 
                        betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg.weight_decay)

    print (device)
    for t in range(cfg.epochs):
        print(f"Epoch {t+1}")
        train(train_loader, val_loader, model, loss_f, optimizer,cfg)
        #test(val_dataloader, model, loss_f)
        print("---------------------------------")
    print("Done!")

if __name__ == "__main__":
    trainval()
