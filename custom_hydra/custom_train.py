
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
#from ray import air, tune
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
######## 주영 수정 #######
# from sklearn.metrics import roc_auc_score
from libauc.metrics import auc_roc_score
#########################
from data_loader import dataset_CheXpert # Load our custom loader
from data_loader.dataset_CheXpert import *

best_val_roc_auc = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train(dataloader, val_loader, model, loss_f, optimizer, cfg):
    size = len(dataloader.dataset)
    global best_val_roc_auc
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        data_X, label_y = X.to(device), y.to(device)
        pred = model(data_X)
        ######### 주영 수정 ########
        pred = torch.sigmoid(pred) # for multi-label
        ###########################
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
        #tune.report(loss=val_loss, accuracy=val_roc_auc)
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
            ########### 주영 수정 #########
            pred = torch.sigmoid(pred)
            ##############################
            val_loss += loss_f(pred, y).item()
            val_pred.append(pred.cpu().detach().numpy())
            val_true.append(y.cpu().numpy())
    val_true = np.concatenate(val_true)
    val_pred = np.concatenate(val_pred)
    ########## 주영 수정 #########
    auc_roc_scores = auc_roc_score(val_true, val_pred)
    val_roc_auc = np.mean(auc_roc_scores)
    ##############################
    val_loss /= num_batches
    #print(f"Batch ID: {idx}, Accuracy: {(100*correct):>0.1f}, loss: {loss:>7f}, [{current:>5d}/{size:>5d}], val_roc_auc: {val_roc_auc:>4f}, Best_val_score: {best_val_roc_auc:>4f}, [{current:>5d}/{size:>5d}]")
    return val_loss, val_roc_auc
@hydra.main(
    version_base = None,
    config_path='config',
    config_name = 'config'
    )
def trainval(cfg: DictConfig):
    ########## 1103 추가 - configuration 출력 ###########
    print(OmegaConf.to_yaml(cfg))
    ####################################################
    train_dataset = dataset_CheXpert.ChexpertDataset('train', **cfg.Dataset)
    val_dataset = dataset_CheXpert.ChexpertDataset('valid', **cfg.Dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, **cfg.Dataloader.train)
    val_loader = torch.utils.data.DataLoader(val_dataset, **cfg.Dataloader.test)
    model = instantiate(cfg.model)
    model = model.to(device)
    loss_f = instantiate(cfg.loss)
    ######## 예나 수정 (optimizer hydra로) #######
    #optimizer = instantiate(cfg.optimizer, params=model.parameters())   ##When using optimizer/loss from torch.utils
    optimizer = instantiate(cfg.optimizer, model=model, loss_fn=loss_f)  ##When using optimizer/loss from libauc
    #############################################
   
    print (device)
    for t in range(cfg.epochs):
        print(f"Epoch {t+1}")
        train(train_loader, val_loader, model, loss_f, optimizer, cfg)
        #test(val_dataloader, model, loss_f)
        print("---------------------------------")
    # with tune.checkpoint_dir(cfg.epoch) as checkpoint_dir:
    #     path = os.path.join(checkpoint_dir, "checkpoint")
    #     torch.save((model.state_dict(), optimizer.state_dict()), path)
    print("Done!")
if __name__ == "__main__":
    trainval()
