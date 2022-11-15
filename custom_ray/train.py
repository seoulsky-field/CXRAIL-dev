import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from libauc.metrics import auc_roc_score

# hydra
from hydra.utils import instantiate
## ray
from ray import tune

# 내부 모듈
from custom_utils.custom_metrics import *
from custom_utils.custom_reporter import *
from custom_utils.transform import *
from custom_utils.utils import param_override
from data_loader.dataset_CheXpert import *

best_val_roc_auc = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(dataloader, val_loader, model, loss_f, optimizer, cfg, epoch):
    size = len(dataloader.dataset)
    global best_val_roc_auc
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        data_X, label_y = X.to(device), y.to(device)

        pred = model(data_X)
        pred = torch.sigmoid(pred) # for multi-label
        loss = loss_f(pred, label_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 500 == 0:
            loss, current = loss.item(), batch * len(data_X)
            val_loss, val_roc_auc, val_pred, val_true = val(val_loader, model, loss_f)

            tune.report(loss=loss, val_loss=val_loss, val_score=val_roc_auc, best_val_roc_auc=best_val_roc_auc, current_epoch=epoch+1, progress_of_epoch=f"{100*current/size:.1f} %")

            if best_val_roc_auc < val_roc_auc:
                best_val_roc_auc = val_roc_auc
                tune.report(loss=loss, val_loss=val_loss, val_score=val_roc_auc, best_val_roc_auc=best_val_roc_auc, current_epoch=epoch+1, progress_of_epoch=f"{100*current/size:.1f} %")
                torch.save(model.state_dict(), cfg.ckpt_name)
                # print("Best model saved.")
                report_metrics(val_pred, val_true, print_classification_result=False)
            # print(f"Batch ID: {batch}, loss: {loss:>7f}, val_loss = {val_loss:>7f}, val_roc_auc: {val_roc_auc:>4f}, Best_val_score: {best_val_roc_auc:>4f}, [{current:>5d}/{size:>5d}]")
        
        model.train()


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
            pred = torch.sigmoid(pred)
            val_loss += loss_f(pred, y).item()
            val_pred.append(pred.cpu().detach().numpy())
            val_true.append(y.cpu().numpy())
        val_true = np.concatenate(val_true)
        val_pred = np.concatenate(val_pred)
        auc_roc_scores = auc_roc_score(val_true, val_pred)
        val_roc_auc = np.mean(auc_roc_scores)
        val_loss /= num_batches
    return val_loss, val_roc_auc, val_pred, val_true        


def trainval(config, hydra_cfg):
    train_dataset = ChexpertDataset('train', **hydra_cfg.Dataset, transforms=create_train_transforms(hydra_cfg))
    val_dataset = ChexpertDataset('valid', **hydra_cfg.Dataset, transforms=create_val_transforms(hydra_cfg))

    train_loader = DataLoader(train_dataset, **hydra_cfg.Dataloader.train)
    val_loader = DataLoader(val_dataset, **hydra_cfg.Dataloader.test)

    model = instantiate(hydra_cfg.model)
    model = model.to(device)
    loss_f = instantiate(hydra_cfg.loss)

    if hydra_cfg.optimizer._target_.startswith('torch'):
        optimizer = instantiate(
            hydra_cfg.optimizer, 
            params=model.parameters(), 
            lr=param_override(hydra_cfg.optimizer['lr'], config['lr']),
            weight_decay=param_override(hydra_cfg.optimizer['weight_decay'], config['weight_decay']),
            )
    else:
        optimizer = instantiate(
            hydra_cfg.optimizer, 
            model=model, 
            loss_fn=loss_f, 
            lr=param_override(hydra_cfg.optimizer['lr'], config['lr']),
            weight_decay=param_override(hydra_cfg.optimizer['weight_decay'], config['weight_decay']),
            ) 

    # print (device)
    for epoch in range(hydra_cfg.epochs):
        # print(f"Epoch {epoch+1}")
        train(train_loader, val_loader, model, loss_f, optimizer, hydra_cfg, epoch)
       
    #     print("---------------------------------")
    # print("Done!")

    
if __name__ == "__main__":
    trainval()
