import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from libauc.metrics import auc_roc_score
from torchmetrics.classification import MultilabelAUROC

# hydra
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

## ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.air.config import ScalingConfig

# 내부 모듈
from custom_utils.custom_metrics import *
from custom_utils.custom_reporter import *
from custom_utils.transform import create_transforms
from data_loader.dataset_CheXpert import *
from custom_utils.print_tree import print_config_tree


#log = logging.getLogger(__name__)

best_val_roc_auc = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(hydra_cfg, dataloader, val_loader, model, loss_f, optimizer, epoch, best_val_roc_auc):
    config = hydra_cfg.mode


    size = len(dataloader.dataset)
    #global best_val_roc_auc
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


            if best_val_roc_auc < val_roc_auc:
                best_val_roc_auc = val_roc_auc
                if config.execute_mode == 'default':
                    torch.save(model.state_dict(), HydraConfig.get().run.dir + '/' + hydra_cfg.ckpt_name)
                    print("Best model saved.")
                elif config.execute_mode == 'raytune':
                    torch.save(model.state_dict(), hydra_cfg.ckpt_name)
                    print("Best model saved.")

            report_metrics(val_pred, val_true, print_classification_result=False)
            print(f"loss: {loss:>7f}, val_loss = {val_loss:>7f}, val_roc_auc: {val_roc_auc:>4f}, Best_val_score: {best_val_roc_auc:>4f}, epoch: {epoch+1}, Batch ID: {batch}[{current:>5d}/{size:>5d}]")

                
            if config.execute_mode == 'raytune':
                result_metrics = {
                            'epoch' : epoch+1, 
                            'Batch_ID': batch,
                            'loss' : loss, 
                            'val_loss' : val_loss, 
                            'val_score' : val_roc_auc, 
                            'best_val_score' : best_val_roc_auc, 
                            'progress_of_epoch' : f"{100*current/size:.1f} %"}

                # tune.report -> session.report (https://docs.ray.io/en/latest/_modules/ray/air/session.html#report)
                session.report(metrics = result_metrics)
        
        model.train()
    return best_val_roc_auc
        


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
        val_true_tensor = torch.from_numpy(val_true)
        val_pred_tensor = torch.from_numpy(val_pred)
        auroc = MultilabelAUROC(num_labels=5, average="macro", thresholds=None)
        auc_roc_scores = auroc(val_pred_tensor, val_true_tensor)
        val_roc_auc = torch.mean(auc_roc_scores).numpy()
        val_loss /= num_batches
    return val_loss, val_roc_auc, val_pred, val_true        


def trainval(config, hydra_cfg, best_val_roc_auc = 0):
    
    if hydra_cfg.mode.execute_mode == 'default':
        cfg = config
    elif hydra_cfg.mode.execute_mode == 'raytune':
        cfg = {'batch_size':None, 'rotate_degree':None, 'lr':None, 'weight_decay':None}
        for key in cfg.keys():
            try:
                cfg[key] = config[key]
            except:
                cfg[key] = hydra_cfg[key]

    train_dataset = CXRDataset('train', **hydra_cfg.Dataset, transforms=create_transforms(hydra_cfg, 'train', cfg['rotate_degree']))
    val_dataset = CXRDataset('valid', **hydra_cfg.Dataset, transforms=create_transforms(hydra_cfg, 'valid', cfg['rotate_degree']))

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], **hydra_cfg.Dataloader.train)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'],  **hydra_cfg.Dataloader.test)

    model = instantiate(hydra_cfg.model)
    model = model.to(device)
    loss_f = instantiate(hydra_cfg.loss)

    if hydra_cfg.optimizer._target_.startswith('torch'):
        optimizer = instantiate(
            hydra_cfg.optimizer, 
            params = model.parameters(), 
            lr =  cfg['lr'],
            weight_decay = cfg['weight_decay'] ,
            )
    else:
        optimizer = instantiate(
            hydra_cfg.optimizer, 
            model = model, 
            loss_fn = loss_f, 
            lr = cfg['lr'],
            weight_decay = cfg['weight_decay'],
            ) 

    for epoch in range(hydra_cfg.epochs):
        best_val_roc_auc = train(hydra_cfg, train_loader, val_loader, model, loss_f, optimizer, epoch, best_val_roc_auc)


# @hydra.main(
#     version_base = None, 
#     config_path='config', 
#     config_name = 'config'
# )
# def main(hydra_cfg: DictConfig):
#     config = hydra_cfg.mode
#     assert (config.execute_mode == 'default'), "change hydra mode into default. Ray should be executed in main.py"
#     if hydra_cfg.get("print_config"):
#         #log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
#         print_config_tree(hydra_cfg, resolve=True, save_to_file=True)
    
#     trainval(config, hydra_cfg)
    
# if __name__ == "__main__":
