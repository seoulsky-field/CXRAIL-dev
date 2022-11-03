from multiprocessing import cpu_count
from functools import partial

from ray import tune
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from ray import air

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import timm

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from data_loader import dataset_CheXpert # Load our custom loader
from data_loader.dataset_CheXpert import *
#from cfg import cfg (outdated as we use hydra)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(dataloader, val_loader, model, loss_f, optimizer, cfg):
    size = len(dataloader.dataset)
    best_val_roc_auc = 0
    
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        data_X, label_y = X.to(device), y.to(device)
        pred = model(data_X)
        pred = torch.softmax(pred, dim=1) # for multi-label
        loss = loss_f(pred, label_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(data_X)
            val_loss, val_roc_auc = val(val_loader, model, loss_f)
            if best_val_roc_auc < val_roc_auc:
                best_val_roc_auc = val_roc_auc
                torch.save(model.state_dict(), cfg.ckpt_name)
                print("Best model saved.")
            print(f"Batch ID: {batch}, loss: {loss:>7f}, val_loss = {val_loss:>7f}, val_roc_auc: {val_roc_auc:>4f}, Best_val_score: {best_val_roc_auc:>4f}, [{current:>5d}/{size:>5d}]")
            
        tune.report(loss=loss, val_loss=val_loss)
        print('[Train loop break for fast process test]')
        break

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


def trainval(config, hydra_cfg):  
    print(f'Check parameter space : {config}')
    print(f'check hydra config : {hydra_cfg}')

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
                            batch_size=config['batch_size'],
                            num_workers=hydra_cfg.num_workers,
                            shuffle=True,
                            drop_last=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=config['batch_size'],
                            num_workers=hydra_cfg.num_workers,
                            shuffle=False,
                            drop_last=False)
    
    model = timm.create_model(
        model_name=hydra_cfg.model_name, 
        pretrained=hydra_cfg.pretrained, 
        num_classes=hydra_cfg.num_classes
        ).to(device)
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=hydra_cfg.weight_decay
        )

    print (device)
    for t in range(hydra_cfg.epochs):
        print(f"Epoch {t+1}")
        train(train_loader, val_loader, model, loss_f, optimizer, hydra_cfg)
        #test(val_dataloader, model, loss_f)
        print("---------------------------------")
    print("Done!")

# ========== The main function =========
@hydra.main(
    version_base = None, 
    config_path='../hydra_ref/config', 
    config_name = 'basic_config'
    )
def main(cfg: DictConfig):
    param_space = {
        'lr': tune.loguniform(0.0001, 0.1),
        'batch_size': tune.choice([32, 64]),
    }

    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
    )

    reporter = tune.CLIReporter(
        metric_columns=['loss', 'val_loss']
    )
    from functools import partial
    result = tune.run(
        partial(trainval, hydra_cfg=cfg),
        config=param_space,
        num_samples=cfg.tuning_iteration,
        scheduler=scheduler,
        progress_reporter=reporter,
        resources_per_trial={"cpu": int(round(cpu_count()/2)), "gpu": 1},
    )

if __name__=="__main__":
    main()