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

# ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.air.config import ScalingConfig

# 내부 모듈
from custom_utils.custom_metrics import report_metrics
from custom_utils.custom_reporter import *
from custom_utils.transform import create_transforms
from data_loader.data_loader import CXRDataset
from custom_utils.print_tree import print_config_tree


# log = logging.getLogger(__name__)

best_val_roc_auc = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def c_train(
    hydra_cfg, dataloader, val_loader, model, loss_f, optimizer, epoch, best_val_roc_auc
):
    config = hydra_cfg
    # global best_model_state
    size = len(dataloader.dataset)
    # global best_val_roc_auc
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        data_X, label_y = X.to(device), y.to(device)

        pred = model(data_X)
        # pred = torch.sigmoid(pred) # for multi-label
        loss = loss_f(pred, label_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 500 == 0:
            loss, current = loss.item(), batch * len(data_X)
            val_loss, val_roc_auc, val_pred, val_true = c_val(val_loader, model, loss_f)

            if best_val_roc_auc < val_roc_auc:
                best_val_roc_auc = val_roc_auc
                best_model_state = model.state_dict()

                # if config == 'default':
                #     torch.save(model.state_dict(), HydraConfig.get().run.dir + '/' + hydra_cfg.ckpt_name)
                #     print("Best model saved.")
                # elif config == 'raytune':
                #     torch.save(model.state_dict(), hydra_cfg.ckpt_name)
                #     print("Best model saved.")

            report_metrics(val_pred, val_true, print_classification_result=False)
            print(
                f"loss: {loss:>7f}, "
                f"val_loss = {val_loss:>7f}, "
                f"val_roc_auc: {val_roc_auc:>4f}, "
                f"Best_val_score: {best_val_roc_auc:>4f}, "
                f"epoch: {epoch+1}, "
                f"Batch ID: {batch}[{current:>5d}/{size:>5d}]"
            )

            # if config.execute_mode == 'raytune':
            #     result_metrics = {
            #                 'epoch' : epoch+1,
            #                 'Batch_ID': batch,
            #                 'loss' : loss,
            #                 'val_loss' : val_loss,
            #                 'val_score' : val_roc_auc,
            #                 'best_val_score' : best_val_roc_auc,
            #                 'progress_of_epoch' : f"{100*current/size:.1f} %"}

            #     # tune.report -> session.report (https://docs.ray.io/en/latest/_modules/ray/air/session.html#report)
            #     session.report(metrics = result_metrics)

        model.train()
    return best_val_roc_auc, best_model_state


def c_val(dataloader, model, loss_f):
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
            # pred = torch.sigmoid(pred)
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


def c_trainval(hydra_cfg, best_val_roc_auc=0):

    cfg = hydra_cfg.conditional_train

    # conditional training
    condition_train_dataset = CXRDataset(
        "train",
        **hydra_cfg.Dataset,
        transforms=create_transforms(hydra_cfg, "train", cfg["rotate_degree"]),
        conditional_train=True,
    )
    condition_train_loader = DataLoader(
        condition_train_dataset, **hydra_cfg.Dataloader.conditional_train
    )
    val_dataset = CXRDataset(
        "valid",
        **hydra_cfg.Dataset,
        transforms=create_transforms(hydra_cfg, "valid", cfg["rotate_degree"]),
        conditional_train=False,
    )
    val_loader = DataLoader(val_dataset, **hydra_cfg.Dataloader.test)
    model_ct = instantiate(cfg.model)
    model_ct = model_ct.to(device)
    loss_f_ct = instantiate(cfg.loss)
    if cfg.optimizer._target_.startswith("torch"):
        optimizer_ct = instantiate(
            cfg.optimizer,
            params=model_ct.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
    else:
        optimizer_ct = instantiate(
            cfg.optimizer,
            model=model_ct,
            loss_fn=loss_f_ct,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )

    print("#############################################")
    print("########## Conditional-Train Start ##########")
    print("#############################################")
    for epoch in range(cfg.epochs):
        best_val_roc_auc, best_model_state = c_train(
            hydra_cfg.conditional_train,
            condition_train_loader,
            val_loader,
            model_ct,
            loss_f_ct,
            optimizer_ct,
            epoch,
            best_val_roc_auc,
        )
    print("#############################################")
    print("########### Conditional-Train End ###########")
    print("#############################################")

    return best_model_state
