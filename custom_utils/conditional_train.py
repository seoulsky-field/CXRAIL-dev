# Reference Papaer: https://arxiv.org/abs/1911.06475


import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# hydra
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

# 내부 모듈
from custom_utils.custom_metrics import AUROCMetricReporter
from custom_utils.custom_reporter import *
from custom_utils.transform import create_transforms
from data_loader.data_loader import CXRDataset


# best_val_roc_auc = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def c_val(hydra_cfg, dataloader, model, loss_f, num_classes):
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

        auroc_reporter = AUROCMetricReporter(
            hydra_cfg=hydra_cfg, preds=val_pred, targets=val_true, mode="train"
        )
        val_roc_auc = auroc_reporter.get_macro_auroc_score()
        val_loss /= num_batches

    return val_roc_auc


def c_trainval(hydra_cfg, logger, best_val_roc_auc=0):

    num_classes = hydra_cfg.num_classes


    train_dataset = CXRDataset(
        "train",
        **hydra_cfg.Dataset,
        transforms=create_transforms(hydra_cfg.Dataset, "train", conditional=True),
        conditional_train=True,
    )
    val_dataset = CXRDataset(
        "valid",
        **hydra_cfg.Dataset,
        transforms=create_transforms(hydra_cfg.Dataset, "valid"),
        conditional_train=False,
    )
    loader = DataLoader(train_dataset, batch_size=32, **hydra_cfg.Dataloader.train)
    val_loader = DataLoader(val_dataset, batch_size=32, **hydra_cfg.Dataloader.valid)
    size = len(loader.dataset)

    model = instantiate(hydra_cfg.model)
    model = model.to(device)

    loss_f = nn.BCEWithLogitsLoss()  # hard_coded from the ref paper
    optimizer = optim.Adam(          # hard_coded from the ref paper 
        model.parameters(), lr=0.0001, weight_decay=1e-4
    )  

    print("#############################################")
    print("########## Conditional-Train Start ##########")
    print("#############################################")

    # logger.info('Starting Conditional-Train...')
    for epoch in range(5):

        model.train()
        for batch, (X, y) in enumerate(loader):
            data_X, label_y = X.to(device), y.to(device)

            pred = model(data_X)
            pred = torch.sigmoid(pred)  # for multi-label
            loss = loss_f(pred, label_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 500 == 0:
                loss, current = loss.item(), batch * len(data_X)
                val_roc_auc = c_val(hydra_cfg, val_loader, model, loss_f, num_classes)

                if best_val_roc_auc < val_roc_auc:
                    best_val_roc_auc = val_roc_auc
                    best_model_state = model.state_dict()

                # report_metrics(val_pred, val_true, print_classification_result=False)
                print(
                    f"val_roc_auc: {val_roc_auc:>4f}, "
                    f"Best_val_score: {best_val_roc_auc:>4f}, "
                    f"epoch: {epoch+1}, "
                    f"Batch ID: {batch}[{current:>5d}/{size:>5d}]"
                )

            model.train()

    print("#############################################")
    print("########### Conditional-Train End ###########")
    print("#############################################")
    # logger.info('Ending Conditional-Train...')

    return best_model_state
