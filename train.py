import os

# import logging
import numpy as np
import pandas as pd
import random
import pprint
import time
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from libauc.metrics import auc_roc_score
from torchmetrics.classification import MultilabelAUROC

# hydra
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

# ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.air.config import ScalingConfig
from ray.air.integrations.wandb import setup_wandb

# wandb
import wandb

# rich
from rich.progress import track
from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.progress import DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from rich.rule import Rule

# 내부 모듈
from custom_utils.custom_metrics import *
from custom_utils.custom_metrics import report_metrics
from custom_utils.custom_reporter import *
from custom_utils.transform import create_transforms
from data_loader.data_loader import CXRDataset
from custom_utils.print_tree import print_config_tree
from custom_utils.seed import seed_everything
from custom_utils.conditional_train import c_trainval
from custom_utils.custom_logger import TrainerLogger
from setting import LoadExperimentSetting


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()


def train(
    hydra_cfg,
    dataloader,
    val_loader,
    model,
    loss_f,
    optimizer,
    epoch,
    best_val_roc_auc,
    val_loss,
    val_roc_auc,
    hparam,
    ckpt_path,
    epoch_progress,
    epoch_task_id,
    valid_progress,
    stop_patience,
    logger,
):
    use_amp = hydra_cfg.get("use_amp")

    size = len(dataloader.dataset)
    model.train()
    # If you want to check the one epoch time, use this code.
    # training_start_time = time.time()

    for batch, (X, y) in enumerate(dataloader):
        data_X, label_y = X.to(device), y.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.autocast(device_type=str(device)):
                pred = model(data_X)
                pred = torch.sigmoid(pred)  # for multi-label
                loss = loss_f(pred, label_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(data_X)
            pred = torch.sigmoid(pred)  # for multi-label
            loss = loss_f(pred, label_y)

            loss.backward()
            optimizer.step()

        if batch % 500 == 0:
            loss, current = loss.item(), batch * len(data_X)
            val_loss, val_roc_auc, val_pred, val_true = val(
                val_loader, model, loss_f, valid_progress
            )

            if best_val_roc_auc < val_roc_auc:
                stop_patience = 0
                best_val_roc_auc = val_roc_auc
                save_dict = {
                    "Dataset": hydra_cfg.get("Dataset")["dataset"],
                    "Optimizer": hydra_cfg.get("optimizer")["_target_"].split(".")[-1],
                    "loss_func": hydra_cfg.get("loss")["_target_"].split(".")[-1],
                    "Model": hydra_cfg.get("model")["model_name"],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }

                torch.save(save_dict, ckpt_path)
                print("Best model saved.")
            else:
                stop_patience += 1
                if stop_patience == hydra_cfg.stop_patience:
                    print(
                        f"Stop patience reached: {stop_patience}/{hydra_cfg.stop_patience}"
                    )
                    return best_val_roc_auc, val_loss, val_roc_auc, stop_patience

            report_metrics(val_pred, val_true, print_classification_result=False)
            print(
                f"loss: {loss:>7f}, "
                f"val_loss = {val_loss:>7f}, "
                f"val_roc_auc: {val_roc_auc:>4f}, "
                f"Best_val_score: {best_val_roc_auc:>4f}, "
                f"epoch: {epoch+1}, "
                f"Early stop patience: {stop_patience}/{hydra_cfg.stop_patience}, "
                f"Batch ID: {batch}[{current:>5d}/{size:>5d}]"
            )

            result_metrics = {
                "epoch": epoch + 1,
                "Batch_ID": batch,
                "loss": loss,
                "val_loss": val_loss,
                "val_roc_auc": val_roc_auc,
                "best_val_score": best_val_roc_auc,
            }

            # log (default: wandB only, ray included: wandb, ray reporter)
            logger.info(result_metrics)
            if hydra_cfg.get("logging") is not None:
                wandb.log(result_metrics)
            if hparam == "raytune":
                result_metrics["progress_of_epoch"] = f"{100*current/size:.1f} %"
                session.report(metrics=result_metrics)

        epoch_progress.update(
            epoch_task_id,
            epoch=epoch + 1,
            batch=batch + 1,
            loss=loss,
            val_loss=val_loss,
            val_auc=val_roc_auc,
            best_val_auc=best_val_roc_auc,
            advance=1,
        )

        model.train()

    # If you want to check the one epoch time, use this code.
    # print(f"One Epoch Finished. Time(s): {(time.time()-training_start_time):>5f}")

    return best_val_roc_auc, val_loss, val_roc_auc, stop_patience


def val(dataloader, model, loss_f, valid_progress):
    num_batches = len(dataloader)
    val_loss = 0
    valid_task_id = valid_progress.add_task(
        "", valid_batch=1, valid_loader_size=num_batches
    )

    model.eval()
    with torch.no_grad():
        val_pred = []
        val_true = []

        for batch, (X, y) in enumerate(dataloader):
            valid_progress.update(valid_task_id, valid_batch=batch)
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
        val_roc_auc = float(torch.mean(auc_roc_scores))
        val_loss /= num_batches

    valid_progress.update(valid_task_id, visible=False)

    return val_loss, val_roc_auc, val_pred, val_true

def load_dataset(hydra_cfg):
    # Dataset
    train_dataset = CXRDataset("train", **hydra_cfg.Dataset,)
    val_dataset = CXRDataset("valid", **hydra_cfg.Dataset,)

    train_loader = instantiate(hydra_cfg.DataLoader.train, dataset=train_dataset)
    val_loader = instantiate(hydra_cfg.DataLoader.train, dataset=val_dataset)

    return train_loader, val_loader


def trainval(config, hydra_cfg, hparam, best_val_roc_auc=0):
    
    # conditional training
    if hydra_cfg.get("conditional_train") is not None:
        best_model_state = c_trainval(hydra_cfg, best_val_roc_auc=0)
        model = instantiate(hydra_cfg.model)  # load best model
        model.load_state_dict(best_model_state)
        model.reset_classifier(num_classes=5)
    else:
        model = instantiate(hydra_cfg.model)

    model = nn.DataParallel(model)  # Multi-GPU
    model = model.to(device)

    # experiment setting
    loader = LoadExperimentSetting(hydra_cfg, model)
    train_loader, val_loader = loader.set_dataloader()
    optimizer = loader.set_optimizer(mode='soft')
    loss_f = loader.set_criterion()
    search_space = loader.return_search_space()


    # Initialize WandB
    if hydra_cfg.get("logging") is not None:
        wandb_cfg = OmegaConf.to_container(hydra_cfg.logging.config, resolve=True)
    if hparam == "none":
        ckpt_path = os.path.join(hydra_cfg.save_dir, hydra_cfg.ckpt_name)
        if hydra_cfg.get("logging") is not None:
            wandb.init(**hydra_cfg.logging.setup, config=wandb_cfg)

    elif hparam == "raytune":
        ckpt_path = hydra_cfg.ckpt_name
        logdir = session.get_trial_dir()
        if hydra_cfg.get("logging") is not None:
            with open(logdir + "params.pkl", "rb") as f:
                params = pickle.load(f)
            wandb_cfg = {key: params.get(key, wandb_cfg[key]) for key in wandb_cfg}

            wandb_r = setup_wandb(
                project=hydra_cfg.project_name,
                dir=session.get_trial_dir(),
                config=wandb_cfg,
            )

    custom_logger = TrainerLogger(filePath=os.path.dirname(ckpt_path))
    logger = custom_logger.init_trainer_logger()

    

    # rich
    # label_progress = Progress(
    #     TimeElapsedColumn(),
    #     TextColumn('{task.description}')
    # )

    epoch_progress = Progress(
        TextColumn("[bold blue] Training epoch {task.fields[epoch]}: "),
        BarColumn(style="magenta"),
        TextColumn("[bold] {task.fields[batch]} / {task.fields[loader_size]}"),
        TimeRemainingColumn(),
        TextColumn(
            "loss: {task.fields[loss]:>5f} | val/loss: {task.fields[val_loss]:>5f} | val/auc: {task.fields[val_auc]:>5f} | best_val/auc: {task.fields[best_val_auc]:>5f}"
        ),
    )

    valid_progress = Progress(
        TextColumn("[bold blue]   Validation: "),
        BarColumn(),
        TextColumn(
            "[bold] {task.fields[valid_batch]} / {task.fields[valid_loader_size]}"
        ),
        TimeRemainingColumn(),
    )

    group = Group(
        # label_progress,
        Rule(style="#AAAAAA"),
        epoch_progress,
        valid_progress,
    )

    live = Live(group)
    val_loss = 0
    val_roc_auc = 0
    best_val_roc_auc = 0
    loader_size = len(train_loader)

    # train
    with live:
        epoch_id = epoch_progress.add_task(
            "",
            epoch=1,
            batch=1,
            loader_size=loader_size,
            loss=0,
            val_loss=val_loss,
            val_auc=val_roc_auc,
            best_val_auc=best_val_roc_auc,
            total=loader_size,
        )
        stop_patience = 0
        for epoch in range(hydra_cfg.epochs):

            epoch_progress.reset(epoch_id)

            best_val_roc_auc, val_loss, val_roc_auc, stop_patience = train(
                hydra_cfg,
                train_loader,
                val_loader,
                model,
                loss_f,
                optimizer,
                epoch,
                best_val_roc_auc,
                val_loss,
                val_roc_auc,
                hparam,
                ckpt_path,
                epoch_progress,
                epoch_id,
                valid_progress,
                stop_patience,
                logger,
            )
            if stop_patience == hydra_cfg.stop_patience:
                print("Stop patience reached, train loop terminating.")
                logger.info(
                    f"Stop patience reached: {stop_patience}/{hydra_cfg.stop_patience}, train loop terminating."
                )
                break

    if hydra_cfg.get("logging") is not None:
        wandb.finish()

    return search_space

# @hydra.main(version_base=None, config_path="config", config_name="config")
# def main(hydra_cfg: DictConfig):
#     config = hydra_cfg
#     seed_everything(hydra_cfg.seed)

#     if hydra_cfg.get("print_config"):
#         # log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
#         print_config_tree(hydra_cfg, resolve=True, save_to_file=True)

#     trainval(config, hydra_cfg)


# if __name__ == "__main__":
#     main()
