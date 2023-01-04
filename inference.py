import os
import logging
import numpy as np
import pandas as pd
import random
import wandb
import pprint
from tqdm import tqdm
from libauc.metrics import auc_roc_score
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from typing import Optional

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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

# rich
import rich
from rich.progress import track

# 내부 모듈
from custom_utils.custom_metrics import TestMetricsReporter, AUROCMetricReporter
from custom_utils.custom_reporter import *
from custom_utils.transform import create_transforms
from data_loader.data_loader import CXRDataset
from custom_utils.print_tree import print_config_tree
from custom_utils.seed import seed_everything
from custom_utils.custom_logger import Logger


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict(hydra_cfg, model, test_loader):  # , loss_f, optimizer):
    model.eval()
    with torch.no_grad():
        test_pred = []
        test_true = []

        for _, (X, y) in track(
            enumerate(test_loader),
            description="[bold blue]Inference: ",
            total=len(test_loader),
        ):

            X = X.to(device, dtype=torch.float)
            y_pred = model(X)
            y_pred = torch.sigmoid(y_pred)

            test_pred.append(y_pred.cpu().detach().numpy())
            test_true.append(y.cpu().numpy())

        test_pred = np.concatenate(test_pred)
        test_true = np.concatenate(test_true)

        auroc_reporter = AUROCMetricReporter(preds=test_pred, targets=test_true)

        rich.print(f"micro auc: {auroc_reporter.get_micro_auroc_score():>.4f}")
        rich.print(f"macro auc: {auroc_reporter.get_macro_auroc_score():>.4f}")
        rich.print(f"class aucs: {auroc_reporter.get_class_auroc_score()}")

        for idx in range(hydra_cfg.num_classes):
            fpr, tpr, thresholds, ix = auroc_reporter.get_auroc_details(
                test_true[:, idx], test_pred[:, idx]
            )
            rich.print(
                f"auroc class{idx}: TPR: {tpr[ix]:>.4f}, FPR: {fpr[ix]:>.4f}, Best Threshold: {thresholds[ix]:>.4f}"
            )

        test_roc_auc = auroc_reporter.get_macro_auroc_score()

    return test_roc_auc


def load_model(hydra_cfg, check_point_path):
    check_point = torch.load(check_point_path)
    model_name = check_point.get("model", None)
    model_state = check_point.get("model_state_dict", None)

    try:
        model = instantiate(hydra_cfg.models.resnet)
        model.load_state_dict(model_state)
    except BaseException:
        model = instantiate(hydra_cfg.models.densenet)
        model.load_state_dict(model_state)

    model = model.to(device)

    return model, model_name


@hydra.main(version_base=None, config_path="config", config_name="test.yaml")
def main(hydra_cfg: DictConfig):
    seed_everything(hydra_cfg.seed)
    custom_logger = Logger(mode="test", filePath=hydra_cfg.log_dir)
    logger = custom_logger.initLogger()

    test_dataset = CXRDataset(
        "test",
        **hydra_cfg.Dataset,
        transforms=create_transforms(hydra_cfg, "valid"),
        conditional_train=False,
    )
    test_loader = DataLoader(test_dataset, **hydra_cfg.Dataloader.test)
    check_point_yaml = os.path.join(
        os.getcwd(), "./logs/checkpoints/checkpoint_path.yaml"
    )

    with open(check_point_yaml) as f:
        check_point_paths = yaml.load(f, Loader=yaml.FullLoader)

    additional_info = []
    test_score = []
    for log_dir, check_point_path in check_point_paths.items():
        model, model_name = load_model(hydra_cfg, check_point_path)
        # test
        test_roc_auc = predict(hydra_cfg, model, test_loader)
        test_score.append(test_roc_auc)
        additional_info.append(model_name)

        logger.info("%s: %s", log_dir, test_roc_auc)
    # for score, name in zip(test_score, additional_info):
    #     print(name, score)

    return test_score


if __name__ == "__main__":

    test_score = main()
