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


def predict(hydra_cfg, model, test_loader):
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
    train_info = load_hydra_config(check_point_path)

    # except BaseException:
    #     train_info = {
    #         'Dataset': check_point.get("Dataset", None),
    #         'Model': check_point.get("Model", None),
    #         'Optimizer': check_point.get("Optimizer", None),
    #         'loss_func': check_point.get("loss_func", None),
    #     }

    model_name = train_info["Model"]
    model_state = check_point.get("model_state_dict", None)

    model_config = hydra_cfg[model_name]
    model = instantiate(model_config)
    model = model.to(device)

    return model, train_info


def load_dataset(hydra_cfg, check_point_path):
    train_info = load_hydra_config(check_point_path)
    data_name = train_info["Dataset"]
    Dataset_cfg = hydra_cfg[data_name]

    test_dataset = CXRDataset(
        "test",
        **Dataset_cfg,
        transforms=create_transforms(Dataset_cfg, "valid"),
        conditional_train=False,
    )
    return test_dataset


def load_hydra_config(check_point_path):

    if "trainval" in check_point_path:
        # ray logging file structure
        config_yaml = check_point_path.split("trainval")[0] + ".hydra/config.yaml"
        hydra_yaml = check_point_path.split("trainval")[0] + ".hydra/hydra.yaml"
        override_yaml = check_point_path.split("trainval")[0] + ".hydra/overrides.yaml"
    else:
        config_yaml = check_point_path.split("best_saved.pth")[0] + ".hydra/config.yaml"
        hydra_yaml = check_point_path.split("best_saved.pth")[0] + ".hydra/hydra.yaml"
        override_yaml = (
            check_point_path.split("best_saved.pth")[0] + ".hydra/overrides.yaml"
        )

    with open(config_yaml) as f:
        config_ = yaml.load(f, Loader=yaml.FullLoader)
    with open(hydra_yaml) as f:
        hydra_ = yaml.load(f, Loader=yaml.FullLoader)
    with open(override_yaml) as f:
        override_ = yaml.load(f, Loader=yaml.FullLoader)

    hydra_config = hydra_["hydra"]["runtime"]["choices"]
    multirun_config = "\n".join(s for s in override_)

    report_configs = {
        "epoch": config_.get("epochs", "None"),
        "hparams_search": hydra_config.get("hparams_search", "None"),
        "Model": hydra_config.get("model", "None"),
        "loss_func": hydra_config.get("loss", "None"),
        "Optimizer": hydra_config.get("optimizer", "None"),
        "Dataset": hydra_config.get("Dataset", "None"),
        "Multirun": multirun_config,
    }
    return report_configs


def save_result_csv(report_configs, hydra_cfg):
    # try:
    columns = [
        "log_dir",
        "test_roc_auc",
        "Dataset",
        "Model",
        "Optimizer",
        "loss_func",
        "hparams_search",
        "Multirun",
        "epoch",
    ]
    # except Exception:
    #     # for user who doesn't have hydra config
    #     colums = list(report_configs.keys())

    result_df = pd.DataFrame(report_configs, columns=columns)
    result_df.set_index("log_dir", inplace=True)

    save_path = os.path.join(hydra_cfg.log_dir, "inference_result.csv")
    result_df.to_csv(save_path, index=True)
    print(result_df)
    return result_df


@hydra.main(version_base=None, config_path="config", config_name="test.yaml")
def main(hydra_cfg: DictConfig):
    seed_everything(hydra_cfg.seed)
    custom_logger = Logger(mode="test", filePath=hydra_cfg.log_dir)
    logger = custom_logger.initLogger()

    check_point_yaml = os.path.join(
        os.getcwd(), "./logs/checkpoints/checkpoint_path.yaml"
    )

    with open(check_point_yaml) as f:
        check_point_paths = yaml.load(f, Loader=yaml.FullLoader)

    test_score = []
    report_configs_dict = []
    for log_dir, check_point_path in check_point_paths.items():

        test_dataset = load_dataset(hydra_cfg, check_point_path)
        test_loader = DataLoader(test_dataset, **hydra_cfg.Dataloader.test)
        model, report_configs = load_model(hydra_cfg, check_point_path)

        # test
        test_roc_auc = predict(hydra_cfg, model, test_loader)
        test_score.append(test_roc_auc)

        report_configs["log_dir"] = log_dir
        report_configs["test_roc_auc"] = test_roc_auc
        report_configs_dict.append(report_configs)

        # score logging
        logger.info("%s: %s", log_dir, test_roc_auc)

    result_df = save_result_csv(report_configs_dict, hydra_cfg)
    return test_score


if __name__ == "__main__":
    test_score = main()
