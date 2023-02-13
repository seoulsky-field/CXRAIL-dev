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


def predict(hydra_cfg, model, test_loader, train_columns):
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

        auroc_reporter = AUROCMetricReporter(
            hydra_cfg=hydra_cfg,
            preds=test_pred,
            targets=test_true,
            target_columns=train_columns,
            mode="test",
        )

        micro_auroc_score = round(auroc_reporter.get_micro_auroc_score(), 4)
        macro_auroc_score = round(auroc_reporter.get_macro_auroc_score(), 4)
        class_auroc_score = auroc_reporter.get_class_auroc_score()

        rich.print(f"Micro AUROC: {micro_auroc_score}")
        rich.print(f"Macro AUROC: {macro_auroc_score}")
        rich.print(f"Class AUROC: {class_auroc_score}")

        # train_columns = list(hydra_cfg[dataset_name].train_cols)

        rich.print("")
        for idx in range(hydra_cfg.num_classes):
            fpr, tpr, thresholds, ix = auroc_reporter.get_auroc_details(
                test_true[:, idx], test_pred[:, idx]
            )
            rich.print(
                f"{train_columns[idx]} Class AUROC Informations || TPR: {tpr[ix]:>.4f}, FPR: {fpr[ix]:>.4f}, Best Threshold: {thresholds[ix]:>.4f}"
            )
        rich.print("")

        if hydra_cfg.save_auroc_plot:
            plot_save_dir = os.path.join(hydra_cfg.log_dir, "images")

            if not os.path.exists(plot_save_dir):
                os.mkdir(plot_save_dir)

            for idx in range(hydra_cfg.num_classes):
                auroc_reporter.plot_class_auroc_details(
                    targets=test_true[:, idx],
                    preds=test_pred[:, idx],
                    col_name=train_columns[idx],
                )
            auroc_reporter.plot_overlap_roc_curve()

    return micro_auroc_score, macro_auroc_score, class_auroc_score


def load_model(hydra_cfg, check_point_path):
    check_point = torch.load(check_point_path)
    train_info = load_hydra_config(check_point_path)

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

    # load config
    ## define path
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

    ## load yaml
    with open(config_yaml) as f:
        config_ = yaml.load(f, Loader=yaml.FullLoader)
    with open(hydra_yaml) as f:
        hydra_ = yaml.load(f, Loader=yaml.FullLoader)
    with open(override_yaml) as f:
        override_ = yaml.load(f, Loader=yaml.FullLoader)

    hydra_config = hydra_["hydra"]["runtime"]["choices"]
    #multirun_config = "\n".join(s for s in override_)

    report_configs = {
        "epoch": config_.get("epochs", "None"),
        "hparams_search": hydra_config.get("hparams_search", "None"),
        "Model": hydra_config.get("model", "None"),
        "loss_func": hydra_config.get("loss", "None"),
        "Optimizer": hydra_config.get("optimizer", "None"),
        "Dataset": hydra_config.get("Dataset", "None"),
        "Multirun": dict(),
        "Override": dict(),
    }

    # override check
    for ov in override_:
        print(ov)

        #override_dic = {'Dataset' : {'train_size': 1000, 'root_path': '/home'}}
        #override_dic = {'Dataset' : ['train_size=1000', 'root_path=/home']}

        if '.' in ov:
            target = ov.split('.')[0]
            parameter = ov.split('.')[1].split('=')[0]
            value = ov.split('.')[1].split('=')[1]
            print(f"\ntarget: {target}\nparameter: {parameter}\nvalue: {value}\n") 

            print(report_configs[target])
            
            # report_configs[target] = []
            try:
                report_configs['Override'][target][parameter] = value
            
            except KeyError:
                report_configs['Override'][target] = {}
                report_configs['Override'][target][parameter] = value

            #[target].append(ov.split('.')[1])

            #report_configs["Multirun"].remove(ov)
            #x = car.setdefault("trim", "TLX")
            #[parameter] = value
            #.setdefault(target, )
            #report_configs["Override"][target]e# += {parameter: value}


            print(report_configs)
        
        else:
            report_configs["Multirun"].setdefault(ov.split('=')[0], ov.split('=')[1])
    return report_configs


def save_result_csv(report_configs, hydra_cfg, train_columns):
    columns = [
        "log_dir",
        "test_roc_auc",
        "micro_roc_auc",
    ]
    columns += train_columns
    columns += [
        "Dataset",
        "Model",
        "Optimizer",
        "loss_func",
        "hparams_search",
        "Multirun",
        "epoch",
    ]

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

        # load test dataset
        test_dataset = load_dataset(hydra_cfg, check_point_path)
        test_loader = DataLoader(test_dataset, **hydra_cfg.Dataloader.test)

        # load configs from train log
        model, report_configs = load_model(hydra_cfg, check_point_path)

        dataset_name = report_configs["Dataset"]
        train_columns = list(hydra_cfg[dataset_name].train_cols)

        print(train_columns)
        
        ### inference ### 
        micro_auroc_score, macro_auroc_score, class_auroc_score = predict(
            hydra_cfg, model, test_loader, train_columns
        )
        test_score.append(macro_auroc_score)

        # save additional configs
        report_configs["log_dir"] = log_dir
        report_configs["test_roc_auc"] = macro_auroc_score
        report_configs["micro_roc_auc"] = micro_auroc_score

        for class_name, score in zip(train_columns, class_auroc_score):
            report_configs[class_name] = score

        report_configs["log_dir"] = log_dir
        report_configs_dict.append(report_configs)

        # score logging
        logger.info("%s: %s", log_dir, macro_auroc_score)

    result_df = save_result_csv(report_configs_dict, hydra_cfg, train_columns)

    return test_score


if __name__ == "__main__":
    test_score = main()
