
import os
import logging
import numpy as np
import pandas as pd
import random
import wandb
import pprint
from tqdm import tqdm
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
from data_loader.data_loader import *
from custom_utils.print_tree import print_config_tree
from custom_utils.seed import seed_everything


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict(hydra_cfg, model, test_loader):#, loss_f, optimizer):
    model.eval()
    with torch.no_grad():
        test_pred = []
        test_true = []

        for batch_id, (X, y) in enumerate(test_loader):
            assert len(test_loader) > 0
            
            X = X.to(device, dtype=torch.float)
            y_pred = model(X)
            y_pred = torch.sigmoid(y_pred)
            test_pred.append(y_pred.cpu().detach().numpy())
            test_true.append(y.cpu().numpy())
        test_pred = np.concatenate(test_pred)
        test_true = np.concatenate(test_true)

        test_pred_tensor = torch.from_numpy(test_pred)
        test_true_tensor = torch.from_numpy(test_true)

        auroc = MultilabelAUROC(num_labels=5, average="macro", thresholds=None)
        auc_roc_scores = auroc(test_pred_tensor, test_true_tensor)
        test_roc_auc = torch.mean(auc_roc_scores).numpy()

        print(f"Test accuracy: {test_roc_auc:>4f}")

    return test_roc_auc
  

@hydra.main(
    version_base = None, 
    config_path='config', 
    config_name = 'config'
)  
def main(hydra_cfg:DictConfig):
    seed_everything(hydra_cfg.seed)

    test_dataset = CXRDataset('test', **hydra_cfg.Dataset, transforms=create_transforms(hydra_cfg, 'valid'), conditional_train=False,)
    test_loader = DataLoader(test_dataset, **hydra_cfg.Dataloader.test)

    model = instantiate(hydra_cfg.model)
    model = model.to(device)
    #check_point_path = os.path.join(HydraConfig.get().run.dir, 'best_saved.pt')
    check_point_path = '/home/CheXpert_code/ynkng/CXRAIL-dev/logs/2022-12-27_06-10-27/Dataset.train_size=3000/trainval_2022-12-27_06-10-27/trainval_3941525e_2_batch_size=64,lr=0.0003_2022-12-27_06-11-22/best_saved.pth'
    check_point = torch.load(check_point_path)
    model.load_state_dict(check_point)

    # test
    test_roc_auc = predict(hydra_cfg, model, test_loader)


if __name__ == '__main__':

    main()