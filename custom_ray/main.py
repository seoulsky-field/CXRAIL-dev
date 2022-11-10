'''
주요 수정 사항 : run, tuner 중 run으로 통일 (추후에 변경 가능)
'''

import os
import torch
import torch.nn as nn
import multiprocessing
from functools import partial

# ray
import ray
from ray import air, tune
from ray.tune import Trainable, run
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import ScalingConfig

# reporter
from ray.tune import CLIReporter
from ray.tune.experiment import Trial
from typing import Any, Callable, Dict, List, Optional, Union
from custom_utils.custom_reporter import TrialTerminationReporter

# hydra
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from train import trainval

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@hydra.main(
    version_base = None, 
    config_path='config', 
    config_name = 'config'
)
def main(cfg: DictConfig):
    param_space = OmegaConf.to_container(instantiate(cfg.ray.param_space))

    # get hydra config
    hydra_config = hydra.core.hydra_config.HydraConfig.get()

    #set config #1: default
    # search_alg = instantiate(cfg.ray.search_alg, space=param_space)
    # scheduler = instantiate(cfg.ray.scheduler)
    # reporter = TrialTerminationReporter(
    #     parameter_columns=["lr"],
    #     metric_columns=["loss", "val_loss", "val_score",  "current_epoch",  "progress_of_epoch"])

    #set config #2: use hydra
    scheduler = instantiate(cfg.ray.scheduler)
    search_alg = instantiate(cfg.ray.search_alg, space=param_space)
    reporter = CLIReporter(
        parameter_columns=['lr', 'weight_decay'],
        metric_columns=['loss', 'val_loss', 'val_score', 'best_val_roc_auc', 'current_epoch', 'progress_of_epoch']
    )# instantiate(cfg.ray.reporter)
    scheduler = instantiate(cfg.ray.scheduler)

    # execute run
    result = tune.run(
        partial(trainval, hydra_cfg=cfg),
        config = param_space,
        num_samples = cfg.ray.num_samples,
        scheduler = scheduler,
        search_alg = search_alg, 
        progress_reporter=reporter,
        resources_per_trial={
                'cpu': int(round(multiprocessing.cpu_count()/2)), 
                'gpu': int(torch.cuda.device_count()),
                },
        local_dir=os.path.join(
            hydra_config.sweep.dir,
            hydra_config.sweep.subdir),
        # name=cfg.ray.name,
        )



if __name__ == "__main__":
    main()