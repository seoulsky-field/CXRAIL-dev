import os
import torch
import torch.nn as nn
import multiprocessing
from functools import partial
import wandb

# ray
import ray
from ray import air, tune
from ray.tune import Trainable, run
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import ScalingConfig
from ray.air.integrations.wandb import setup_wandb

# reporter
from ray.tune import CLIReporter
from ray.tune.experiment import Trial
from typing import Any, Callable, Dict, List, Optional, Union
from custom_utils.custom_reporter import TrialTerminationReporter

# hydra
import hydra
from omegaconf import DictConfig, OmegaConf, errors
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

# 내부모듈
from train import trainval
from custom_utils.print_tree import print_config_tree
from custom_utils.seed import seed_everything


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
working_dir = os.getcwd()


def default(hydra_cfg, hparam):
    config = hydra_cfg
    print("working dir: " + os.getcwd())
    trainval(config, hydra_cfg, hparam, best_val_roc_auc=0)


def raytune(hydra_cfg, hparam):
    print("working dir: " + os.getcwd())

    param_space = OmegaConf.to_container(
        instantiate(hydra_cfg.hparams_search.param_space)
    )
    tune_config = instantiate(hydra_cfg.hparams_search.tune_config)
    run_config = instantiate(hydra_cfg.hparams_search.run_config)

    # execute run
    tuner = tune.Tuner(
        trainable=tune.with_resources(
            partial(trainval, hydra_cfg=hydra_cfg, hparam=hparam, best_val_roc_auc=0),
            {
                "cpu": int(round(multiprocessing.cpu_count() / 2)),
                "gpu": int(torch.cuda.device_count()),
            },
        ),
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )
    analysis = tuner.fit()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(hydra_cfg: DictConfig):
    seed_everything(hydra_cfg.seed)
    if hydra_cfg.get("print_config"):
        # log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(hydra_cfg, resolve=True, save_to_file=True)

    # search
    hparam = hydra_cfg.hparams_search.name
    print("hyperparameter search:", hparam)
    if hparam == "raytune":
        raytune(hydra_cfg, hparam)
    else:
        default(hydra_cfg, hparam)

    os.chdir(working_dir)


if __name__ == "__main__":

    main()
